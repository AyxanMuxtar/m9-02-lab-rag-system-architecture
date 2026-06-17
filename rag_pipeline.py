"""
RAG Pipeline — Retrieve → Augment → Generate

Uses:
  - Ollama (local) with gemma4:e4b for generation
  - Ollama nomic-embed-text for embeddings
  - ChromaDB as the vector store

NOTE: We do NOT use the Gemini API at all. Everything runs locally via Ollama.

Why chunking isn't needed here:
  Each entry in knowledge_base.json is already a single, self-contained passage
  (~1-3 sentences). In a real scenario with full documents, we would need to
  split them into smaller chunks (e.g., 200-500 tokens) so that each chunk fits
  within the embedding model's context window and the retriever can return
  focused, relevant passages rather than entire documents.
"""

import json
import requests
import chromadb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
GENERATION_MODEL = "gemma4:e4b"
EMBEDDING_MODEL = "nomic-embed-text"
KNOWLEDGE_BASE_PATH = "knowledge_base.json"
COLLECTION_NAME = "knowledge_base"
TOP_K = 3  # number of passages to retrieve

# ---------------------------------------------------------------------------
# 1. EMBEDDING HELPER — calls Ollama's /api/embed endpoint
# ---------------------------------------------------------------------------

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Get embeddings for a list of texts using Ollama's embedding model."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": texts},
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def get_embedding(text: str) -> list[float]:
    """Get embedding for a single text."""
    return get_embeddings([text])[0]


# ---------------------------------------------------------------------------
# 2. INDEXING — load knowledge base and store in ChromaDB
# ---------------------------------------------------------------------------

def load_knowledge_base(path: str = KNOWLEDGE_BASE_PATH) -> list[dict]:
    """Load passages from the JSON knowledge base."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(entries: list[dict], client: chromadb.ClientAPI) -> chromadb.Collection:
    """
    Embed each passage and add it to a Chroma collection with its source
    metadata.  Returns the collection handle.
    """
    # Delete existing collection if it exists, so re-runs are idempotent
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Prepare data
    ids = [entry["id"] for entry in entries]
    documents = [entry["text"] for entry in entries]
    metadatas = [{"source": entry["source"]} for entry in entries]

    # Get embeddings via Ollama
    embeddings = get_embeddings(documents)

    # Add to ChromaDB
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"✅ Indexed {len(entries)} passages into ChromaDB collection '{COLLECTION_NAME}'")
    return collection


# ---------------------------------------------------------------------------
# 3. QUERY — retrieve top-k passages and assemble a grounded prompt
# ---------------------------------------------------------------------------

def retrieve(question: str, collection: chromadb.Collection, top_k: int = TOP_K) -> dict:
    """
    Retrieve the top-k most relevant passages for a question.
    Returns the ChromaDB query result dict.
    """
    query_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return results


def build_prompt(question: str, results: dict) -> str:
    """
    Assemble the grounded RAG prompt with:
      - An instruction to answer ONLY from context
      - Retrieved passages tagged with their source
      - The user question
    """
    passages_block = ""
    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0]), start=1
    ):
        passages_block += f"[Source: {meta['source']}]\n{doc}\n\n"

    prompt = f"""You are a helpful assistant. Answer the question ONLY using the context passages below.
If the answer is not contained in the passages, respond with: "I don't know — the knowledge base does not contain this information."

For every fact in your answer, cite the source in parentheses, e.g. (source: handbook.md).

--- CONTEXT PASSAGES ---
{passages_block}
--- END OF CONTEXT ---

Question: {question}

Answer:"""

    return prompt


# ---------------------------------------------------------------------------
# 4. GENERATE — send prompt to Ollama's gemma4:e4b model
# ---------------------------------------------------------------------------

def generate_answer(prompt: str) -> str:
    """Send the assembled prompt to gemma4:e4b via Ollama and return the answer."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": GENERATION_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,      # low temperature for factual answers
                "num_predict": 512,      # max tokens to generate
            },
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()


# ---------------------------------------------------------------------------
# 5. RAG PIPELINE — end-to-end function
# ---------------------------------------------------------------------------

def rag_query(question: str, collection: chromadb.Collection, top_k: int = TOP_K) -> dict:
    """
    Full RAG pipeline:
      1. Retrieve top-k passages
      2. Build grounded prompt
      3. Generate cited answer
    Returns dict with question, retrieved sources, and answer.
    """
    # Retrieve
    results = retrieve(question, collection, top_k=top_k)

    # Build prompt
    prompt = build_prompt(question, results)

    # Generate
    answer = generate_answer(prompt)

    # Collect sources for reporting
    sources = [
        {"source": meta["source"], "text": doc[:80] + "..."}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]

    return {
        "question": question,
        "top_k": top_k,
        "retrieved_sources": sources,
        "answer": answer,
    }


def print_result(result: dict) -> None:
    """Pretty-print a RAG result."""
    print("=" * 70)
    print(f"❓ QUESTION: {result['question']}")
    print(f"   (top_k = {result['top_k']})")
    print("-" * 70)
    print("📚 RETRIEVED SOURCES:")
    for i, src in enumerate(result["retrieved_sources"], 1):
        print(f"   {i}. [{src['source']}] {src['text']}")
    print("-" * 70)
    print(f"💬 ANSWER:\n{result['answer']}")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Setup ---
    print("🔧 Initializing ChromaDB (in-memory) and indexing knowledge base...\n")
    chroma_client = chromadb.Client()  # ephemeral in-memory client
    entries = load_knowledge_base()
    collection = build_index(entries, chroma_client)
    print()

    # --- Required questions ---
    questions = [
        "How long do I have to get a full refund?",           # answerable
        "How do I reset my password?",                         # answerable
        "What is the company's stock price today?",            # NOT in KB — must decline
    ]

    results = []
    for q in questions:
        result = rag_query(q, collection, top_k=TOP_K)
        print_result(result)
        results.append(result)

    # --- Optional stretch: top-1 vs top-3 comparison ---
    print("\n" + "=" * 70)
    print("🔬 STRETCH: Comparing top-1 vs top-3 retrieval")
    print("=" * 70 + "\n")

    stretch_question = "How long do I have to get a full refund?"

    result_top1 = rag_query(stretch_question, collection, top_k=1)
    print_result(result_top1)

    print("""
📝 TRADE-OFF ANALYSIS (top-1 vs top-3):
   With top-1, the model gets only the single most relevant passage, which
   may miss complementary information (e.g., details about store credit
   after 30 days). With top-3, the model sees more context and can give a
   richer answer, but risks including irrelevant passages that dilute focus
   or confuse the model. The sweet spot depends on the knowledge base
   density and passage quality.
""")

    # --- Save results to file ---
    output_path = "results.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# RAG Pipeline Results\n\n")
        f.write("**Model:** Ollama gemma4:e4b (local)\n")
        f.write("**Embeddings:** Ollama nomic-embed-text (local)\n")
        f.write("**Vector Store:** ChromaDB (in-memory)\n\n")

        for r in results:
            f.write(f"## Question: {r['question']}\n\n")
            f.write("### Retrieved Sources (top 3)\n\n")
            for i, src in enumerate(r["retrieved_sources"], 1):
                f.write(f"{i}. **[{src['source']}]** {src['text']}\n")
            f.write(f"\n### Answer\n\n{r['answer']}\n\n---\n\n")

        # Stretch
        f.write("## Stretch: Top-1 vs Top-3 Comparison\n\n")
        f.write(f"### Question: {result_top1['question']} (top-1)\n\n")
        f.write("### Retrieved Sources (top 1)\n\n")
        for i, src in enumerate(result_top1["retrieved_sources"], 1):
            f.write(f"{i}. **[{src['source']}]** {src['text']}\n")
        f.write(f"\n### Answer (top-1)\n\n{result_top1['answer']}\n\n")
        f.write(
            "### Trade-off Analysis\n\n"
            "With top-1, the model gets only the single most relevant passage, which "
            "may miss complementary information (e.g., details about store credit "
            "after 30 days). With top-3, the model sees more context and can give a "
            "richer answer, but risks including irrelevant passages that dilute focus "
            "or confuse the model. The sweet spot depends on the knowledge base "
            "density and passage quality.\n"
        )

    print(f"📄 Results saved to {output_path}")
