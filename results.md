# RAG Pipeline Results

**Model:** Ollama gemma4:e4b (local)
**Embeddings:** Ollama nomic-embed-text (local)
**Vector Store:** ChromaDB (in-memory)

## Question: How long do I have to get a full refund?

### Retrieved Sources (top 3)

1. **[policy.md]** Our refund policy allows a full refund within 30 days of purchase, provided the ...
2. **[policy.md]** To cancel your subscription, open Account Settings and choose End Plan. Cancella...
3. **[policy.md]** Premium plan members get priority support, with a guaranteed first response with...

### Answer

You have 30 days from the date of purchase to receive a full refund (source: policy.md). This requires that the item be unused and in its original packaging (source: policy.md).

---

## Question: How do I reset my password?

### Retrieved Sources (top 3)

1. **[it.md]** Reset your password from the login screen by clicking 'Forgot password'. A reset...
2. **[policy.md]** To cancel your subscription, open Account Settings and choose End Plan. Cancella...
3. **[it.md]** The error code 0x80070005 means 'access denied'. Run the application as administ...

### Answer

You can reset your password from the login screen by clicking 'Forgot password' (source: it.md). A reset link will be emailed to your registered address and expires after one hour for security (source: it.md).

---

## Question: What is the company's stock price today?

### Retrieved Sources (top 3)

1. **[it.md]** Company laptops back up automatically to the cloud every night at 2am while conn...
2. **[handbook.md]** Employees may park in lot B after 6pm on weekdays. Lot A is reserved for visitor...
3. **[facilities.md]** The office kitchen is restocked every Monday and Thursday. Please label personal...

### Answer

I don't know — the knowledge base does not contain this information.

---

## Stretch: Top-1 vs Top-3 Comparison

### Question: How long do I have to get a full refund? (top-1)

### Retrieved Sources (top 1)

1. **[policy.md]** Our refund policy allows a full refund within 30 days of purchase, provided the ...

### Answer (top-1)

You have 30 days from the date of purchase to receive a full refund, provided that the item is unused and in its original packaging (source: policy.md).

### Trade-off Analysis

With top-1, the model gets only the single most relevant passage, which may miss complementary information (e.g., details about store credit after 30 days). With top-3, the model sees more context and can give a richer answer, but risks including irrelevant passages that dilute focus or confuse the model. The sweet spot depends on the knowledge base density and passage quality.
