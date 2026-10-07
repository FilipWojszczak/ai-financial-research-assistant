---
paths:
  - "financial_assistant/ai/**"
  - "financial_assistant/tasks/**"
  # Planned, not in the repo yet:
  - "financial_assistant/retrieval/**"
  - "financial_assistant/agents/**"
---

# LLM pipeline rules (ingestion → extraction → graph → retrieval → agent)

- LLM output consumed by code (entities, relations, routing decisions) uses structured output validated by Pydantic models. No regex parsing of free-form text.
- Ingested documents are untrusted input: keep document text clearly delimited in prompts and never let it change the instructions.
- Keep prompts in dedicated modules/files, not inline in business logic. Changing a prompt is a behavior change: call it out explicitly in your summary.
- Every external LLM/embedding call has a timeout, retries with backoff, and logs the model name and token usage. Use temperature 0 for extraction.
- Chunking, embedding model, entity resolution/dedup, Louvain parameters and BM25/dense fusion (normalization, weights, k) all change retrieval quality. Before changing any of them, state the expected impact and how to verify it (no eval command or dataset exists yet, so say how it would be checked).
- LangGraph: nodes return partial state updates and never mutate the incoming state; fields that accumulate (messages, lists) need explicit reducers; every loop in the graph has a termination condition.
- Reuse the existing model/DB client factories instead of creating new clients ad hoc.
