---
paths:
  - "financial_assistant/ai/**"
  - "financial_assistant/tasks/**"
  - "financial_assistant/retrieval/**"
  - "financial_assistant/agents/**"
---

Follow the `llm-engineering` skill.

# LLM pipeline rules (ingestion → extraction → graph → retrieval → agent)

- These settings all change retrieval quality. Before changing any of them, state the expected impact and how to verify it (no eval command or dataset exists yet, so say how it would be checked):
  - chunking: parent 2000/200, child 400/50 chars (`RecursiveCharacterTextSplitter` in `ai/document_ingestion.py`);
  - embedding model: `models/gemini-embedding-001` at 768 dims, defined once in `ai/embeddings.py` and shared by chunk, community-summary and query embedding;
  - entity resolution/dedup: entities and relationship endpoints are keyed by `name.lower().strip()` (`ai/graph_extraction.py`);
  - community detection: networkx Louvain with `seed=42` (`ai/community_detection.py`);
  - hybrid retrieval (`retrieval/`, settings in `core/config.py`): BM25 and dense top `retrieval_bm25_k`/`retrieval_dense_k` (20/20) child chunks fused with RRF (`retrieval_rrf_k` 60, equal weights) → top `retrieval_fused_children` (8) → at most `retrieval_max_parents` (5) parents per search, `retrieval_max_context_sources` (10) chunk sources per turn; community summaries: dense top `retrieval_community_k` (3) above `retrieval_community_min_similarity` (0.5); `retrieval_hnsw_ef_search` (100).
- Every retriever filters documents through `accessible_documents` (`retrieval/scope.py`), the only place that encodes access control. Never take the user or thread filters from LLM tool arguments.
- Changing the chunking or the embedding model also requires re-embedding all documents (and, for the model, a migration): flag it and ask, never do it silently.
- Model calls go through `ai_request`, and embeddings through `embed_texts` / `embed_query` (all in `ai/requests.py`): they apply the per-call deadline, stall retries, embedding batching and tracing spans. Don't call `ainvoke`/`aembed_documents`/`aembed_query` directly.
- AI results are cached per document in `AttemptCheckpoints` (`ai/checkpoints.py`), keyed by `sha256(model + prompt)`, so a retried ingestion reuses them. New per-document model calls should use it, too. A new `CheckpointKind` needs a migration (it is the Postgres enum `checkpoint_kind_enum`). Changing a prompt or model invalidates the existing checkpoints.
