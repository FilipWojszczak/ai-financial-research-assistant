---
paths:
  - "tests/**"
  - "**/test_*.py"
  - "**/*_test.py"
  - "**/conftest.py"
---

# Testing rules

- pytest only. Mark tests that call a real LLM/embedding API with `@pytest.mark.llm` (skipped unless `RUN_LLM_TESTS=1`). Tests get the `db` marker automatically when they use a fixture listed in `_DB_FIXTURES` in `tests/conftest.py` (`session`, `private_database`, `outbox_db`); a new fixture that connects to Postgres must be added to that set. Unmarked tests must run offline.
- Unit tests never call real LLMs: use fakes from `langchain_core` (`GenericFakeChatModel`, `FakeListChatModel`, `DeterministicFakeEmbedding`) or injected stubs.
- Test LangGraph nodes as plain functions (state in → partial state update out) before testing the compiled graph.
- Shared setup goes into fixtures in `conftest.py`, not copy-pasted between tests.
- When fixing a bug, first write a test that fails because of it, then fix the code.
- Never weaken or delete an existing assertion just to make a test pass; if the expected behavior changed, say so explicitly.
