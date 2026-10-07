---
name: spec
description: Interview me about a new feature and write a self-contained spec before any code is written.
disable-model-invocation: true
argument-hint: "<short feature description>"
---

Feature to specify: $ARGUMENTS

Do not write or change any code in this task.

1. Read CLAUDE.md and the code this feature will touch.
2. Interview me with the AskUserQuestion tool, a few questions at a time. Skip the obvious; dig into the hard parts:
   - data model and schema changes (pgvector dimensions, indexes, migrations, re-embedding),
   - LLM calls: which model, prompts, structured output schema, expected token cost per document/query,
   - failure modes: timeouts, rate limits, bad LLM output, partial ingestion,
   - impact on retrieval quality and how we will measure it,
   - test strategy (what is unit-tested with fakes vs. `llm`/`db`-marked tests),
   - what is explicitly out of scope.
3. Keep going until no major open question remains (at most ~4 rounds).
4. Write `docs/specs/<kebab-case-name>.md` with the sections: Goal, Non-goals, Design (modules, files and interfaces to change), Data & migrations, LLM calls & cost, Edge cases, Test plan, End-to-end verification steps, Open questions.
5. Finish by suggesting that I implement it in a new conversation, in plan mode, starting from the spec file.
