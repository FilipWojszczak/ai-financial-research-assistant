---
name: ready
description: Final quality gate before a commit - format, lint, type-check, fast tests, independent review and a commit message proposal.
disable-model-invocation: true
argument-hint: "[commit]"
allowed-tools:
  - Bash(uv run ruff *)
  - Bash(uv run pyright *)
  - Bash(uv run pytest *)
  - Bash(git status *)
  - Bash(git diff *)
  - Bash(git log *)
---

Current branch:
!`git branch --show-current`

Working tree:
!`git status --short`

Prepare the current changes for a commit:

1. Run `uv run ruff format .` and `uv run ruff check --fix .`.
2. Run `uv run pyright` and `uv run pytest -m "not llm and not db" -x -q`.
3. If anything fails, fix the root cause (no blanket `# type: ignore`, `noqa` or skipped tests) and re-run. After two unsuccessful attempts, stop and report what is still failing.
4. Use the `code-reviewer` subagent on the diff. Fix only findings that affect correctness, security or cost; list the rest without changing code.
5. Summarize: what changed, every command you ran with its result, and anything I should verify manually.
6. Propose a Conventional Commit message in English. If the arguments below contain `commit` and the branch is not `main`, create the commit; otherwise wait for my approval.

Arguments: $ARGUMENTS
