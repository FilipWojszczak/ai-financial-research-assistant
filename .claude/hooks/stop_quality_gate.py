#!/usr/bin/env python3
"""Stop hook: quality gate on the Python files changed in the working tree.

Runs `ruff check` and `pyright` only on changed .py files (fast, no network, no DB).
If something fails, it blocks the end of the turn once and sends the errors to Claude
to fix. If it still fails on the retry, it lets Claude stop and shows you a warning,
so it never loops.

Controlled by the CLAUDE_STOP_GATE env var (set in .claude/settings.json -> "env"):
  "ruff,pyright" (default) | "ruff" | "off"
"""

import json
import os
import shutil
import subprocess
import sys
import time
from typing import NoReturn

MAX_OUTPUT_CHARS = 4000
MAX_FILES = 100
# ruff + pyright share this budget. It must stay below the hook timeout in
# .claude/settings.json (180 s, minus the git calls), or the hook is killed and the
# gate is skipped silently instead of warning.
GATE_BUDGET_SECONDS = 150


def notify(message: str) -> NoReturn:
    print(json.dumps({"systemMessage": message}))
    sys.exit(0)


def sh(args: list[str], cwd: str, timeout: float = 150) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )


def changed_python_files(cwd: str) -> list[str]:
    files: set[str] = set()
    for args in (
        ["git", "diff", "--name-only", "--diff-filter=ACMR", "HEAD"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    ):
        try:
            out = sh(args, cwd, timeout=10)
        except Exception:
            return []
        if out.returncode != 0:
            return []
        files.update(line.strip() for line in out.stdout.splitlines() if line.strip())
    return sorted(
        f
        for f in files
        if f.endswith((".py", ".pyi")) and os.path.isfile(os.path.join(cwd, f))
    )


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    tools = {
        t.strip() for t in os.environ.get("CLAUDE_STOP_GATE", "ruff,pyright").split(",")
    }
    if "off" in tools or not shutil.which("uv"):
        notify("Stop gate: skipped (off or uv not found).")

    cwd = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or "."
    files = changed_python_files(cwd)
    if not files:
        notify("Stop gate: no changed Python files.")
    skipped = len(files) - MAX_FILES
    files = files[:MAX_FILES]
    skipped_note = (
        f"Stop gate: only the first {MAX_FILES} changed Python files were checked "
        f"({skipped} skipped). Run /ready for a full check."
        if skipped > 0
        else ""
    )

    deadline = time.monotonic() + GATE_BUDGET_SECONDS

    def remaining() -> float:
        # A non-positive value makes subprocess.run raise TimeoutExpired at once.
        return max(deadline - time.monotonic(), 0.1)

    failures: list[str] = []
    try:
        if "ruff" in tools:
            r = sh(
                [
                    "uv",
                    "run",
                    "--quiet",
                    "ruff",
                    "check",
                    "--output-format",
                    "concise",
                    *files,
                ],
                cwd,
                timeout=remaining(),
            )
            if r.returncode != 0 and "error: Failed to spawn" not in r.stderr:
                failures.append("## ruff check\n" + (r.stdout or r.stderr).strip())
        if "pyright" in tools:
            r = sh(
                ["uv", "run", "--quiet", "pyright", *files],
                cwd,
                timeout=remaining(),
            )
            if r.returncode != 0 and "error: Failed to spawn" not in r.stderr:
                failures.append("## pyright\n" + (r.stdout or r.stderr).strip())
    except subprocess.TimeoutExpired:
        print(
            json.dumps(
                {
                    "systemMessage": f"Stop gate: ruff/pyright did not finish within "
                    f"{GATE_BUDGET_SECONDS} s, so the check was skipped. Run /ready."
                }
            )
        )
        sys.exit(0)

    if not failures:
        notify(f"Stop gate: passed ({len(files)} files). {skipped_note}".strip())

    report = "\n\n".join(failures)
    if len(report) > MAX_OUTPUT_CHARS:
        report = report[:MAX_OUTPUT_CHARS] + "\n... (truncated)"
    if skipped_note:
        report += "\n\n" + skipped_note

    if data.get("stop_hook_active"):
        # Already retried once in this turn: don't loop, just warn the user.
        print(
            json.dumps(
                {
                    "systemMessage": "Stop gate: ruff/pyright still report problems in "
                    "changed files. "
                    "Ask Claude to fix them or run /ready."
                }
            )
        )
        sys.exit(0)

    print(
        json.dumps(
            {
                "decision": "block",
                "reason": "Quality gate failed on changed Python files. "
                "Fix the root causes "
                "(no blanket `# type: ignore` / `noqa`), then finish.\n\n" + report,
                "systemMessage": "Stop gate: failed, errors sent to Claude.",
            }
        )
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
