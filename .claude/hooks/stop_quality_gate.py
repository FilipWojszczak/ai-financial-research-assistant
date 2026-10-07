#!/usr/bin/env python3
"""Stop hook: quality gate on the Python files Claude edited in this turn.

Runs `ruff check` and `pyright` only on those files (fast, no network, no DB).
If something fails, it blocks the end of the turn once and sends the errors to Claude
to fix. If it still fails on the retry, it lets Claude stop and shows you a warning,
so it never loops.

The files come from a per-session marker file that format_python.py appends to after
each Edit/Write; with no marker, the gate is skipped. Not checked here: files changed
only via Bash or by you, gitignored files, and errors that an edit causes in other,
unedited files (pyright reports only on the files it is given). /ready and CI cover
those.

Controlled by the CLAUDE_STOP_GATE env var (set in .claude/settings.json -> "env"):
  "ruff,pyright" (default) | "ruff" | "off"
"""

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
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


def stop_gate_marker(session_id: str) -> str:
    # Keep in sync with format_python.py.
    safe_id = re.sub(r"[^\w-]", "", session_id) or "unknown"
    return os.path.join(tempfile.gettempdir(), f"claude-stop-gate-{safe_id}")


def sh(args: list[str], cwd: str, timeout: float = 150) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )


def edited_python_files(marker: str, cwd: str) -> list[str]:
    """Paths recorded by format_python.py that still exist and are not gitignored."""
    try:
        with open(marker) as f:
            paths = {line.strip() for line in f if line.strip()}
    except OSError:
        return []
    files = sorted(p for p in paths if os.path.isfile(os.path.join(cwd, p)))
    if not files:
        return []
    # Gitignored files (e.g. data/ scratch scripts) are out of scope, as in pyright's
    # exclude. check-ignore prints the ignored paths and exits 1 if there are none.
    try:
        out = sh(["git", "check-ignore", "--", *files], cwd, timeout=10)
        ignored = set(out.stdout.splitlines())
    except Exception:
        ignored = set()
    return [f for f in files if f not in ignored]


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    marker = stop_gate_marker(data.get("session_id", ""))
    if not os.path.exists(marker):
        notify("Stop gate: skipped (no Python edits by Claude this turn).")
    cwd = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or "."
    edited = edited_python_files(marker, cwd)
    # Consume the marker now; it is re-written only when this turn gets blocked, so
    # the retry checks the same files again even if Claude makes no further edits.
    with contextlib.suppress(FileNotFoundError):
        os.remove(marker)

    tools = {
        t.strip() for t in os.environ.get("CLAUDE_STOP_GATE", "ruff,pyright").split(",")
    }
    if "off" in tools or not shutil.which("uv"):
        notify("Stop gate: skipped (off or uv not found).")

    if not edited:
        notify("Stop gate: skipped (edited Python files were deleted or are ignored).")
    skipped = len(edited) - MAX_FILES
    files = edited[:MAX_FILES]
    skipped_note = (
        f"Stop gate: only the first {MAX_FILES} edited Python files were checked "
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
                    # pyproject.toml sets fix = true; report issues to Claude
                    # instead of silently rewriting files at the end of the turn.
                    "--no-fix",
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
                    "edited files. "
                    "Ask Claude to fix them or run /ready."
                }
            )
        )
        sys.exit(0)

    with contextlib.suppress(OSError), open(marker, "a") as f:
        f.writelines(path + "\n" for path in edited)
    print(
        json.dumps(
            {
                "decision": "block",
                "reason": "Quality gate failed on Python files you edited. "
                "Fix the root causes "
                "(no blanket `# type: ignore` / `noqa`), then finish.\n\n" + report,
                "systemMessage": "Stop gate: failed, errors sent to Claude.",
            }
        )
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
