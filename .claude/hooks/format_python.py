#!/usr/bin/env python3
"""PostToolUse hook for Edit|Write: format the edited Python file with ruff.

Only formatting + import sorting here. Full lint (e.g. unused imports) runs at the
end of the turn in stop_quality_gate.py, so ruff does not delete an import Claude
added a moment ago and is about to use in the next edit.

It also appends the file's path to a per-session marker file, so stop_quality_gate.py
checks only the Python files Claude edited in the project this turn.

Never blocks. Every run reports what it did as a systemMessage shown in the chat.
"""

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import NoReturn


def notify(message: str) -> NoReturn:
    print(json.dumps({"systemMessage": f"format_python: {message}"}))
    sys.exit(0)


def stop_gate_marker(session_id: str) -> str:
    # Keep in sync with stop_quality_gate.py.
    safe_id = re.sub(r"[^\w-]", "", session_id) or "unknown"
    return os.path.join(tempfile.gettempdir(), f"claude-stop-gate-{safe_id}")


def run(args: list[str], cwd: str) -> None:
    subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=45)


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    file_path = (data.get("tool_input") or {}).get("file_path", "")
    if not file_path.endswith((".py", ".pyi")) or not os.path.isfile(file_path):
        notify("skipped (not a Python file)")

    project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or "."
    # Files outside the project (e.g. scratch scripts) are not ours to reformat.
    real_project = os.path.realpath(project_dir)
    real_file = os.path.realpath(file_path)
    if os.path.commonpath([real_project, real_file]) != real_project:
        notify("skipped (outside the project)")

    name = os.path.relpath(real_file, real_project)
    # Record the file for the Stop gate, which checks only the files Claude edited this
    # turn (recorded even if formatting fails).
    with (
        contextlib.suppress(OSError),
        open(stop_gate_marker(data.get("session_id", "")), "a") as marker,
    ):
        marker.write(name + "\n")

    if not shutil.which("uv"):
        notify("skipped (uv not found)")

    try:
        run(
            [
                "uv",
                "run",
                "--quiet",
                "ruff",
                "check",
                "--select",
                "I",
                "--fix",
                file_path,
            ],
            project_dir,
        )
        run(["uv", "run", "--quiet", "ruff", "format", file_path], project_dir)
    except Exception:
        # formatting is best-effort; never break the edit flow
        notify(f"formatting failed for {name}")

    notify(f"formatted {name}")


if __name__ == "__main__":
    main()
