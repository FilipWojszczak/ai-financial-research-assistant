#!/usr/bin/env python3
"""PreToolUse hook for Bash: project-specific guardrails.

deny -> the command is blocked and Claude gets the reason (so it can do the right thing)
ask  -> you get a confirmation prompt, even in Auto mode

Stdlib only. Input: hook JSON on stdin.
Output: decision JSON on stdout (or nothing = no opinion).
"""

import json
import os
import re
import subprocess
import sys

# `git`, optionally followed by global options (`-C <dir>`, `-c k=v`, `--no-pager`),
# so `git -C . commit` is not missed.
GIT = r"\bgit(?:\s+(?:-[Cc]\s+\S+|-{1,2}[\w-]+(?:=\S+)?))*"

DENY_RULES = [
    (
        re.compile(r"(^|[\s;&|(])(pip3?|python3?\s+-m\s+pip|uv\s+pip)\s+install\b"),
        "Use uv, never pip: `uv add <pkg>` (runtime) or "
        "`uv add --dev <pkg>` (dev tools).",
    ),
    (
        # --force, -f (also combined, e.g. -uf) and a leading + in a refspec (+main).
        re.compile(
            GIT
            + r"\s+push\b.*(\s--force(?!-with-lease)\b|\s-[a-zA-Z]*f[a-zA-Z]*\b|\s\+\S)"
        ),
        "Force push is blocked. Use `git push --force-with-lease` if a rewrite is "
        "really needed (it will ask the user).",
    ),
]

# .env, .env.local, .env.test ... but not templates like .env.example /
# .env.test.example
ENV_FILE = re.compile(r"(?<![\w.-])\.env(?:\.[\w.-]+)?(?![\w.-])")

ASK_RULES = [
    (
        re.compile(
            r"\b(DROP\s+(TABLE|SCHEMA|DATABASE|INDEX|EXTENSION)|TRUNCATE)\b",
            re.IGNORECASE,
        ),
        "Destructive SQL (DROP/TRUNCATE).",
    ),
    (
        re.compile(r"\bDELETE\s+FROM\b(?![^;]*\bWHERE\b)", re.IGNORECASE),
        "DELETE without WHERE.",
    ),
    (
        re.compile(
            r"\bdocker(-compose|\s+compose)?\b.*\b(down\b.*(\s-v\b|--volumes)|volume\s+(rm|prune)|system\s+prune)"
        ),
        "This can delete Docker volumes, including the Postgres/pgvector data.",
    ),
    (
        # -f/--force (bypasses .gitignore), -A/--all, and a bare `.` or `*` (stage every
        # untracked file that is not ignored, whatever its name).
        re.compile(
            GIT + r"\s+add\b.*(\s--force\b|\s--all\b|\s-[a-zA-Z]*[fA][a-zA-Z]*\b"
            r"|\s(\./?|\*)(?=\s|$|[;&|]))"
        ),
        "Broad or forced `git add` can stage secrets. Add explicit paths instead.",
    ),
    (
        re.compile(GIT + r"\s+push\b.*--force-with-lease"),
        "Force push (with lease) rewrites remote history.",
    ),
    (
        # `git stash *` and `git switch *` are allowed in settings.json, but these
        # variants drop uncommitted work. Flags may come after the branch name, so a
        # prefix rule in settings.json cannot catch them.
        re.compile(
            GIT + r"\s+(stash\s+(drop|clear)\b"
            r"|switch\b.*(\s--force\b|\s--discard-changes\b|\s-[a-zA-Z]*f[a-zA-Z]*\b))"
        ),
        "This can discard uncommitted or stashed work.",
    ),
]

PROTECTED_BRANCHES = {"main", "master"}


def decide(decision: str, reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": decision,
                    "permissionDecisionReason": reason,
                },
                "systemMessage": f"bash_guard: {decision}",
            }
        )
    )
    sys.exit(0)


def current_branch(cwd: str) -> str:
    try:
        out = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    command = (data.get("tool_input") or {}).get("command", "")
    if not command:
        sys.exit(0)

    for pattern, reason in DENY_RULES:
        if pattern.search(command):
            decide("deny", reason)

    # Commits directly on main/master are blocked (unless the same command creates a
    # branch first).
    if re.search(GIT + r"\s+commit\b", command) and not re.search(
        GIT + r"\s+(switch\s+-c|checkout\s+-b)\b", command
    ):
        cwd = data.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or "."
        if current_branch(cwd) in PROTECTED_BRANCHES:
            decide(
                "deny",
                "Never commit directly to main. Create a branch first, e.g. "
                "`git switch -c feat/<short-name>`.",
            )

    if any(not m.group(0).endswith(".example") for m in ENV_FILE.finditer(command)):
        decide("ask", "The command touches a .env file with secrets.")

    for pattern, reason in ASK_RULES:
        if pattern.search(command):
            decide("ask", reason)

    print(json.dumps({"systemMessage": "bash_guard: no opinion"}))
    sys.exit(0)


if __name__ == "__main__":
    main()
