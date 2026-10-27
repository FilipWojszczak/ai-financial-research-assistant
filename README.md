# AI Financial Research Assistant

## Admin scripts

Both scripts are a dry run unless `--yes` is passed.

- Orphaned documents: list documents of deleted accounts with
  `uv run python -m financial_assistant.services.publish_orphaned_documents --list`,
  then make chosen ones public with
  `... --document-id <id> [--document-id <id> ...] --yes`.
- Deleted users: list tombstoned users that own no documents with
  `uv run python -m financial_assistant.services.purge_deleted_users --list`,
  then remove them with `... [--user-id <id> ...] --yes`.
