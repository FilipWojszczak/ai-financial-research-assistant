# Orphaned documents on account deletion

Branch: `fix/orphaned-documents-on-account-deletion`. This must land **before** the
[hybrid RAG chat](hybrid-rag-chat.md), which would make the bug below searchable by
every user.

## Goal

`DELETE /auth/me` must never turn a user's private documents into public ones.

Today `User.documents` is a plain relationship with no delete cascade, so
`session.delete(user)` makes SQLAlchemy run `UPDATE document SET owner_id = NULL`
for the user's documents. `owner_id IS NULL` means "public" (see the comment in
`models/document.py` and the `or_(owner_id IS NULL, owner_id = me)` filters in
`api/routers/documents.py`), so every private document of a deleted account becomes
readable by everyone.

New behaviour:

- `DELETE /auth/me` takes a new optional flag `delete_documents` (default `false`). With
  `true`, the user's documents (and their PDFs) are deleted together with the account, and
  the user row is removed completely.
- Otherwise (the default), a user who owns documents is **tombstoned**, not deleted. Their `user` row stays
  (anonymized, inactive, with a `deleted_at` timestamp). Their documents keep `owner_id`,
  so they match nobody's `owner_id = me` filter and are not `NULL`/public: every existing
  and future query already excludes them, with no filter changes.
- A user who owns no documents is hard-deleted exactly as today.
- An administrator can list these orphaned documents and make **chosen** ones public
  with a CLI script. Nothing is ever published automatically.
- A second CLI script removes tombstoned users completely once they no longer own any
  document (for example after the admin published all of them).

Decisions taken during the interview:

| Topic | Decision |
| --- | --- |
| Where the fix lands | Separate `fix/` branch, before the chat feature |
| How documents keep their owner | Tombstone user row (keeps the `owner_id` foreign key valid) |
| What marks a document as orphaned | Its owner has `deleted_at IS NOT NULL`; no new column on `document` |
| Admin interface | CLI service module, not an API endpoint (no admin role exists) |
| Publishing | Only explicitly named document ids; dry run unless `--yes`; all-or-nothing |
| Retention of orphaned documents | Indefinite, accepted; no deletion tool or retention period |
| User choice on deletion | `delete_documents: bool = false` in the `DELETE /auth/me` body; default keeps the safe behaviour (retain, inaccessible) |
| Purging tombstones | Second CLI script; removes tombstoned users that own no documents; dry run unless `--yes` |

## Non-goals

- No admin role, no admin API endpoint.
- No automatic or bulk "publish all" option, and no automatic expiry.
- No tool to delete orphaned documents or their PDFs, and no retention period: orphaned
  documents are kept until an admin publishes them (accepted).
- No per-document choice at account deletion: it is all of the user's documents or none.
- No `DELETE /documents/{id}` changes; the bulk delete reuses its approach (rows first,
  then files).
- No change to `DELETE /documents/{id}` or to how documents are uploaded.
- No backfill: documents orphaned by earlier deletions are already `owner_id IS NULL`
  and cannot be told apart from intentionally public ones. At spec time the dev
  database has no such documents (all 10 are owned).
- No new `DocumentStatus` value: `status` is the ingestion lifecycle.

## Design

### Model and migration

`models/user.py`: add

```python
deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
```

and change the relationship to
`documents: Mapped[list[Document]] = relationship(back_populates="owner",
passive_deletes="all")`.

*Why `passive_deletes="all"`:* it stops SQLAlchemy from touching child rows when a user
is deleted, so `session.delete(user)` on a user who still owns documents fails with an
`IntegrityError` from the foreign key instead of silently nulling `owner_id`. This is
the safety net against re-introducing the bug from any other code path.

Migration `add_user_deleted_at` (autogenerate, then review): one nullable column, no
default, no backfill. Downgrade drops the column. Per the database rules the column is
declared on the model, so autogenerate sees it.

### `core/accounts.py` (new)

```python
TOMBSTONE_PASSWORD_HASH = "!"   # not a valid hash: verify_password() must return False
DOCUMENT_LOCK_TIMEOUT = "5s"

class DocumentsBusyError(Exception): ...   # a document is locked by a running ingestion

def tombstone_email(user_id: int) -> str:
    return f"deleted-{user_id}@deleted.invalid"

async def delete_account(
    session: AsyncSession, user: User, *, delete_documents: bool = False
) -> list[int]:
    """Delete or tombstone the account; return ids of deleted documents.

    The caller commits, then removes the returned documents' PDFs.
    """
```

`delete_account` (no commit inside; the router commits, as today):

1. `delete_documents=True`:
   - `SET LOCAL lock_timeout = '5s'`;
   - `DELETE FROM document WHERE owner_id = :id RETURNING id` as a Core statement. The
     database cascades remove chunks, graph rows, ingestion checkpoints and outbox rows
     (`ON DELETE CASCADE`, already in place);
   - a lock-timeout error (SQLSTATE `55P03`) is turned into `DocumentsBusyError`.
     *Why:* the ingestion worker holds `FOR NO KEY UPDATE` on the document row for a whole
     attempt, which can take minutes to hours, and `DELETE` conflicts with it. Without a
     timeout the account-deletion request would hang for that long.
2. `owns_documents = await session.scalar(select(exists().where(Document.owner_id ==
   user.id)))` (always false after step 1).
3. No documents → `await session.delete(user)` (the user row is removed completely).
4. Documents (default path) → tombstone: `user.email = tombstone_email(user.id)`,
   `user.hashed_password = TOMBSTONE_PASSWORD_HASH`, `user.is_active = False`,
   `user.deleted_at = func.now()`.

Anonymizing the email frees the unique email, so the person can register again. The new
account gets a new id and cannot see the old documents. Existing JWTs are rejected by the
existing `is_active` check (403), and login fails (`authenticate_user` already requires
`is_active`).

`schemas/auth.py`: `AccountDelete` gets `delete_documents: bool = False` (still
`extra="forbid"`).

`api/routers/auth.py`: `delete_my_account` calls
`delete_account(session, user, delete_documents=payload.delete_documents)` instead of
`session.delete(user)`.

- The password check happens first and is unchanged.
- `DocumentsBusyError` → rollback and `409` ("A document is still being processed; try
  again later, or delete the account without deleting documents"). Nothing changes: the
  user and every document stay as they were.
- After a successful commit, remove the PDFs of the deleted documents with
  `delete_document_file`, **best effort**: a failure is logged (document id, exception
  type) and does not fail the request, because the account is already gone.
  `services/cleanup_storage` removes any leftover `<id>.pdf` whose row no longer exists
  after its normal minimum age, so a missed file is cleaned up later.
- The endpoint `description` states both behaviours: by default the documents the user
  uploaded are retained and become inaccessible until an administrator publishes them;
  with `delete_documents=true` they are deleted with the account.
- `204` stays the response in all successful cases.

Check `core/security.verify_password` with the tombstone value: if the hashing library
raises on a malformed hash, use an unusable but well-formed hash instead (generated once
at import, never stored elsewhere). A unit test pins this.

### Admin script `services/publish_orphaned_documents.py` (new)

```
python -m financial_assistant.services.publish_orphaned_documents --list
python -m financial_assistant.services.publish_orphaned_documents --document-id 12 --document-id 15
python -m financial_assistant.services.publish_orphaned_documents --document-id 12 --document-id 15 --yes
```

Same style as the other `services/` entrypoints (argparse, `asyncio.run`, the existing
session maker, `logging`). A document is **orphaned** when its owner row has
`deleted_at IS NOT NULL`.

- `--list`: prints id, ticker, type, year, filename, status, former owner id and the
  owner's `deleted_at`, ordered by `deleted_at` then id. Prints "No orphaned documents."
  when there are none. Mutually exclusive with `--document-id`.
- `--document-id` (repeatable, positive ints, at least one unless `--list`): without
  `--yes` it is a **dry run**: prints what would be published and exits 0 without writing.
- With `--yes`, in **one transaction**:
  1. select the requested documents with their owners;
  2. if any id is missing, is already public, is owned by a live user, or is duplicated,
     print every offending id with the reason and exit 1 with nothing changed
     (all-or-nothing);
  3. `UPDATE document SET owner_id = NULL WHERE id = ANY(:ids) AND owner_id IN (SELECT id
     FROM "user" WHERE deleted_at IS NOT NULL)`; if the row count differs from the number
     of ids (a concurrent change), roll back and exit 1;
  4. commit; log one INFO line per document (id, former owner id) as the audit trail, and
     print a summary.
- Documents that are not `completed` can be published too (they are excluded from search
  by status until they complete); the list shows the status so the admin can decide.
- Exit codes: 0 success or dry run, 1 refused/failed, 2 usage error (argparse).
- Add the command to `CLAUDE.md` (Commands).

*Why a CLI and not an endpoint:* the project has no admin role; adding one is a separate
authorization feature. Running the module needs shell access to the deployment, which is
the intended gate.

### Admin script `services/purge_deleted_users.py` (new)

```
python -m financial_assistant.services.purge_deleted_users --list
python -m financial_assistant.services.purge_deleted_users            # dry run, all eligible
python -m financial_assistant.services.purge_deleted_users --user-id 7 --yes
python -m financial_assistant.services.purge_deleted_users --yes       # all eligible
```

A tombstoned user is **eligible** when `deleted_at IS NOT NULL` and no `document` row has
`owner_id = user.id`. That is the state after the admin published (or the owner deleted)
the last document.

- `--list`: prints every tombstoned user as `id`, `deleted_at` and `documents owned`
  (a count), marking the eligible ones (count 0). Mutually exclusive with `--yes`
  and `--user-id`.
- Without `--yes` it is a **dry run**: prints the users that would be removed.
- `--user-id` (repeatable) restricts the run to those users. An id that is unknown, is
  not tombstoned, or still owns documents is reported with its reason and exits 1
  without deleting anything (all-or-nothing, as for publishing). Without `--user-id`,
  every eligible user is removed and ineligible tombstones are silently left alone.
- With `--yes`, one transaction runs
  `DELETE FROM "user" WHERE deleted_at IS NOT NULL AND id = ANY(:ids) AND NOT EXISTS
  (SELECT 1 FROM document WHERE owner_id = "user".id) RETURNING id` and commits. The
  foreign key is the final guard: if a document appeared in the meantime the statement
  fails with an `IntegrityError`, which rolls everything back and exits 1. Live users
  (`deleted_at IS NULL`) can never match.
- Anything that references `user.id` with `ON DELETE CASCADE` (the chat tables, once they
  exist) is removed by the database. Chat checkpoints are not: `delete_account` already
  removed them when the account was tombstoned, so nothing is left for the script to
  clean.
- Logs one INFO line per removed user id and prints a summary. Exit codes as for the
  publish script. Add both scripts to `CLAUDE.md` (Commands).

Typical admin flow: `publish_orphaned_documents --list`, publish the documents worth
keeping, then `purge_deleted_users --yes` to drop the tombstones that no longer own
anything. Documents that are never published keep their tombstone owner forever
(accepted).

## Data & migrations

One migration adding `user.deleted_at` (above). No data changes, no re-embedding, no
changes to documents, chunks or the graph. The PDFs stay in document storage;
`services/cleanup_storage.py` only removes files of documents that no longer exist (check
this when implementing: it must not treat tombstone-owned documents as removable).

## External calls & cost

None. No LLM or embedding calls.

## Edge cases

- **User with documents deletes their account twice:** the second request carries a token
  of an inactive user and gets `403`, as for any inactive user.
- **Deleting while one of their documents is still `processing`:**
  - default path: ingestion continues; the document ends up `completed` or `failed` and
    stays orphaned;
  - `delete_documents=true`: the `DELETE` waits at most 5 s for the row lock, then the
    request fails with `409` and nothing changes. The user retries later or deletes the
    account without deleting documents.
- **`delete_documents=true` for a user without documents:** same as today (row removed);
  nothing to delete.
- **PDF removal fails after commit:** logged; the cleanup service removes the file later.
  The account and documents are already gone, so the request still returns `204`.
- **Ingestion task for a document deleted this way:** the worker already handles a missing
  document (`document is None` → skip), as with `DELETE /documents/{id}`.
- **Purge script and a tombstoned user whose last document was just published:** eligible
  immediately. If a document appears between the check and the delete, the FK makes the
  transaction fail and nothing is removed.
- **Purged user's email:** the anonymized `deleted-<id>@deleted.invalid` disappears with
  the row; ids are never reused (serial), so no old token can ever match a new account.
- **Same email registers again:** allowed (the email was anonymized); new id, no access to
  old documents.
- **Tombstone email through `UserRead`:** `EmailStr` rejects the `.invalid` domain, so a
  tombstoned user must never be serialized through `UserRead`. They cannot reach
  `/auth/me` anyway (403). The admin script prints plain strings.
- **A user with documents who never confirmed the password:** unchanged; the password
  check happens before `delete_account`.
- **Direct `session.delete(user)` for a user with documents anywhere else:** raises
  `IntegrityError` (safety net), instead of publishing the documents.
- **Admin publishes a document of a live user, a public one or an unknown id:** refused,
  nothing changes.
- **Two admins publish at once:** the second sees the row count mismatch and rolls back.
- **Rollback of the migration** while tombstoned users exist: dropping `deleted_at` loses
  the marker, and the documents would then look owned by a regular (inactive) user. They
  stay inaccessible, but the admin script can no longer find them. Document this in the
  migration docstring.

## Test plan

**Unit** (`tests/unit/…`):
- `core/test_accounts.py`: `tombstone_email` format; `verify_password(anything,
  TOMBSTONE_PASSWORD_HASH)` returns `False` and does not raise.
- `services/test_publish_orphaned_documents.py`: argument rules (`--list` excludes ids,
  ids required otherwise, non-positive ids rejected) and the refusal-message builder, with
  no database.
- `services/test_purge_deleted_users.py`: argument rules (`--list` excludes `--yes` and
  `--user-id`, non-positive ids rejected).
- `schemas/test_auth.py` (or the existing schema tests): `AccountDelete` defaults
  `delete_documents` to `False` and still rejects extra fields.

**Integration** (`tests/integration/…`, Postgres; the existing `client`, `user_factory`,
`token_factory`, `document_factory` fixtures):
- `api/routers/test_auth.py` (extend, keeping the existing deletion tests unchanged):
  - user **without** documents → `204`, row gone;
  - user **with** a private document → `204`; the row remains with `is_active=False`,
    anonymized email, `deleted_at` set; the document's `owner_id` is unchanged;
  - another user cannot list or get that document (`404`); a document the user had
    uploaded as public (`owner_id NULL`) is unaffected and stays public;
  - the old token → `403`; login with the old email → `401`;
  - registering the same email again succeeds, with a different id and an empty document
    list;
  - **regression:** deleting a user who owns documents must not leave any of them with
    `owner_id IS NULL`.
  - `delete_documents=true`: the user row, the documents and their chunks are gone, the
    PDFs are removed (`tmp_path` storage, as in the existing document tests), a public
    document of someone else is untouched, and the response is `204`;
  - `delete_documents=true` while a document row is locked `FOR NO KEY UPDATE` by another
    connection (same technique as the existing ingestion lock tests, on committed data):
    `409` after about the lock timeout, with the user and every document unchanged;
  - a failing PDF removal (storage function patched to raise) still returns `204` and
    logs.
- `core/test_accounts.py`: `session.delete(user)` for a user who owns a document raises
  `IntegrityError` (the `passive_deletes="all"` safety net).
- `services/test_purge_deleted_users.py`: only tombstoned users without documents are
  removed; a tombstone that still owns a document and a live user are never touched;
  dry run changes nothing; an ineligible `--user-id` refuses the whole batch; after
  publishing the last document of a tombstone with the other script, the purge removes
  it.
- `services/test_publish_orphaned_documents.py`: `--list` returns only orphaned documents;
  publishing a chosen subset leaves the others orphaned and makes the chosen ones visible
  to another user; dry run changes nothing; any invalid id (unknown, public, live owner,
  duplicate) refuses the whole batch with nothing changed.

## End-to-end verification steps

1. `uv run alembic upgrade head`; `\d "user"` shows `deleted_at`.
2. Start the API. With two throwaway users A and B: A uploads a private PDF, A calls
   `DELETE /auth/me`. Check in `psql` that A's row is anonymized and the document's
   `owner_id` is still A's id; B's `GET /documents` and `GET /documents/{id}` do not show
   it; A's old token gets `403`.
3. `--list` shows the document. `--document-id <id>` (dry run) changes nothing. Re-run
   with `--yes`: B can now list it. A second id that is not orphaned is refused with
   nothing changed.
4. With a third throwaway user C who owns one document: `DELETE /auth/me` with
   `{"password": "...", "delete_documents": true}` returns `204`; C's row is gone, the
   document row and `<DOCUMENT_STORAGE_PATH>/<id>.pdf` are gone.
5. Publish the orphaned document from step 3 (user A's), then
   `purge_deleted_users --list` shows A as eligible; a dry run lists A; `--yes` removes
   A's row. Running `--list` again shows no tombstones left.
6. Quality gates: `uv run ruff check --fix . && uv run ruff format .`, `uv run pyright`,
   `uv run pytest -m "not llm and not db" -x -q`, `uv run pytest -m integration -x -q`.

## Open questions

None open. Decided during the interview: orphaned documents are retained indefinitely
(no deletion tool or retention period), and the user chooses at account deletion whether
their documents are deleted (`delete_documents`, default `false`).
