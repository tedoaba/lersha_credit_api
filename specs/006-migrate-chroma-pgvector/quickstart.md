# Quickstart: pgvector Migration (006-migrate-chroma-pgvector)

**Branch**: `006-migrate-chroma-pgvector`

This guide gives a developer all the commands needed to set up the environment and validate the pgvector migration locally.

---

## Prerequisites

- Docker Compose with `pgvector/pgvector:pg16` image (replaces `postgres:16`)
- `uv` installed and `.venv` active
- `.env` populated (copy from `.env.example`, ensure `CHROMA_DB_PATH` is removed)

---

## 1. Start infrastructure

```bash
# Note: postgres image must be pgvector/pgvector:pg16 after the docker-compose change
docker compose up -d postgres redis mlflow
```

---

## 2. Apply the pgvector migration

```bash
# Run all three migrations (001, 002, 003) on a fresh DB
uv run alembic upgrade head

# Verify the extension and tables exist
docker compose exec postgres psql -U lersha -d lersha \
  -c "\dx vector" \
  -c "\d rag_documents" \
  -c "\d rag_audit_log"
```

Expected: `vector` extension listed, both tables show correct columns including `embedding vector(384)`.

---

## 3. Populate the document store

```bash
# Run the ingestion script (one-time operational step)
uv run python -m backend.scripts.populate_pgvector

# Expected output:
# [OK] Ingested 42 documents in 3 batch(es). Failures: 0.
```

---

## 4. Run the backend

```bash
make api
# or
uv run uvicorn backend.main:app --reload --port 8000
```

---

## 5. Run integration tests

```bash
# Run only the pgvector integration tests
uv run pytest backend/tests/integration/test_rag_pgvector.py -v

# Run full suite with coverage
uv run pytest backend/tests/ --cov=backend --cov-fail-under=80
```

---

## 6. Validate latency benchmark

```bash
# Quick latency smoke test (requires backend running)
uv run python -c "
import time, json
from backend.chat.rag_engine import retrieve_docs
start = time.perf_counter()
results = retrieve_docs('Model predicted: Eligible')
elapsed_ms = (time.perf_counter() - start) * 1000
print(f'Retrieved {len(results)} docs in {elapsed_ms:.1f}ms')
assert elapsed_ms < 50, f'Latency {elapsed_ms:.1f}ms exceeds 50ms target'
print('PASS')
"
```

---

## 7. Verify audit log population

```bash
docker compose exec postgres psql -U lersha -d lersha \
  -c "SELECT id, query_text, retrieved_ids, latency_ms FROM rag_audit_log ORDER BY id DESC LIMIT 5;"
```

---

## 8. ChromaDB removal (Phase 2 — after validation)

Only run once all integration tests pass and audit log is confirmed working:

```bash
# Remove chromadb from dependencies
uv remove chromadb

# Verify import is gone
uv run python -c "import backend.chat.rag_engine; print('OK')"

# Clean up docker volumes
docker volume rm lersha_credit_api_chroma_data 2>/dev/null || true
```

---

## Rollback

```bash
# Revert migration 003 only
uv run alembic downgrade 002

# Restore chromadb (if already removed)
uv add "chromadb>=0.5.0"
```
