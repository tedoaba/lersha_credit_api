# Quickstart: Application-Level Security & Reliability Hardening

**Feature Branch**: `004-harden-app-security`
**Updated**: 2026-03-29

---

## Prerequisites

- Python 3.12
- `uv` installed (`pip install uv` or official installer)
- Docker + Docker Compose
- PostgreSQL running (or use `make docker-up`)
- Redis running (added to docker-compose in this feature)

---

## 1. Install Dependencies

```bash
# Install all runtime + dev dependencies (including new: alembic, celery, redis, etc.)
make install
# or: uv sync --extra dev
```

Verify new packages are available:
```bash
uv run python -c "import alembic, celery, redis, pythonjsonlogger, gunicorn; print('OK')"
```

---

## 2. Apply Database Migrations

> ⚠️ **Run this against a real PostgreSQL instance.**

```bash
# From repo root (not backend/)
uv run alembic upgrade head
```

Expected output:
```
INFO  [alembic.runtime.migration] Running upgrade  -> xxxx, initial_schema
INFO  [alembic.runtime.migration] Running upgrade xxxx -> yyyy, add_inference_jobs
```

---

## 3. Start the Full Stack (Development)

```bash
# Start PostgreSQL + Redis + MLflow
make docker-up

# Start API (single worker, hot reload)
make api

# Start Celery worker (separate terminal)
uv run celery -A backend.worker worker --loglevel=info
```

Or start everything via Docker:
```bash
docker compose up -d
```

---

## 4. Developer Setup (One-Time)

```bash
# Install pre-commit hooks into .git/hooks
uv run pre-commit install
```

After this, every `git commit` automatically runs ruff lint, ruff format, and mypy.

---

## 5. Quality Checks

```bash
# Lint
make lint

# Type check
make typecheck

# Pre-commit (all files)
make pre-commit

# All CI quality gates
make ci-quality
```

---

## 6. Verify Rate Limiting

```bash
# Run 11 POST requests — the 11th should return 429
for i in $(seq 1 11); do
  curl -s -o /dev/null -w "%{http_code}\n" \
    -X POST http://localhost:8000/v1/predict/ \
    -H "X-API-Key: <your-api-key>" \
    -H "Content-Type: application/json" \
    -d '{"source":"Batch Prediction","number_of_rows":1}'
done
```

Expected: 10 × `202`, 1 × `429`

---

## 7. Verify Health Check

```bash
curl http://localhost:8000/health
# → {"db":"ok","chroma":"ok"} with HTTP 200

# With Request ID header
curl -H "X-Request-ID: my-trace-id" http://localhost:8000/health
# → Response headers include X-Request-ID: my-trace-id
```

---

## 8. Verify JSON Logs

```bash
# Start the stack and check logs
docker compose logs backend | python -m json.tool | head -20

# Or locally:
make api 2>&1 | python -m json.tool | head -5
```

Every line must be valid JSON with `asctime`, `levelname`, `name`, `message` fields.

---

## 9. Running Tests

```bash
make test
# or:
uv run pytest backend/tests/ -v
```

---

## Production Deployment Notes

The `backend/Dockerfile` CMD runs **Alembic migrations then Gunicorn**:

```dockerfile
CMD ["sh", "-c", "uv run alembic upgrade head && uv run gunicorn backend.main:app --worker-class uvicorn.workers.UvicornWorker --workers 4 --bind 0.0.0.0:8000 --timeout 120"]
```

The Celery worker requires a **separate container** with the same image:
```yaml
# In docker-compose.yml (when adding worker service):
worker:
  build:
    context: .
    dockerfile: backend/Dockerfile
  command: uv run celery -A backend.worker worker --loglevel=info --concurrency=4
  env_file: .env
  depends_on:
    - redis
    - postgres
```

---

## Environment Variables Added in This Feature

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://redis:6379/0` | Celery broker and result backend |

All other variables unchanged. See `.env.example` for the full list.
