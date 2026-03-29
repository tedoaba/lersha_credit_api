# Quickstart: Infrastructure Production Hardening

**Feature**: `005-infra-prod-hardening`  
**Date**: 2026-03-29

---

## Prerequisites

- Docker Desktop (or Docker Engine ≥ 24) installed and running
- `uv` installed (`pip install uv` or `curl -Ls https://astral.sh/uv/install.sh | sh`)
- A domain name with DNS A-record pointed at the production host (for HTTPS; skip for local testing)
- Existing `.env` file at project root (copy from `.env.example`)

---

## Development Workflow (after this feature)

```bash
# 1. Clone and install
git clone <repo>
cd lersha_credit_api
uv sync --extra dev
uv run pre-commit install

# 2. Start full dev stack (auto-loads docker-compose.override.yml)
docker compose up

# Edit backend/ or ui/ files → hot reload is active
# Backend: http://localhost:8000
# UI: http://localhost:8501
# MLflow: http://localhost:5000
```

---

## Production Deployment Workflow

```bash
# 1. Copy and fill environment file
cp .env.example .env
# Edit .env: set real DB creds, API keys, MLFLOW_S3_BUCKET, etc.

# 2. Create secrets files (NEVER commit these)
mkdir -p secrets
echo -n "your-real-api-key" > secrets/api_key
echo -n "your-real-gemini-key" > secrets/gemini_api_key

# 3. Configure your domain in Caddyfile
# Edit Caddyfile: replace "your-domain.com" with your real domain

# 4. Create the MLflow tracking database (ONE-TIME, before first startup)
docker compose -f docker-compose.yml -f docker-compose.prod.yml run --rm postgres \
  psql -h postgres -U "${POSTGRES_USER:-lersha}" -c "CREATE DATABASE mlflow;"
# Or if postgres is already running:
# docker compose exec postgres psql -U "${POSTGRES_USER:-lersha}" -c "CREATE DATABASE mlflow;"

# 5. Start production stack
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# 6. Verify health
curl https://your-domain.com/health
# Expected: {"db":"ok","chroma":"ok"}

# 7. Verify worker
uv run celery -A backend.worker inspect active

# 8. Check migrations
uv run alembic current    # → should show head revision
```

---

## Quality Targets (make targets)

```bash
make test           # pytest + 80% coverage gate
make pre-commit     # all pre-commit hooks on every file
make typecheck      # mypy on backend/
make lint           # ruff check
make format         # ruff format

# Docker validation
docker compose -f docker-compose.yml -f docker-compose.prod.yml config
```

---

## Database Restore

```bash
# Restore from a specific backup file
make restore-db BACKUP_FILE=./backups/2026-03-29_daily.sql.gz
```

---

## MLflow Model Registration (training time)

When training a new model, register it to the MLflow Model Registry:

```python
import mlflow.sklearn

# At the end of your training script:
mlflow.sklearn.log_model(
    pipeline,
    artifact_path="model",
    registered_model_name="lersha-xgboost"  # or lersha-random_forest, lersha-catboost
)
```

Then promote `Staging → Production` in the MLflow UI at `http://mlflow:5000`.

Once promoted, the prediction service will automatically use the registry version on next startup.
