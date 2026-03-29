# Contract: Production Stack External Interfaces

**Feature**: `005-infra-prod-hardening`  
**Date**: 2026-03-29

---

## Overview

This feature does not change any HTTP API routes or Pydantic schemas. The external contract is defined by the **Caddy reverse proxy routing rules** — the public-facing interface that maps incoming HTTPS requests to upstream services.

---

## Routing Contract (Caddyfile)

```
your-domain.com {
    reverse_proxy /v1/* backend:8000
    reverse_proxy /* ui:8501
}
```

| Incoming Path | Routed To | Notes |
|---|---|---|
| `https://your-domain.com/v1/*` | `backend:8000` | All versioned API routes (predict, results, health, docs) |
| `https://your-domain.com/*` | `ui:8501` | Streamlit UI (all non-API paths) |
| `http://your-domain.com/*` | Redirect 301 to HTTPS | Caddy automatic HTTP → HTTPS redirect |

---

## Docker Secrets Contract

The backend and worker services expect the following secrets to be mounted:

| Secret name | Mount path | Config field | Fallback |
|---|---|---|---|
| `api_key` | `/run/secrets/api_key` | `config.api_key` | `$API_KEY` env var |
| `gemini_api_key` | `/run/secrets/gemini_api_key` | `config.gemini_api_key` | `$GEMINI_API_KEY` env var |

---

## MLflow Model Registry Contract

| Attribute | Value |
|---|---|
| Registry URI | `models:/lersha-{model_name}/Production` |
| Supported model names | `xgboost`, `random_forest`, `catboost` |
| Fallback | Local `.pkl` path from config if registry unreachable |
| Client | `mlflow.sklearn.load_model(...)` |

---

## Backup Service Environment Contract

The backup service reads these environment variables:

| Variable | Maps to | Example |
|---|---|---|
| `POSTGRES_HOST` | Database hostname | `postgres` |
| `POSTGRES_DB` | Database name | `lersha` |
| `POSTGRES_USER` | Database user | `lersha` |
| `POSTGRES_PASSWORD` | Database password | (from `.env`) |
| `SCHEDULE` | Cron expression | `@daily` |
| `BACKUP_KEEP_DAYS` | Daily retention count | `30` |
| `BACKUP_KEEP_WEEKS` | Weekly retention count | `4` |
| `BACKUP_KEEP_MONTHS` | Monthly retention count | `6` |

---

## Makefile Interface (new targets)

| Target | Invocation | Description |
|---|---|---|
| `pre-commit` | `make pre-commit` | Run all pre-commit hooks across all files |
| `typecheck` | `make typecheck` | Run mypy static type checker on `backend/` |
| `restore-db` | `make restore-db BACKUP_FILE=<path>` | Restore database from compressed backup |
