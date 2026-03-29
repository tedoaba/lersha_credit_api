# Data Model: Infrastructure Production Hardening

**Feature**: `005-infra-prod-hardening`  
**Date**: 2026-03-29

---

## Overview

This feature is infrastructure-only. It does not introduce new application-level data models or database tables. The entities below are **configuration and deployment artefacts** — they describe the shape of new files/config objects introduced by this feature.

---

## Entity 1 — `Caddyfile` (Routing Configuration)

**What it represents**: The Caddy reverse proxy's virtual host routing configuration. Lives at the project root.

| Field | Value / Type | Notes |
|---|---|---|
| Virtual host | `your-domain.com` | Must be replaced with real domain before production deploy |
| Route `/v1/*` | upstream `backend:8000` | All API traffic |
| Route `/*` | upstream `ui:8501` | All UI traffic (catchall) |
| TLS | Auto (Let's Encrypt) | Enabled via Caddy's automatic HTTPS |

**State transitions**: N/A (static config file)

---

## Entity 2 — `_read_secret(name, env_var)` Helper (Config Layer)

**What it represents**: A function added to `backend/config/config.py` that abstracts the secret-reading strategy. Callers never know whether the value came from a Docker secret file or an environment variable.

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Docker secret name → path `/run/secrets/{name}` |
| `env_var` | `str` | Fallback environment variable name |
| Return value | `str \| None` | Secret value, or `None` if neither source has it |

**Validation rules**:
- Callers that need a non-nullable secret (e.g. `api_key`) MUST raise `ValueError` after calling `_read_secret` if result is `None`.
- The function MUST `strip()` the file value to remove trailing newlines written by `echo` or editors.

---

## Entity 3 — Docker Compose Service Definitions

### Base (`docker-compose.yml`)

| Service | Image | Key mounts | Depends on |
|---|---|---|---|
| `postgres` | `postgres:16` | `postgres_data:/var/lib/postgresql/data` | — |
| `redis` | `redis:7-alpine` | — | — |
| `backend` | `./backend/Dockerfile` | models, chroma_data, output, logs | postgres, redis |
| `worker` | `./backend/Dockerfile` | models, chroma_data, output, logs | postgres, redis |
| `ui` | `./ui/Dockerfile` | — | backend |
| `mlflow` | `ghcr.io/mlflow/mlflow:latest` | `mlruns_data:/mlruns` | — |

### Dev Override (`docker-compose.override.yml`)

Augments the base:

| Service | Override |
|---|---|
| `postgres` | Exposes `5432:5432` |
| `redis` | Exposes `6379:6379` |
| `backend` | Command: `uvicorn --reload`, bind mounts `./backend:/app/backend` + `./ui:/app/ui`, exposes `8000:8000` |
| `ui` | Exposes `8501:8501` |
| `mlflow` | Exposes `5000:5000`, uses SQLite backend |

### Production (`docker-compose.prod.yml`)

Adds to / overrides base:

| Service | Override / Addition |
|---|---|
| `caddy` | `caddy:2-alpine`, ports 80+443, `caddy_data` + `caddy_config` volumes |
| `backend` | Command: `alembic upgrade head && gunicorn ... --workers 4`, secrets mounted |
| `worker` | Secrets mounted |
| `mlflow` | Command uses PostgreSQL backend + S3 artifact root, env: AWS creds |
| `backup` | `prodrigestivill/postgres-backup-local`, volume `./backups:/backups` |

---

## Entity 4 — Secrets Files (`./secrets/`)

**What it represents**: Plaintext files containing secret values, mounted into containers via Docker Compose secrets. Never committed to git.

| Filename | Maps to config field | Docker secret path |
|---|---|---|
| `secrets/api_key` | `config.api_key` | `/run/secrets/api_key` |
| `secrets/gemini_api_key` | `config.gemini_api_key` | `/run/secrets/gemini_api_key` |

**Validation rules**:
- `secrets/` MUST be in `.gitignore`.
- Files MUST contain only the raw secret value (no shell variable syntax, no quotes).
- `_read_secret()` MUST strip whitespace from file contents.

---

## Entity 5 — Named Volumes (Production additions)

| Volume name | Purpose | Used by |
|---|---|---|
| `caddy_data` | Caddy TLS certificate cache | `caddy` service |
| `caddy_config` | Caddy runtime configuration | `caddy` service |

Existing volumes (`postgres_data`, `chroma_data`, `mlruns_data`) are unchanged.

---

## Entity 6 — Backup Artefacts (`./backups/`)

**What it represents**: Directory receiving daily `pg_dump` compressed snapshots from the backup service.

| Attribute | Value |
|---|---|
| File format | `.sql.gz` (gzip-compressed pg_dump) |
| Naming | Managed by `prodrigestivill/postgres-backup-local` (date-stamped) |
| Retention | Daily: 30 days, Weekly: 4 weeks, Monthly: 6 months |
| `.gitignore` | `backups/` entry required |

---

## Entity 7 — MLflow Model Registry Entry

**What it represents**: A versioned model record in the MLflow Model Registry, used to decouple model serving from filesystem paths.

| Attribute | Value |
|---|---|
| Registry name pattern | `lersha-{model_name}` (e.g. `lersha-xgboost`) |
| Stage | `Production` (promoted via MLflow UI from `Staging`) |
| Load URI | `models:/lersha-{model_name}/Production` |
| Fallback | Local `.pkl` path from `config.{model}_model_36` |
| Registration command | `mlflow.sklearn.log_model(pipeline, artifact_path="model", registered_model_name=f"lersha-{model_name}")` |
