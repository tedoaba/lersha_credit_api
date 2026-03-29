# Implementation Plan: Infrastructure Production Hardening

**Branch**: `005-infra-prod-hardening` | **Date**: 2026-03-29 | **Spec**: [spec.md](spec.md)  
**Input**: Feature specification from `/specs/005-infra-prod-hardening/spec.md`

---

## Summary

Harden the Lersha Credit Scoring System at the infrastructure and operations layer. This plan covers Docker Compose restructuring (base / dev override / prod), Caddy HTTPS reverse proxy, Docker secrets management for `api_key` and `gemini_api_key`, MLflow production backend (PostgreSQL + S3/GCS), MLflow Model Registry fallback loading, Celery worker activation in Compose, automated PostgreSQL backups, Makefile additions, CI/CD updates, and `.env.example` documentation.

No application-level business logic is changed. All changes are confined to compose files, the `Caddyfile`, `backend/config/config.py`, `backend/core/infer_utils.py`, `backend/scripts/` (comment only), `Makefile`, `.github/workflows/ci.yml`, `.env.example`, and `.gitignore`.

---

## Technical Context

**Language/Version**: Python 3.12 (backend), YAML (compose/CI), Caddyfile DSL  
**Primary Dependencies**: Docker Compose v2, Caddy 2, Gunicorn, Uvicorn, MLflow, Celery, Redis, postgres-backup-local  
**Storage**: PostgreSQL 16 (app + MLflow tracking), S3/GCS (MLflow artifacts), `./backups/` (pg_dump)  
**Testing**: pytest (existing suite); `docker compose config` for compose validation  
**Target Platform**: Linux Docker host (production), Docker Desktop (development)  
**Project Type**: Web service (FastAPI backend + Streamlit UI) in monorepo  
**Performance Goals**: Backend: 4 Gunicorn workers; Backup: daily pg_dump within window  
**Constraints**: No application logic changes; backward-compatible dev workflow; no breaking changes to existing `.env` variables  
**Scale/Scope**: Single-host Docker deployment; designed to extend to Kubernetes/ECS later

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Gate | Status | Notes |
|---|---|---|---|
| P1-MODULAR | No cross-module imports from new files | ✅ PASS | All changes are to config/compose/ops files |
| P2-PEP | `_read_secret` and `load_prediction_models` changes must be ruff-clean | ✅ PASS | Will enforce ruff + mypy annotations |
| P3-LOG | Fallback path in `load_prediction_models` must emit `WARNING` | ✅ PASS | Planned in implementation |
| P4-EXC | Exception in MLflow registry load must not be silently swallowed | ✅ PASS | Catch → log WARNING → fallback; re-raise ValueError |
| P5-CONFIG | `_read_secret` helper lives in `config.py`; new vars documented in `.env.example` | ✅ PASS | `secrets/` path abstracted in config layer |
| P6-API | No API route changes | ✅ PASS | Caddy proxies existing routes unchanged |
| P7-TEST | No new business logic introduced; existing coverage gate unaffected | ✅ PASS | `_read_secret` and registry fallback are testable units |
| P8-DB | Backup service addresses P8-DB retained-audit-trail requirement | ✅ PASS | Daily pg_dump, 30-day retention |
| P9-SEC | Caddy enforces HTTPS; secrets moved from env vars to Docker secrets | ✅ PASS | Direct access to backend:8000 forbidden by compose net config |
| P10-OBS | MLflow tracking storage migrated from SQLite to PostgreSQL | ✅ PASS | Resolves unsafe concurrent write issue |
| P11-CONT | Three-file compose pattern implemented as mandated | ✅ PASS | Base / override / prod |
| P12-CI | CI updated: mypy step, compose validation, workflow_dispatch | ✅ PASS | All gates maintained |

**No constitution violations. No Complexity Tracking required.**

---

## Project Structure

### Documentation (this feature)

```text
specs/005-infra-prod-hardening/
├── plan.md              ← this file
├── research.md          ← Phase 0: all architectural decisions
├── data-model.md        ← Phase 1: config + deployment entities
├── quickstart.md        ← Phase 1: dev + prod workflow
├── contracts/
│   └── routing-and-secrets.md  ← Phase 1: all external interfaces
└── tasks.md             ← Phase 2 output (speckit.tasks command)
```

### Source Code (files modified or created)

```text
lersha_credit_api/
├── Caddyfile                        ← NEW: Caddy reverse proxy config
├── docker-compose.yml               ← MODIFIED: base, no hardcoded env, no host ports
├── docker-compose.override.yml      ← NEW: dev overrides (hot reload, port bindings)
├── docker-compose.prod.yml          ← NEW: prod overrides (Caddy, Gunicorn, backup, MLflow PG)
├── .env.example                     ← MODIFIED: add Redis, MLflow S3, secrets vars
├── .gitignore                       ← MODIFIED: add secrets/ and backups/
├── Makefile                         ← MODIFIED: add pre-commit, typecheck, restore-db targets
├── .github/
│   └── workflows/
│       └── ci.yml                   ← MODIFIED: workflow_dispatch, mypy step, compose validate
├── secrets/                         ← NEW (gitignored): api_key, gemini_api_key files
└── backend/
    ├── config/
    │   └── config.py                ← MODIFIED: add _read_secret(), apply to api_key + gemini_api_key
    ├── core/
    │   └── infer_utils.py           ← MODIFIED: load_prediction_models() MLflow registry fallback
    └── scripts/
        └── register_model.py        ← NEW: comment-only script documenting registration command
```

**Structure Decision**: This is a pure infrastructure change layered on top of the existing `backend/` + `ui/` monorepo. No new packages or directories are added to the Python source tree beyond a single documentation script.

---

## Implementation Phases

### Phase A — Compose Restructure (base + dev override)
*Prerequisite: None*

**A.1** — Strip `docker-compose.yml` to base-only:
- Remove ALL host port bindings (`ports:` sections) from all services
- Remove the commented-out worker block → replace with active `worker` service definition
- Keep healthchecks on `postgres` and `redis`
- Worker service: image from `backend/Dockerfile`, command `uv run celery -A backend.worker worker --loglevel=info`, `depends_on: [redis, postgres]`, volumes identical to backend (models, chroma_data, output, logs), `env_file: .env`
- Add `environment` block to all services (using `${VAR:-default}` syntax, no hardcoded values)

**A.2** — Create `docker-compose.override.yml` (dev):
- `backend`: command=`uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000`, bind mounts `./backend:/app/backend` + `./ui:/app/ui`, port `8000:8000`
- `postgres`: port `5432:5432`
- `redis`: port `6379:6379`
- `ui`: port `8501:8501`
- `mlflow`: port `5000:5000`, command using SQLite backend (dev-only)

---

### Phase B — Production Compose File
*Prerequisite: Phase A*

**B.1** — Create `docker-compose.prod.yml`:
- **`caddy` service**: image `caddy:2-alpine`, ports `80:80` + `443:443`, volumes `./Caddyfile:/etc/caddy/Caddyfile:ro` + `caddy_data:/data` + `caddy_config:/config`, `depends_on: [backend, ui]`
- **`backend` override**: command=`sh -c "uv run alembic upgrade head && uv run gunicorn backend.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000"`, secrets block mounted (`api_key` + `gemini_api_key`)
- **`worker` override**: secrets block mounted (`api_key` + `gemini_api_key`)
- **`mlflow` override**: command with `--backend-store-uri postgresql://${DB_USER}:${DB_PASSWORD}@postgres:5432/mlflow` and `--default-artifact-root s3://${MLFLOW_S3_BUCKET}/artifacts`, environment: `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`
- **`backup` service**: image `prodrigestivill/postgres-backup-local`, environment `POSTGRES_HOST=postgres`, `POSTGRES_DB=${POSTGRES_DB}`, `POSTGRES_USER=${POSTGRES_USER}`, `POSTGRES_PASSWORD=${POSTGRES_PASSWORD}`, `SCHEDULE=@daily`, `BACKUP_KEEP_DAYS=30`, `BACKUP_KEEP_WEEKS=4`, `BACKUP_KEEP_MONTHS=6`, volume `./backups:/backups`, `depends_on: [postgres]`
- **Top-level `secrets` block**: `api_key: {file: ./secrets/api_key}`, `gemini_api_key: {file: ./secrets/gemini_api_key}`
- **Top-level `volumes` block**: add `caddy_data:` and `caddy_config:`

---

### Phase C — Caddyfile
*Prerequisite: Phase B*

**C.1** — Create `Caddyfile` at project root:
```
your-domain.com {
    reverse_proxy /v1/* backend:8000
    reverse_proxy /* ui:8501
}
```

---

### Phase D — Secrets Management in Config
*Prerequisite: None (independent of compose)*

**D.1** — Add `_read_secret(name: str, env_var: str) -> str | None` to `backend/config/config.py`:
```python
def _read_secret(name: str, env_var: str) -> str | None:
    """Read a secret from Docker secrets file, falling back to environment variable."""
    secret_path = Path(f"/run/secrets/{name}")
    if secret_path.exists():
        return secret_path.read_text(encoding="utf-8").strip()
    return os.getenv(env_var)
```

**D.2** — Apply to `api_key` and `gemini_api_key`:
- `self.api_key = _read_secret("api_key", "API_KEY")`
- `self.gemini_api_key = _read_secret("gemini_api_key", "GEMINI_API_KEY")`
- Keep existing validation (`raise ValueError if None`) immediately after assignment

**D.3** — Type annotation: `_read_secret` must have full `-> str | None` return type annotation to satisfy `[P2-PEP]` and mypy.

---

### Phase E — MLflow Model Registry Fallback
*Prerequisite: None (independent)*

**E.1** — Refactor `load_prediction_models` in `backend/core/infer_utils.py`:

```python
def load_prediction_models(model_name: str):
    """Load a trained prediction model by name.
    
    Tries MLflow Model Registry first (models:/lersha-{model_name}/Production).
    Falls back to local .pkl path if registry is unavailable or model not promoted.
    """
    pkl_path_map = {
        "xgboost": config.xgb_model_36,
        "random_forest": config.rf_model_36,
        "catboost": config.cab_model_36,
    }
    if model_name not in pkl_path_map:
        raise ValueError(f"Unknown model_name '{model_name}'. Expected: xgboost | random_forest | catboost")
    
    registry_uri = f"models:/lersha-{model_name}/Production"
    try:
        model = mlflow.sklearn.load_model(registry_uri)
        mlflow.set_tag("model_source", f"registry:{registry_uri}")
        logger.info("Model '%s' loaded from MLflow registry", model_name)
        return model
    except Exception as e:  # registry unreachable or model not promoted
        logger.warning(
            "MLflow registry unavailable for '%s' (%s); falling back to local pkl.",
            model_name, e,
        )
    
    # Fallback path
    try:
        model = joblib.load(pkl_path_map[model_name])
        mlflow.set_tag("model_source", f"local:{pkl_path_map[model_name]}")
        logger.info("Model '%s' loaded from local pkl", model_name)
        return model
    except Exception as e:
        logger.error("Failed to load model '%s' from local pkl: %s", model_name, e, exc_info=True)
        raise RuntimeError(f"Failed to load model '{model_name}' from both registry and local pkl") from e
```

**E.2** — Add import: `import mlflow.sklearn` (already have `import mlflow`; add `.sklearn` sub-import)

---

### Phase F — Model Registration Script
*Prerequisite: None*

**F.1** — Create `backend/scripts/register_model.py`:
- File contains only a module docstring explaining when and how to run registration
- Documents the `mlflow.sklearn.log_model(...)` call pattern
- Notes: promotion from Staging → Production is done via the MLflow UI
- No executable code (purely documentation for operators)

---

### Phase G — .gitignore and .env.example Updates
*Prerequisite: None*

**G.1** — `.gitignore` additions:
- `secrets/`
- `backups/`

**G.2** — `.env.example` additions (in appropriate sections):
```
# ─── MLflow artifact store (S3) ───────────────────────────────────────────────
MLFLOW_S3_BUCKET=lersha-mlflow-artifacts
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
# GCP alternative: GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# ─── Secrets management ───────────────────────────────────────────────────────
# Production: use Docker secrets (./secrets/api_key, ./secrets/gemini_api_key)
# Cloud alternatives: AWS Secrets Manager, GCP Secret Manager, K8s Secrets
```

---

### Phase H — Makefile Updates
*Prerequisite: None*

**H.1** — `pre-commit` target already exists — verify it is `uv run pre-commit run --all-files` ✅ (confirmed in existing Makefile line 78)

**H.2** — `typecheck` target already exists — verify it is `uv run mypy backend/` ✅ (confirmed in existing Makefile line 75)

**H.3** — Add `restore-db` target:
```makefile
restore-db:
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "Usage: make restore-db BACKUP_FILE=./backups/<filename>.sql.gz"; \
		exit 1; \
	fi
	gunzip -c $(BACKUP_FILE) | uv run python -c \
		"from backend.config.config import config; import subprocess; subprocess.run(['psql', config.db_uri], stdin=open('$(BACKUP_FILE)'))" 
```

> **Simpler alternative**: Use `psql $(DB_URI)` directly with `PGPASSWORD` — avoids Python subprocess complexity. Final implementation should use the simpler form:

```makefile
restore-db:
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "Usage: make restore-db BACKUP_FILE=./backups/<date>.sql.gz"; \
		exit 1; \
	fi
	@echo "Restoring database from $(BACKUP_FILE)..."
	gunzip -c $(BACKUP_FILE) | psql $(DB_URI)
	@echo "Restore complete."
```

**H.4** — Update `docker-build` target:
```makefile
docker-build:
	@echo "NOTE: For production builds, use docker-compose.prod.yml"
	docker build -f backend/Dockerfile -t lersha-backend:latest .
	docker build -f ui/Dockerfile -t lersha-ui:latest .
```

**H.5** — Update `.PHONY` line to include `restore-db`

**H.6** — Update `help` target to include `restore-db` in the Docker section

---

### Phase I — CI/CD Updates
*Prerequisite: None*

**I.1** — Add `workflow_dispatch:` trigger to `.github/workflows/ci.yml`:
```yaml
on:
  push:
    branches: ["**"]
  pull_request:
    branches: ["main"]
  workflow_dispatch:
```

**I.2** — The `lint` job already includes `mypy` step (confirmed: line 34-35 of existing `ci.yml`). **No change needed.**

**I.3** — Add compose validation step to `build` job (after Docker Buildx setup, before image builds):
```yaml
- name: Validate compose files
  run: docker compose -f docker-compose.yml -f docker-compose.prod.yml config > /dev/null
```

> Note: The prod compose validation requires AWS/S3 env vars to be set. Use dummy values in CI:
```yaml
- name: Validate compose files
  env:
    POSTGRES_USER: lersha
    POSTGRES_PASSWORD: lersha
    POSTGRES_DB: lersha
    API_KEY: ci-test-key
    GEMINI_MODEL: gemini-1.5-pro
    GEMINI_API_KEY: placeholder
    MLFLOW_S3_BUCKET: placeholder-bucket
    AWS_ACCESS_KEY_ID: placeholder
    AWS_SECRET_ACCESS_KEY: placeholder
    DB_USER: lersha
    DB_PASSWORD: lersha
  run: docker compose -f docker-compose.yml -f docker-compose.prod.yml config > /dev/null
```

---

## Dependency Order (implementation sequence)

```
Phase D (config _read_secret)    ─── independent, no blockers
Phase E (MLflow fallback)        ─── independent, no blockers
Phase F (register_model.py)      ─── independent, no blockers
Phase G (.gitignore + .env)      ─── independent, no blockers
Phase H (Makefile)               ─── independent, no blockers
Phase I (CI/CD)                  ─── independent, no blockers
Phase A (base compose)           ─── independent, Phase B depends on it
Phase C (Caddyfile)              ─── independent, Phase B depends on it
Phase B (prod compose)           ─── depends on Phase A + C
```

All phases can be implemented in parallel by separate contributors. The only sequential dependency is A → B.

---

## Complexity Tracking

No constitution violations. No complexity tracking entry required.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| MLflow PostgreSQL backend requires `mlflow` DB to exist before server starts | Medium | High | Add `postgres_db_mlflow` init to `db_init.py` or document manual `CREATE DATABASE mlflow` step |
| Caddy ACME challenge fails in firewalled environments | Low | Medium | Document self-signed cert fallback; add `tls internal` option to Caddyfile for testing |
| `secrets/` files include trailing newline causing auth failures | Medium | Medium | `_read_secret()` always `.strip()` the file content |
| `BACKUP_FILE` variable with spaces in `make restore-db` breaks shell | Low | Low | Quote `$(BACKUP_FILE)` in Makefile recipe |
| `docker compose config` in CI fails due to undefined prod-only vars | Medium | Medium | Pass dummy env vars in CI step (see Phase I.3) |
| Worker and backend both load models at startup → doubled memory | Low | Medium | Document: add `BACKEND_SKIP_MODEL_LOAD=1` env var as future optimization if needed |
