# Tasks: Infrastructure Production Hardening

**Input**: Design documents from `/specs/005-infra-prod-hardening/`  
**Prerequisites**: plan.md ✅ | spec.md ✅ | research.md ✅ | data-model.md ✅ | contracts/ ✅ | quickstart.md ✅

**No tests requested** — this is a pure infrastructure/ops feature with no new business logic requiring test suites.  
Validation is done via `docker compose config`, `make typecheck`, `make pre-commit`, and runtime smoke tests per quickstart.md.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no shared dependencies)
- **[Story]**: Which user story this task belongs to (US1–US8)
- All paths are relative to project root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Pre-conditions and gitignore/env hygiene that everything else depends on.

- [x]  Add `secrets/` and `backups/` entries to `.gitignore`
- [x]  [P] Create `secrets/` directory with placeholder `README.md` explaining expected file names (`api_key`, `gemini_api_key`) and that they are gitignored
- [x]  [P] Create `backups/` directory with placeholder `.gitkeep` so the mount point exists before the backup service runs

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core file changes that multiple user-story phases depend on. All can run in parallel with one another.

**⚠️ CRITICAL**: Phases 3–10 cannot begin until this phase is complete.

- [x]  [P] Add `_read_secret(name: str, env_var: str) -> str | None` helper function to `backend/config/config.py` (reads from `/run/secrets/{name}` if the file exists, otherwise falls back to `os.getenv(env_var)`; always `.strip()` the result)
- [x]  [P] Apply `_read_secret` to `api_key` assignment in `backend/config/config.py`: replace `os.getenv("API_KEY")` with `_read_secret("api_key", "API_KEY")`; keep existing `ValueError` guard immediately after
- [x]  [P] Apply `_read_secret` to `gemini_api_key` assignment in `backend/config/config.py`: replace `os.getenv("GEMINI_API_KEY")` with `_read_secret("gemini_api_key", "GEMINI_API_KEY")`; keep existing `ValueError` guard immediately after
- [x]  [P] Update `.env.example` — add the following new sections below the existing `Redis / Celery` section:
  ```
  # ─── MLflow artifact store ───────────────────────────────────────────────────
  MLFLOW_S3_BUCKET=lersha-mlflow-artifacts
  AWS_ACCESS_KEY_ID=
  AWS_SECRET_ACCESS_KEY=
  # GCP alternative: GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

  # ─── Secrets management ───────────────────────────────────────────────────────
  # Production: use Docker secrets (./secrets/api_key, ./secrets/gemini_api_key)
  # Cloud alternatives: AWS Secrets Manager, GCP Secret Manager, K8s Secrets
  
  # ─── Docker Compose DB vars (used by mlflow and backup services) ──────────────
  DB_USER=${POSTGRES_USER}
  DB_PASSWORD=${POSTGRES_PASSWORD}
  ```

**Checkpoint**: Config helper and env documentation complete — all compose and code phases can now proceed in parallel.

---

## Phase 3: User Story 1 — HTTPS / TLS via Caddy (Priority: P1) 🎯

**Goal**: All external traffic is served over HTTPS with auto-provisioned TLS. Port 80 redirects to HTTPS automatically.

**Independent Test**: Start prod stack → `curl -k https://your-domain.com/health` returns `{"db":"ok","chroma":"ok"}`. Caddy logs show certificate issued.

- [x]  [US1] Create `Caddyfile` at project root with the following content:
  ```
  your-domain.com {
      reverse_proxy /v1/* backend:8000
      reverse_proxy /* ui:8501
  }
  ```
  Include a leading comment block explaining: (1) replace `your-domain.com` with the real domain, (2) for local testing without a domain add `tls internal` on a new line inside the block to issue a self-signed cert

---

## Phase 4: User Story 2 — Environment Separation (Priority: P1) 🎯

**Goal**: `docker compose up` (no flags) starts the dev stack with hot reload. The prod stack requires explicit `-f` flags and has no bind mounts.

**Independent Test**: `docker compose up` starts backend with uvicorn `--reload` active; editing a file in `./backend/` triggers a reload in logs. `docker compose -f docker-compose.yml -f docker-compose.prod.yml config > /dev/null` exits 0.

### Phase 4a — Restructure `docker-compose.yml` (base)

- [x]  [US2] Rewrite `docker-compose.yml` as the base-only file:
  - **Remove** all host `ports:` bindings from every service (postgres, redis, backend, ui, mlflow)
  - **Uncomment** the worker service block; remove the comment header and make it a full active service definition with: `build.dockerfile: backend/Dockerfile`, `command: uv run celery -A backend.worker worker --loglevel=info`, `restart: unless-stopped`, `depends_on: {redis: {condition: service_healthy}, postgres: {condition: service_healthy}}`, `env_file: .env`, `environment: {DB_URI: "postgresql://${POSTGRES_USER:-lersha}:${POSTGRES_PASSWORD:-lersha}@postgres:5432/${POSTGRES_DB:-lersha}", REDIS_URL: "redis://redis:6379/0", MLFLOW_TRACKING_URI: "http://mlflow:5000"}`, `volumes: [./backend/models:/app/backend/models:ro, chroma_data:/app/chroma_db, mlruns_data:/app/mlruns, ./output:/app/output, ./logs:/app/logs]`
  - **Keep** healthchecks on postgres and redis
  - **Keep** the `volumes:` top-level block with `postgres_data`, `chroma_data`, `mlruns_data`
  - **Do not add** caddy_data or caddy_config here (prod-only)

### Phase 4b — Create `docker-compose.override.yml` (dev)

- [x]  [US2] Create `docker-compose.override.yml` at project root with dev overrides only:
  - `postgres`: add `ports: ["5432:5432"]`
  - `redis`: add `ports: ["6379:6379"]`
  - `backend`: add `command: uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000`, add `ports: ["8000:8000"]`, add bind mounts `volumes: [./backend:/app/backend, ./ui:/app/ui]` (in addition to base volumes — compose merges these)
  - `ui`: add `ports: ["8501:8501"]`
  - `mlflow`: add `ports: ["5000:5000"]`, override `command` to use SQLite for dev: `mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root /mlruns`

---

## Phase 5: User Story 3 — Secrets Management (Priority: P1) 🎯

**Goal**: API keys are injected via Docker secrets (file-based), never visible in `docker inspect` env listing. Falls back to env vars for dev.

**Independent Test**: Stop backend; create `secrets/api_key` with a test value; start prod stack → backend starts and `GET /health` succeeds → `docker inspect lersha-backend` shows no `API_KEY` in `Env` array.

- [x]  [US3] Create `docker-compose.prod.yml` at project root — **this is the primary deliverable file for Phases 5, 6, 7, 8**. Structure:
  - Top-level `secrets:` block:
    ```yaml
    secrets:
      api_key:
        file: ./secrets/api_key
      gemini_api_key:
        file: ./secrets/gemini_api_key
    ```
  - Top-level `volumes:` block — add `caddy_data:` and `caddy_config:` (postgres_data, chroma_data, mlruns_data still declared in base)
  - `backend` service override: `command: sh -c "uv run alembic upgrade head && uv run gunicorn backend.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000"`, add `secrets: [api_key, gemini_api_key]`
  - `worker` service override: add `secrets: [api_key, gemini_api_key]`
  - *(caddy, mlflow, backup services are added in later phases — add them as empty stubs now or all at once in T013/T015/T016)*

  > **Note**: It is acceptable to create the full `docker-compose.prod.yml` in one go in T011 (including all services below), or to build it incrementally across T011/T012/T013/T015/T016. The implementer should choose whichever is less error-prone.

---

## Phase 6: User Story 4 — Celery Worker in Compose (Priority: P2)

**Goal**: A Redis broker and Celery worker service are fully active in the base compose stack. The worker shares model and ChromaDB mounts with the backend.

**Independent Test**: `docker compose up` → `docker compose logs worker` shows Celery worker online → `uv run celery -A backend.worker inspect active` shows worker connected.

- [x]  [US4] Verify/confirm in `docker-compose.yml` (from T009) that the `worker` service has the correct volume mounts identical to `backend`: `./backend/models:/app/backend/models:ro`, `chroma_data:/app/chroma_db`, `mlruns_data:/app/mlruns`, `./output:/app/output`, `./logs:/app/logs`
  - If any volume is missing from T009's worker definition, add it now
  - Confirm `MLFLOW_TRACKING_URI` is set in worker environment

---

## Phase 7: User Story 5 — MLflow Production Backend (Priority: P2)

**Goal**: MLflow tracking uses PostgreSQL (not SQLite) and S3/GCS for artifacts in production. Experiment records survive container restarts.

**Independent Test**: Start prod stack → restart MLflow container → previously logged runs still visible in MLflow UI at port 5000 (accessed via caddy on `/mlflow` or directly).

- [x]  [US5] Add (or update) the `mlflow` service override in `docker-compose.prod.yml`:
  ```yaml
  mlflow:
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri postgresql://${DB_USER}:${DB_PASSWORD}@postgres:5432/mlflow
      --default-artifact-root s3://${MLFLOW_S3_BUCKET}/artifacts
    environment:
      AWS_ACCESS_KEY_ID: ${AWS_ACCESS_KEY_ID}
      AWS_SECRET_ACCESS_KEY: ${AWS_SECRET_ACCESS_KEY}
  ```
  Add inline comment `# GCS alternative: --default-artifact-root gs://your-bucket/artifacts` and `# GCP: set GOOGLE_APPLICATION_CREDENTIALS in environment instead of AWS vars`

- [x]  [P] [US5] Document the `mlflow` database prerequisite: add a comment block to `docker-compose.prod.yml` (above the `mlflow` service) noting that ops must run `CREATE DATABASE mlflow;` in PostgreSQL before the first production startup, or add a one-time init step. Also add to `quickstart.md` under "Production Deployment Workflow" a step: `docker compose exec postgres psql -U ${POSTGRES_USER} -c "CREATE DATABASE mlflow;"`

---

## Phase 8: User Story 6 — MLflow Model Registry with Fallback (Priority: P2)

**Goal**: `load_prediction_models()` tries the MLflow registry first, falls back silently to local `.pkl` if the registry is unavailable.

**Independent Test**: Stop MLflow container → submit a prediction request → observe `WARNING: MLflow registry unavailable` in backend logs → prediction still succeeds (loaded from `.pkl`).

- [x]  [US6] Refactor `load_prediction_models(model_name: str)` in `backend/core/infer_utils.py`:
  - Build a `pkl_path_map` dict mapping model name → config pkl path
  - Raise `ValueError` immediately for unknown model name (before any loading attempt)
  - Try `mlflow.sklearn.load_model(f"models:/lersha-{model_name}/Production")` in a `try` block
  - On success: call `mlflow.set_tag("model_source", f"registry:models:/lersha-{model_name}/Production")` and log `INFO`
  - On `Exception` (catches network errors + registry errors): log `WARNING` with `exc_info=False` (avoid alarming stack trace for expected fallback), fall through to local pkl logic
  - Try `joblib.load(pkl_path_map[model_name])` in a second `try` block
  - On success: call `mlflow.set_tag("model_source", f"local:{pkl_path_map[model_name]}")` and log `INFO`
  - On `Exception`: log `ERROR` with `exc_info=True`, raise `RuntimeError` wrapping original exception
  - Ensure `import mlflow.sklearn` is added at the top of the file (alongside existing `import mlflow`)
  - Full function must have complete PEP 257 docstring and `-> object` return type annotation

- [x]  [P] [US6] Create `backend/scripts/register_model.py` — documentation-only script:
  - Module-level docstring only; no executable `if __name__ == "__main__"` block
  - Document the training-time registration pattern as a Python comment block:
    ```python
    # Training-time registration (run after model training):
    #
    # import mlflow.sklearn
    # mlflow.sklearn.log_model(
    #     pipeline,
    #     artifact_path="model",
    #     registered_model_name=f"lersha-{model_name}",  # e.g. "lersha-xgboost"
    # )
    #
    # After logging, promote the model in the MLflow UI:
    #   http://mlflow:5000 → Models → lersha-{model_name} → Staging → transition to Production
    #
    # The prediction service (backend/core/infer_utils.py:load_prediction_models)
    # will automatically load from the registry on next startup.
    ```

---

## Phase 9: User Story 7 — PostgreSQL Backups (Priority: P2)

**Goal**: Daily automated database backups appear in `./backups/` with a 30-day retention policy. `make restore-db` restores from a backup file.

**Independent Test**: `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d backup` → `docker compose logs backup` shows scheduler active → within the first minute a backup file appears (use `SCHEDULE=@every 1m` for testing, revert to `@daily`).

- [x]  [US7] Add the `backup` service to `docker-compose.prod.yml`:
  ```yaml
  backup:
    image: prodrigestivill/postgres-backup-local
    restart: unless-stopped
    depends_on:
      - postgres
    environment:
      POSTGRES_HOST: postgres
      POSTGRES_DB: ${POSTGRES_DB:-lersha}
      POSTGRES_USER: ${POSTGRES_USER:-lersha}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-lersha}
      SCHEDULE: "@daily"
      BACKUP_KEEP_DAYS: "30"
      BACKUP_KEEP_WEEKS: "4"
      BACKUP_KEEP_MONTHS: "6"
    volumes:
      - ./backups:/backups
  ```

- [x]  [US7] Add `restore-db` target to `Makefile`:
  - Update `.PHONY` line at the top to include `restore-db`
  - Add the target body:
    ```makefile
    restore-db:
    	@if [ -z "$(BACKUP_FILE)" ]; then \
    		echo "Usage: make restore-db BACKUP_FILE=./backups/<date>.sql.gz"; \
    		exit 1; \
    	fi
    	@echo "Restoring database from $(BACKUP_FILE)..."
    	gunzip -c "$(BACKUP_FILE)" | psql "$(DB_URI)"
    	@echo "Restore complete."
    ```
  - Update `help` target to include under Docker section:
    ```
    	@echo "  make restore-db     Restore PostgreSQL from backup (BACKUP_FILE=path required)"
    ```
  - Update the `docker-build` target to add an `@echo` note:
    ```makefile
    docker-build:
    	@echo "NOTE: For production use docker-compose.prod.yml with docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d"
    	docker build -f backend/Dockerfile -t lersha-backend:latest .
    	docker build -f ui/Dockerfile -t lersha-ui:latest .
    ```

---

## Phase 10: User Story 1b — Caddy in Prod Compose (Priority: P1, compose wiring)

**Goal**: The `caddy` service is declared in `docker-compose.prod.yml` and starts as part of the prod stack, routing HTTPS traffic.

**Independent Test**: `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d caddy` starts without error. `docker compose logs caddy` shows Caddy starting and attempting ACME challenge.

- [x]  [US1] Add the `caddy` service to `docker-compose.prod.yml`:
  ```yaml
  caddy:
    image: caddy:2-alpine
    restart: unless-stopped
    depends_on:
      - backend
      - ui
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
      - caddy_config:/config
  ```
  Confirm `caddy_data` and `caddy_config` are declared in the top-level `volumes:` block of `docker-compose.prod.yml` (added in T011)

---

## Phase 11: User Story 8 — CI/CD Updates (Priority: P3)

**Goal**: `workflow_dispatch` trigger active; compose validation runs in the build job; mypy already in CI (confirmed).

**Independent Test**: Trigger a manual workflow run from GitHub Actions UI on any branch → all three jobs (lint, test, build) run to completion. Build job includes "Validate compose files" step.

- [x]  [P] [US8] Add `workflow_dispatch:` trigger to `.github/workflows/ci.yml` — add it as a third trigger alongside `push` and `pull_request`:
  ```yaml
  on:
    push:
      branches: ["**"]
    pull_request:
      branches: ["main"]
    workflow_dispatch:
  ```

- [x]  [P] [US8] Add compose file validation step to the `build` job in `.github/workflows/ci.yml` — insert **before** the `Build backend image` step:
  ```yaml
  - name: Validate compose files (base + prod)
    env:
      POSTGRES_USER: lersha
      POSTGRES_PASSWORD: lersha
      POSTGRES_DB: lersha
      API_KEY: ci-placeholder
      GEMINI_MODEL: gemini-1.5-pro
      GEMINI_API_KEY: ci-placeholder
      MLFLOW_S3_BUCKET: ci-placeholder-bucket
      AWS_ACCESS_KEY_ID: ci-placeholder
      AWS_SECRET_ACCESS_KEY: ci-placeholder
      DB_USER: lersha
      DB_PASSWORD: lersha
      REDIS_URL: redis://redis:6379/0
    run: docker compose -f docker-compose.yml -f docker-compose.prod.yml config > /dev/null
  ```

---

## Phase 12: Polish & Cross-Cutting Concerns

**Purpose**: End-to-end validation, documentation consistency, and format enforcement.

- [x]  [P] Run `make pre-commit` and fix any ruff or formatting violations introduced by changes to `backend/config/config.py` and `backend/core/infer_utils.py` [P2-PEP]
- [x]  [P] Run `make typecheck` (`uv run mypy backend/`) — fix any type errors from `_read_secret` return type (`str | None`) propagation and `load_prediction_models` return annotation [P2-PEP]
- [x]  [P] Run `docker compose config > /dev/null` (base + override) locally to confirm dev stack is valid
- [x]  [P] Run `docker compose -f docker-compose.yml -f docker-compose.prod.yml config > /dev/null` locally to confirm prod stack is valid (set dummy env vars in shell first)
- [x]  Perform end-to-end smoke test per `quickstart.md`:
  1. `docker compose up` — confirm all base services start, worker appears online
  2. `curl http://localhost:8000/health` → `{"db":"ok","chroma":"ok"}`
  3. `uv run celery -A backend.worker inspect active` → worker connected
  4. `uv run alembic current` → shows head revision
  5. Confirm `docker compose logs backup` shows scheduler message (prod stack)
- [x]  [P] Update `specs/005-infra-prod-hardening/checklists/requirements.md` — mark all checklist items verified after smoke test passes

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup):          No dependencies — start immediately
Phase 2 (Foundational):   Depends on Phase 1 — T001/T002/T003 must complete first
Phase 3 (US1 Caddyfile):  Depends on Phase 2 — but T008 is independent of all Phase 2 tasks
Phase 4 (US2 Compose):    Depends on Phase 2 — T009 must complete before T011
Phase 5 (US3 Secrets):    T011 blocks all remaining compose phases (T012–T017, T019)
Phase 6 (US4 Worker):     Depends on T009 (base compose) and T011 (prod compose)
Phase 7 (US5 MLflow):     Depends on T011 (prod compose file exists)
Phase 8 (US6 Registry):   Independent of all compose tasks — run in parallel with Phase 4–7
Phase 9 (US7 Backup):     Depends on T011 (prod compose file exists)
Phase 10 (US1b Caddy):    Depends on T008 (Caddyfile) and T011 (prod compose)
Phase 11 (US8 CI):        Independent — run in parallel with everything
Phase 12 (Polish):        Depends on all prior phases complete
```

### User Story Dependencies

| Story | Depends On | Can Parallelize With |
|---|---|---|
| US1 (HTTPS/TLS) | Phase 2 only | US3, US6, US8 |
| US2 (Env Separation) | Phase 2 only | US3, US6, US8 |
| US3 (Secrets) | Phase 2 + T009 (US2) | US6, US8 |
| US4 (Worker) | T009 (US2) + T011 (US3) | US5, US7 |
| US5 (MLflow Backend) | T011 (US3) | US4, US7 |
| US6 (Registry Fallback) | Phase 2 only | All others |
| US7 (Backups) | T011 (US3) | US4, US5 |
| US8 (CI/CD) | Phase 2 only | All others |

### Critical Path

```
T001 → T009 → T011 → T013/T017/T019 → T026 (smoke test)
```

### Parallel Opportunities

Within Phase 2: T004, T005, T006, T007 — all fully parallel (different functions/files)  
US6 tasks T015 and T016 — fully parallel (different files)  
US8 tasks T020 and T021 — fully parallel (different sections of `ci.yml`)  
Phase 12: T022, T023, T024, T025, T027 — all fully parallel

---

## Parallel Execution Example: Phase 2

```bash
# All four tasks can run simultaneously:
Task T004: "Add _read_secret() helper to backend/config/config.py"
Task T005: "Apply _read_secret to api_key in backend/config/config.py"
Task T006: "Apply _read_secret to gemini_api_key in backend/config/config.py"
Task T007: "Update .env.example with MLflow S3 and secrets vars"
```

> Note: T004, T005, T006 all touch `config.py`. An LLM agent should do these as one atomic edit. They are logically parallel in that they don't depend on each other's output.

## Parallel Execution Example: US6 (Phase 8)

```bash
# Both tasks fully independent:
Task T015: "Refactor load_prediction_models() in backend/core/infer_utils.py"
Task T016: "Create backend/scripts/register_model.py (docs-only)"
```

---

## Implementation Strategy

### MVP First (US1 + US2 + US3 — the P1 trio)

1. Complete Phase 1: Setup (T001–T003)
2. Complete Phase 2: Foundational (T004–T007)
3. Complete Phase 3: Caddyfile (T008)
4. Complete Phase 4: Compose restructure (T009–T010)
5. Complete Phase 5: Prod compose + secrets wiring (T011)
6. **STOP and VALIDATE**: `docker compose -f docker-compose.yml -f docker-compose.prod.yml config` → exits 0
7. **STOP and VALIDATE**: Dev stack boots with hot reload; prod stack boots with HTTPS

### Full Production Hardening (all 8 user stories)

After MVP validation, continue in order:
- T012 (confirm worker volumes) → immediate
- T013, T014 (MLflow prod backend) → run together
- T015, T016 (registry fallback + register script) → run together
- T017 (backup service) → run together with T018 (Makefile)
- T019 (Caddy in prod compose) → run together with T020, T021 (CI updates)
- T022–T027 (Polish) → run as a final sweep

### Solo Developer Order (sequential, safest)

```
T001 → T002 → T003 → T004-T007 (atomic) → T008 → T009 → T010 → T011 →
T012 → T013 → T014 → T015 → T016 → T017 → T018 → T019 → T020 → T021 →
T022 → T023 → T024 → T025 → T026 → T027
```

---

## Notes

- `[P]` tasks modify different files; a single developer should still do them sequentially but a team can parallelize
- T011 is the largest single task — it scaffolds the entire `docker-compose.prod.yml`. Consider creating it in one atomic edit covering all services (Caddy, MLflow, backup, backend/worker secrets, volumes)
- T009 and T010 **replace** the existing `docker-compose.yml` — do not create new files; the base file is restructured in place
- `secrets/api_key` and `secrets/gemini_api_key` files must be created manually by the operator before production startup — they are deliberately excluded from this task list (they contain real secrets)
- The `mlflow` PostgreSQL database (`CREATE DATABASE mlflow;`) must exist before the first prod startup — documented in T014 and quickstart.md
- After T022 and T023, run `git add -p` to review each diff before committing
- Commit message format per constitution: `feat(infra): [description] [P11-CONT]` / `feat(config): [P5-CONFIG]` / etc.
