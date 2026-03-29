# Feature Specification: Infrastructure Production Hardening

**Feature Branch**: `005-infra-prod-hardening`  
**Created**: 2026-03-29  
**Status**: Draft  
**Input**: User description: "Complete production hardening of the Lersha Credit Scoring System at the infrastructure layer. All changes are in Docker, compose, config, and ops tooling."

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Secure HTTPS Traffic via Reverse Proxy (Priority: P1)

As a system operator, I need all external traffic (API and UI) to be served over HTTPS with automatically-renewed TLS certificates, so that data in transit is protected without manual certificate management.

**Why this priority**: HTTPS is the foundational security requirement for any production service. Without it, all other hardening work is exposed. Let's Encrypt auto-provisioning eliminates the operational burden of certificate rotation.

**Independent Test**: Can be fully tested by pointing a domain at the host, starting the Caddy service alongside backend and UI containers, and confirming that `https://your-domain.com/health` returns a valid JSON response over a verified TLS connection.

**Acceptance Scenarios**:

1. **Given** the Caddy service is running with a valid domain configured, **When** an operator issues an HTTP request to port 80, **Then** the request is permanently redirected to HTTPS (301).
2. **Given** the Caddy service is running, **When** an operator issues an HTTPS request to `/v1/*`, **Then** the request is proxied to the backend service and a valid response is returned.
3. **Given** the Caddy service is running, **When** an operator issues an HTTPS request to `/*` (non-API path), **Then** the request is proxied to the UI service.
4. **Given** a fresh deployment with no prior certificates, **When** Caddy starts for the first time, **Then** it automatically provisions a Let's Encrypt certificate without operator intervention.

---

### User Story 2 - Environment Separation (Dev vs. Prod) (Priority: P1)

As a developer, I need a clean separation between my local development environment and the production deployment configuration, so that I can iterate quickly with hot-reload locally while the production stack runs with hardened, stable settings.

**Why this priority**: Without environment separation, developers risk running production-like configs locally (slow, no hot-reload) or accidentally deploying development settings to production (insecure, unstable).

**Independent Test**: Can be tested independently by running `docker-compose up` in the repo root and confirming the backend reloads on code file changes; and running `docker-compose -f docker-compose.yml -f docker-compose.prod.yml config` to confirm it produces a valid merged config with no host port conflicts.

**Acceptance Scenarios**:

1. **Given** a developer runs `docker-compose up`, **When** a Python source file in `./backend` or `./ui` is modified, **Then** the service restarts and reflects the change without rebuilding the image.
2. **Given** a production deployment using `docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d`, **When** all services start, **Then** the backend runs with multiple workers (gunicorn), bind mounts are absent, and Caddy terminates TLS.
3. **Given** the base `docker-compose.yml`, **When** it is inspected, **Then** it contains no hardcoded environment values and no host port bindings.

---

### User Story 3 - Persistent Secrets Without Leaking Credentials (Priority: P1)

As a security-conscious operator, I need sensitive API keys to be injected via a secure secrets mechanism rather than plain environment variables in compose files, so that credentials are never committed to version control or exposed in `docker inspect` output.

**Why this priority**: Credential leakage is a critical security risk. Docker secrets provide file-based injection that avoids exposing values in environment variable lists.

**Independent Test**: Can be tested by verifying that the backend container reads a known secret value from `/run/secrets/api_key` at runtime, and that the `.env` file and compose files contain no plaintext key values.

**Acceptance Scenarios**:

1. **Given** a `./secrets/api_key` file exists on the host, **When** the backend container starts via the prod compose file, **Then** the application correctly reads the API key from `/run/secrets/api_key`.
2. **Given** no secret file exists but the environment variable `API_KEY` is set, **When** the backend starts, **Then** the application falls back to reading the environment variable without error.
3. **Given** the `secrets/` directory exists, **When** the `.gitignore` is inspected, **Then** the `secrets/` entry is present, preventing accidental commits.

---

### User Story 4 - Asynchronous Background Workers (Priority: P2)

As a system operator, I need a dedicated Celery worker service managed within Docker Compose, so that long-running inference tasks are processed asynchronously and the API remains responsive under load.

**Why this priority**: Celery workers are required for the async prediction pipeline already built in the application layer. Without a Redis broker and worker service defined in Compose, background jobs cannot be dispatched in a containerized deployment.

**Independent Test**: Can be tested by submitting a batch prediction request and then running `celery inspect active` to confirm the task appears in the worker's active task list; or by checking `docker-compose logs worker` for job processing entries.

**Acceptance Scenarios**:

1. **Given** the compose stack is running with a redis and worker service, **When** the worker service starts, **Then** `docker-compose logs worker` shows the Celery worker is online and listening on all queues.
2. **Given** a batch prediction job is submitted to the API, **When** the worker processes it, **Then** results are stored and retrievable via the `/v1/results` endpoint.
3. **Given** the Redis service is unavailable, **When** the worker attempts to start, **Then** it retries connectivity according to its `depends_on` configuration and logs the failure clearly.

---

### User Story 5 - Persistent MLflow Tracking with Cloud Artifact Storage (Priority: P2)

As a data scientist, I need the MLflow tracking server to use a PostgreSQL backend and cloud object storage for artifacts in production, so that experiment records and model artifacts survive container restarts and are not stored on ephemeral local disk.

**Why this priority**: Without a persistent backend, all MLflow experiment data and registered models are lost when the container is recreated. Cloud artifact storage enables team-wide access and long-term retention.

**Independent Test**: Can be tested by restarting the MLflow container and confirming that previously logged runs and registered models are still visible in the MLflow UI.

**Acceptance Scenarios**:

1. **Given** the production MLflow service is configured with a PostgreSQL backend, **When** the MLflow container is restarted, **Then** all previously logged experiment runs remain visible in the tracking UI.
2. **Given** `MLFLOW_S3_BUCKET` is set in the environment, **When** a model is logged, **Then** its artifacts are stored in the configured S3 bucket and accessible from any node with the correct credentials.
3. **Given** `GOOGLE_APPLICATION_CREDENTIALS` is configured, **When** artifact root is set to `gs://bucket/artifacts`, **Then** GCS is used as the artifact store without additional code changes.

---

### User Story 6 - MLflow Model Registry with Filesystem Fallback (Priority: P2)

As an inference engineer, I need the prediction service to load models from the MLflow Model Registry when available, and fall back gracefully to the local `.pkl` file if the registry is unreachable, so that the system continues to operate even if MLflow is unavailable.

**Why this priority**: Resilient model loading prevents a full service outage if the MLflow server is down during a restart or maintenance window.

**Independent Test**: Can be tested by stopping the MLflow service and confirming that a prediction request still returns results (loaded from the local `.pkl` fallback), and that the logs indicate the fallback path was taken.

**Acceptance Scenarios**:

1. **Given** a model is registered as `lersha-{model_name}/Production` in MLflow, **When** the prediction service starts, **Then** it loads the model from the registry.
2. **Given** the MLflow registry is unreachable at startup, **When** the prediction service initialises, **Then** it logs a warning and loads the model from the configured local `.pkl` path.
3. **Given** neither the registry nor the local `.pkl` is available, **When** the service starts, **Then** it raises a descriptive error and fails fast rather than serving incorrect predictions.

---

### User Story 7 - Automated Daily Database Backups (Priority: P2)

As a system operator, I need the PostgreSQL database to be backed up automatically on a daily schedule with configurable retention, so that data can be recovered following accidental deletion or infrastructure failure.

**Why this priority**: Without automated backups, a database failure results in permanent data loss. Daily backups with a 30-day / 4-week / 6-month retention policy cover both short-term incidents and long-term compliance requirements.

**Independent Test**: Can be tested by checking `docker-compose logs backup` for scheduled job entries after startup, and verifying that `.sql.gz` backup files appear in the `./backups/` directory within the scheduled interval.

**Acceptance Scenarios**:

1. **Given** the backup service is running, **When** the scheduled backup window elapses, **Then** a compressed `.sql.gz` file appears in `./backups/` and the service logs confirm success.
2. **Given** backups older than 30 days exist, **When** the backup service runs, **Then** backups exceeding the retention policy are automatically deleted.
3. **Given** a backup file path is provided, **When** `make restore-db` is executed, **Then** the database is restored from the specified backup file.

---

### User Story 8 - Comprehensive Ops Tooling (Makefile, CI, Pre-commit) (Priority: P3)

As a developer or CI pipeline, I need standardised Makefile targets, type-checking integration in CI, and compose file validation in the build pipeline, so that quality gates are enforced consistently across local development and automated workflows.

**Why this priority**: Ops tooling is a force multiplier — once in place it enforces quality automatically. While lower priority than core infrastructure, it prevents regressions and simplifies onboarding.

**Independent Test**: Can be tested by running `make pre-commit`, `make typecheck`, and `make test` locally and confirming all pass with zero errors; and by triggering a `workflow_dispatch` run on any branch in GitHub Actions.

**Acceptance Scenarios**:

1. **Given** the Makefile is updated, **When** `make pre-commit` is run, **Then** all pre-commit hooks execute against the entire codebase and any violations are reported.
2. **Given** the CI workflow is updated, **When** the lint job runs, **Then** `mypy` type-checking executes after `ruff` linting and blocks the pipeline on type errors.
3. **Given** the CI workflow is updated, **When** the build job runs, **Then** `docker-compose config` validates all compose files and the job fails on invalid syntax.
4. **Given** the `workflow_dispatch` trigger is added, **When** a developer manually triggers CI from the GitHub UI on any branch, **Then** the full pipeline executes.

---

### Edge Cases

- What happens when the `secrets/` directory is missing at production startup? The backend must log a clear error identifying which secret path is absent and fall back to env vars gracefully.
- What happens when the S3 bucket is unreachable during MLflow artifact logging? MLflow should surface the error to the operator without silently swallowing it.
- What happens when `make restore-db` is run without specifying a valid backup path? The target must print usage instructions and exit non-zero.
- What happens when `docker-compose.prod.yml` is used without a valid `.env` file? Missing required variables must cause compose to fail at config validation time, not silently at runtime.
- What happens when both `MLFLOW_S3_BUCKET` and `GOOGLE_APPLICATION_CREDENTIALS` are set? The configured `--default-artifact-root` URI scheme takes precedence; the other var is unused.
- What happens when the Caddy service cannot reach the ACME server (air-gapped or firewalled env)? Caddy should fall back to self-signed certificates and log a prominent warning.

---

## Requirements *(mandatory)*

### Functional Requirements

**5.1 — HTTPS / TLS (Caddy Reverse Proxy)**

- **FR-001**: The system MUST include a `Caddyfile` at the project root that routes `/v1/*` requests to the backend service and all other requests to the UI service.
- **FR-002**: The production compose file MUST define a `caddy` service using the `caddy:2-alpine` image, exposing ports 80 and 443, with volumes for `Caddyfile`, `caddy_data`, and `caddy_config`.
- **FR-003**: The `caddy` service MUST depend on `backend` and `ui`, ensuring it starts only after those services are healthy.
- **FR-004**: The system MUST define `caddy_data` and `caddy_config` as named volumes in the production compose file.
- **FR-005**: Caddy MUST auto-provision TLS certificates via Let's Encrypt without operator intervention.

**5.2 — MLflow Production Backend**

- **FR-006**: The production MLflow service MUST be configured with `--backend-store-uri postgresql://${DB_USER}:${DB_PASSWORD}@postgres:5432/mlflow`.
- **FR-007**: The production MLflow service MUST be configured with `--default-artifact-root s3://${MLFLOW_S3_BUCKET}/artifacts` (or `gs://bucket/artifacts` for GCP).
- **FR-008**: The MLflow service MUST have `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` available in its environment for S3 access.
- **FR-009**: `.env.example` MUST document `MLFLOW_S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and the GCP alternative (`GOOGLE_APPLICATION_CREDENTIALS`).

**5.3 — Environment Separation (Compose Override Pattern)**

- **FR-010**: `docker-compose.yml` (base) MUST define all service definitions without hardcoded environment values and without host port bindings.
- **FR-011**: `docker-compose.override.yml` (dev) MUST configure the backend with `uvicorn --reload`, bind mounts for `./backend:/app/backend` and `./ui:/app/ui`, and expose all ports for local access.
- **FR-012**: `docker-compose.prod.yml` (prod) MUST configure the backend with `alembic upgrade head && gunicorn` (4 workers), add Caddy, Redis, worker, and backup services, use the PostgreSQL MLflow backend, and contain no bind mounts.
- **FR-013**: Running `docker-compose up` (without explicit `-f` flags) MUST auto-load `docker-compose.override.yml` for dev mode.
- **FR-014**: Running `docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d` MUST bring up the full production stack.

**5.4 — Secrets Management**

- **FR-015**: `backend/config/config.py` MUST include a `_read_secret(name, env_var)` helper that reads from `/run/secrets/{name}` when the path exists, falling back to `os.getenv(env_var)`.
- **FR-016**: The `_read_secret` helper MUST be applied to `api_key` and `gemini_api_key` configuration fields.
- **FR-017**: The production compose file MUST define a `secrets` block referencing `./secrets/api_key` and `./secrets/gemini_api_key` files, and mount them into the `backend` and `worker` services.
- **FR-018**: The `.gitignore` MUST include a `secrets/` entry.
- **FR-019**: `.env.example` MUST include documentation of cloud-native alternatives: AWS Secrets Manager and GCP Secret Manager.

**5.5 — Celery Worker in Compose**

- **FR-020**: The base `docker-compose.yml` MUST define a `redis` service using `redis:7-alpine` image on port 6379.
- **FR-021**: The base `docker-compose.yml` MUST define a `worker` service using the same image as `backend`, with command `uv run celery -A backend.worker worker --loglevel=info`, depending on `redis` and `postgres`.
- **FR-022**: The `worker` service MUST share the same model and ChromaDB volume mounts as the `backend` service.

**5.6 — MLflow Model Registry**

- **FR-023**: `backend/core/infer_utils.py`'s `load_prediction_models` function MUST first attempt to load from `models:/lersha-{model_name}/Production` via MLflow's model registry.
- **FR-024**: If the MLflow registry is unavailable or the model is not found, the system MUST fall back to loading from the configured local `.pkl` path and log a warning.
- **FR-025**: A training-time registration comment MUST be present in `backend/scripts/` documenting the `mlflow.sklearn.log_model(...)` call with `registered_model_name=f"lersha-{model_name}"`.

**5.7 — PostgreSQL Backups**

- **FR-026**: The production compose file MUST include a `backup` service using `prodrigestivill/postgres-backup-local`, configured with `SCHEDULE=@daily`, `BACKUP_KEEP_DAYS=30`, `BACKUP_KEEP_WEEKS=4`, and `BACKUP_KEEP_MONTHS=6`.
- **FR-027**: The `backup` service MUST mount `./backups:/backups` and depend on `postgres`.
- **FR-028**: The `./backups/` directory MUST be listed in `.gitignore`.
- **FR-029**: The Makefile MUST include a `restore-db` target that accepts a backup file path and pipes it to `psql` using the configured database URI.

**5.8 — Makefile Updates**

- **FR-030**: The Makefile MUST include a `pre-commit` target running `uv run pre-commit run --all-files`.
- **FR-031**: The Makefile MUST include a `typecheck` target running `uv run mypy backend/`.
- **FR-032**: The `docker-build` target MUST log a note directing operators to use `docker-compose.prod.yml` for production builds.
- **FR-033**: The Makefile `help` target MUST be updated to reflect all new targets.

**5.9 — CI/CD Updates**

- **FR-034**: The `.github/workflows/ci.yml` lint job MUST include a `uv run mypy backend/` step executed after the `ruff` linting steps.
- **FR-035**: The `.github/workflows/ci.yml` build job MUST include a step that runs `docker-compose config` (or `docker compose config`) to validate all compose files.
- **FR-036**: The CI workflow MUST include a `workflow_dispatch` trigger to allow manual pipeline execution on any branch.

**5.10 — .env.example Documentation**

- **FR-037**: `.env.example` MUST document `REDIS_URL=redis://redis:6379/0`.
- **FR-038**: `.env.example` MUST document all new variables: `MLFLOW_S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and the GCP alternative `GOOGLE_APPLICATION_CREDENTIALS`.

### Key Entities

- **Caddyfile**: Routing configuration declaring virtual host, proxy targets, and TLS behaviour. Lives at project root.
- **docker-compose.yml (base)**: Service skeleton with shared definitions, no environment specifics, no host ports.
- **docker-compose.override.yml (dev)**: Developer-specific overrides for hot reload and port exposure; auto-loaded by Docker Compose.
- **docker-compose.prod.yml (prod)**: Production overlay adding Caddy, workers, backup, and hardened runtime settings.
- **`_read_secret` helper**: Config-layer function that abstracts Docker secrets / env var fallback for sensitive values.
- **Backup service**: Scheduled container that dumps the PostgreSQL database on a cron schedule and manages retention.
- **MLflow registry entry**: Named model version at `models:/lersha-{model_name}/Production` enabling registry-based model loading.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All eight services (postgres, backend, worker, ui, redis, mlflow, caddy, backup) start successfully and remain healthy when the production stack is launched with a single command.
- **SC-002**: An HTTPS health-check request to `https://your-domain.com/health` returns `{"db":"ok","chroma":"ok"}` within 5 seconds of stack startup.
- **SC-003**: A batch prediction request submitted over HTTPS returns a 202 Accepted response within 3 seconds.
- **SC-004**: The Celery worker comes online and is visible to `celery inspect active` within 30 seconds of stack startup.
- **SC-005**: A database backup file appears in `./backups/` within the first scheduled interval; the backup service logs confirm the operation completed successfully.
- **SC-006**: Restarting the MLflow container does not result in the loss of any previously logged runs or registered models.
- **SC-007**: Stopping the MLflow service does not prevent the prediction API from serving results (fallback to local `.pkl` is transparent to callers).
- **SC-008**: All automated quality gates pass: `make test` achieves ≥ 80% coverage, `make pre-commit` reports zero violations, and `make typecheck` reports zero errors.
- **SC-009**: Running `docker-compose up` (no flags) in a freshly cloned repository starts the development stack with hot-reload active, without requiring any manual configuration.
- **SC-010**: The CI pipeline completes successfully on `workflow_dispatch` for any branch, including compose file validation and mypy type-checking gates.

---

## Assumptions

- A valid public domain name is available and DNS is pointed at the production host for Let's Encrypt to issue certificates. In environments without a public domain, Caddy will issue a self-signed certificate.
- The production host has outbound internet access on port 443 for ACME certificate issuance.
- AWS S3 credentials are available if S3 is the chosen artifact store; GCP credentials are mounted if GCS is chosen. Both are documented but only one needs to be configured per deployment.
- The `./secrets/` directory and its files are pre-created by the operator before the production stack is started; no secret auto-generation is in scope.
- The existing application-layer Celery task definitions (`backend/worker`) are already implemented and functional; this spec covers only the infrastructure wiring.
- Alembic migrations are already authored and up to date; this spec does not include writing new migrations.
- The `prodrigestivill/postgres-backup-local` image is used as the backup solution; custom backup scripts are out of scope.
- Both AWS and GCP artifact store options are documented in `.env.example`, but the deployment operator will activate only one per environment.
- `mypy` is already listed as a dev dependency; no new dependency installation is required for the typecheck target.
- The `workflow_dispatch` trigger applies to the existing `ci.yml` workflow; no new workflow file is created.
- Mobile or browser-based access to the Lersha UI is out of scope for this spec; it focuses solely on the infrastructure layer.
