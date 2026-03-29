# Research: Infrastructure Production Hardening

**Feature**: `005-infra-prod-hardening`  
**Date**: 2026-03-29

---

## Decision 1: Reverse Proxy — Caddy vs Nginx

**Decision**: Caddy 2 (`caddy:2-alpine`)

**Rationale**: Caddy is the spec-mandated choice. It auto-provisions Let's Encrypt TLS certificates with zero configuration beyond a domain name, which removes the cert rotation toil that Nginx with Certbot introduces. The `reverse_proxy` directive is one line per upstream. The project constitution (P9-SEC) explicitly names Caddy as an acceptable reverse proxy.

**Alternatives considered**:
- **Nginx + Certbot**: More widely deployed but requires a separate Certbot sidecar, renewal cron, and manual reload. More configuration surface area.
- **Traefik**: Feature-rich but significantly more complex label-based config for a project this size.

---

## Decision 2: Compose Override Strategy

**Decision**: Three-file pattern — `docker-compose.yml` (base) + `docker-compose.override.yml` (dev auto-loaded) + `docker-compose.prod.yml` (prod explicit merge)

**Rationale**: Docker Compose natively auto-loads `docker-compose.override.yml` when only `docker-compose.yml` is specified. This means the dev workflow is a zero-friction `docker-compose up` while production requires the explicit `-f` flags, making it impossible to accidentally run dev settings in prod. The constitution (P11-CONT) mandates exactly this pattern.

**Current state**: The existing `docker-compose.yml` already has services but mixes port bindings that must move to the override file. The worker service is commented out and must be uncommented and moved to the base file.

**Alternatives considered**:
- Single file with environment variable conditionals: Fragile, not supported natively.
- Separate `docker-compose.dev.yml`: Works but loses the auto-load benefit.

---

## Decision 3: Secrets Injection — Docker Secrets vs Vault

**Decision**: Docker Compose `secrets` block reading from `./secrets/` files, with `_read_secret()` helper providing env var fallback

**Rationale**: Docker secrets are the lowest-friction production secrets mechanism for Docker-based deployments — they mount as files at `/run/secrets/{name}` inside containers, never appearing in `docker inspect` environment listings. The `_read_secret(name, env_var)` helper maintains backward compatibility with the existing env-var-based dev workflow. The constitution (P9-SEC) mandates Docker Secrets for production.

**Note on cloud alternatives**: AWS Secrets Manager and GCP Secret Manager are documented in `.env.example` as alternatives for cloud-native deployments (e.g., ECS, GKE). These require the AWS SDK or GCP client library and a startup hook; they are out of scope for this sprint but must be documented.

**Alternatives considered**:
- HashiCorp Vault: Overkill for this deployment scale.
- Kubernetes Secrets: Only relevant if migrating to K8s.

---

## Decision 4: MLflow Artifact Store — S3 vs GCS

**Decision**: S3 primary, GCS documented as alternative; both configured via `--default-artifact-root` URI scheme

**Rationale**: S3 is the most widely used object store and MLflow has first-class support. The URI scheme (`s3://` vs `gs://`) is the only difference in the MLflow command, so both can be documented without separate code paths. The spec requires both to be documented in `.env.example`.

**Current state**: MLflow is running with `sqlite:///mlruns/mlflow.db` and local `/mlruns` artifact root — unsafe for multi-worker concurrent writes (constitution P10-OBS explicitly calls this out).

---

## Decision 5: MLflow Model Registry Fallback Strategy

**Decision**: Try-except around `mlflow.sklearn.load_model("models:/lersha-{model_name}/Production")`, fall back to `joblib.load(config.{model}_model_36)` on any `mlflow.exceptions.MlflowException` or `ConnectionError`

**Rationale**: A `try/except` on the MLflow call with a `WARNING`-level log and local `.pkl` fallback satisfies both SC-007 (fallback is transparent to callers) and FR-023/FR-024. It also avoids breaking the existing loading logic during the transition period before models are registered.

**Key implementation detail**: The exception catch must be broad enough to catch network errors (MLflow unreachable) and registry errors (model not promoted to Production stage yet), but narrow enough not to swallow genuine model corruption errors. Use `Exception` at the fallback boundary and re-raise `ValueError` (unknown model name) unconditionally.

---

## Decision 6: PostgreSQL Backup Service

**Decision**: `prodrigestivill/postgres-backup-local` Docker image

**Rationale**: This is a battle-tested, single-concern container that runs `pg_dump`, compresses output, and manages retention by count (days/weeks/months). It requires no custom scripting and its env var interface maps directly to the project's existing `POSTGRES_*` variables. Retention policy of 30 days / 4 weeks / 6 months covers both the constitution requirement (P8-DB: minimum 30-day retention) and the spec FR-026.

**`make restore-db` implementation**: Since the Makefile currently has no restore target, the new target will prompt for a backup file path via `$(BACKUP_FILE)` variable (invoked as `make restore-db BACKUP_FILE=./backups/...sql.gz`) and pipe through `gunzip | psql`. This avoids interactive prompts which are problematic in Make.

---

## Decision 7: Gunicorn + Uvicorn Workers (Production Backend)

**Decision**: `alembic upgrade head && gunicorn backend.main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000`

**Rationale**: The constitution (P11-CONT) mandates this exact setup. The `&&` chaining ensures migrations run before the server starts. Four workers is the standard default for a 2-core host (2× CPU count).

**Dev override**: `uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000` in `docker-compose.override.yml`.

---

## Decision 8: CI Compose Validation

**Decision**: `docker compose config -f docker-compose.yml -f docker-compose.prod.yml` in the build job (runs after `docker-compose.yml` and `docker-compose.prod.yml` are both present)

**Rationale**: `docker compose config` (compose v2) validates merged compose files syntactically and reports missing variables. Running it in CI ensures broken prod configs are caught before the Docker build step.

**Note**: The existing CI uses `docker build` directly — the validation step is additive and non-destructive.
