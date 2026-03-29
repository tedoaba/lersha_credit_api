# Feature Specification: Application-Level Security & Reliability Hardening

**Feature Branch**: `004-harden-app-security`
**Created**: 2026-03-29
**Status**: Draft
**Input**: User description: "Harden the Lersha Credit Scoring System at the application code level. All changes are in Python/config files only — no infrastructure."

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Database Schema is Always Applied Before Traffic (Priority: P1)

As a **platform operator**, I want the database schema to be automatically applied at startup so that the application never starts against a mismatched or empty schema, eliminating manual migration steps and preventing data-access errors in production.

**Why this priority**: Schema drift between code and database is the most common cause of production outages. Automated, sequential schema management is a P1 safety requirement.

**Independent Test**: Deploy the application from a fresh database. Verify all tables exist and the application serves requests successfully without any manual SQL execution.

**Acceptance Scenarios**:

1. **Given** a fresh PostgreSQL database with no tables, **When** the application starts, **Then** all required tables are created automatically via migration scripts before the API begins accepting traffic.
2. **Given** an existing database with a previous schema version, **When** the application restarts after a schema update, **Then** only the delta migration is applied, is idempotent, and the application starts successfully.
3. **Given** a migration script that contains an error, **When** the application attempts to start, **Then** the application fails to start with a clear error message rather than serving traffic against a broken schema.
4. **Given** the legacy `db_init.py`, **When** it runs, **Then** it performs only data loading (CSV → database rows) and does not issue any CREATE TABLE or DDL statements.

---

### User Story 2 — Long-Running Inference Requests Do Not Block the API (Priority: P1)

As an **API consumer**, I want credit scoring inference jobs to be processed asynchronously through a job queue so that I receive an immediate job ID and can poll for results, preventing HTTP timeouts on heavy workloads.

**Why this priority**: Synchronous blocking requests cause HTTP timeouts and degrade the entire API for all concurrent users. Decoupling inference from request handling is essential for reliability.

**Independent Test**: Submit a prediction request and verify an immediate `202 Accepted` response with a job ID is returned, then poll the results endpoint until the job completes.

**Acceptance Scenarios**:

1. **Given** a valid inference payload submitted to the predict endpoint, **When** the request is received, **Then** a job ID is returned immediately (under 500 ms) with a `202 Accepted` status.
2. **Given** a queued inference job, **When** the worker processes the job, **Then** the job status transitions from `queued` → `processing` → `completed` (or `failed`) and results are persisted to the database.
3. **Given** a worker that encounters an unhandled exception during inference, **When** the exception occurs, **Then** the job status is set to `failed` with a descriptive error message, and the worker continues processing subsequent jobs.
4. **Given** the Redis broker is temporarily unreachable, **When** a predict request is submitted, **Then** the API returns a `503 Service Unavailable` with an informative message rather than an unhandled exception.

---

### User Story 3 — API Endpoints Are Protected Against Excessive Requests (Priority: P2)

As a **system administrator**, I want rate limiting enforced on the prediction endpoint so that no single client can overwhelm the service, ensuring fair access and protecting downstream resources from abuse.

**Why this priority**: Unthrottled prediction requests can exhaust worker queues and database connection pools, making the service unavailable to legitimate users.

**Independent Test**: Submit 11 consecutive prediction requests from the same IP within one minute and verify the 11th request is rejected with a `429 Too Many Requests` response.

**Acceptance Scenarios**:

1. **Given** a client making 10 prediction requests within a 60-second window, **When** each request is sent, **Then** all 10 requests are accepted with a `202` status.
2. **Given** the same client making an 11th prediction request within the same 60-second window, **When** the request is sent, **Then** a `429 Too Many Requests` response is returned with a clear error message.
3. **Given** a rate-limited client who waits for the window to reset, **When** a new request is sent after the window expires, **Then** the request is accepted normally.
4. **Given** two different client IPs each sending 10 requests in a minute, **When** requests are processed, **Then** each IP is counted independently and neither is rejected.

---

### User Story 4 — All Application Logs Are Structured and Machine-Parseable (Priority: P2)

As an **operations engineer**, I want all application logs emitted as structured JSON so that I can ingest them into log aggregation systems, create dashboards, and set up alerts without custom log parsing rules.

**Why this priority**: Unstructured logs cannot be reliably searched or alerted on in production observability tooling. JSON logs are a prerequisite for operational visibility.

**Independent Test**: Start the application and generate at least one log event; pipe the container's stdout through a JSON parser and confirm all lines are valid JSON with expected fields.

**Acceptance Scenarios**:

1. **Given** the application is running, **When** any log event occurs (startup, request, error), **Then** the log line is valid JSON containing at least `timestamp`, `level`, `name`, and `message` fields.
2. **Given** an exception is logged, **When** the log entry is parsed, **Then** the error details are captured within the structured JSON, not appended as unstructured text.
3. **Given** any module in the codebase calls `get_logger(__name__)`, **When** that logger emits a message, **Then** the output is in the same consistent JSON format without requiring changes to the calling module.

---

### User Story 5 — Every HTTP Request Is Traceable via a Unique ID (Priority: P2)

As a **developer or support engineer**, I want every API request to carry a unique request ID in its response headers so that I can correlate client errors with server-side log entries during debugging.

**Why this priority**: Without request correlation IDs, debugging production issues requires guesswork and is extremely time-consuming.

**Independent Test**: Send an HTTP request with no `X-Request-ID` header; verify the response includes an `X-Request-ID` header with a UUID value. Send a request with a custom `X-Request-ID`; verify the same value is echoed back.

**Acceptance Scenarios**:

1. **Given** an HTTP request without an `X-Request-ID` header, **When** the server processes it, **Then** the response includes an `X-Request-ID` header containing a newly generated UUID.
2. **Given** an HTTP request with an `X-Request-ID` header set by the client, **When** the server processes it, **Then** the response echoes back the same `X-Request-ID` value.
3. **Given** any log line produced during request handling, **When** the log entry is examined, **Then** it includes the same request ID for the entire lifecycle of that request.

---

### User Story 6 — The Health Endpoint Reflects True System Readiness (Priority: P2)

As a **container orchestrator or load balancer**, I want the `/health` endpoint to actively probe all critical dependencies and return a degraded status if any dependency is unavailable, so that traffic is not routed to an instance that cannot serve valid requests.

**Why this priority**: A static health check that always returns 200 provides no operational value. A real liveness probe prevents routing traffic to broken instances.

**Independent Test**: Stop the database container; call `/health` and confirm a `503` response. Restart the database; call `/health` and confirm a `200` response with `{"db":"ok","chroma":"ok"}`.

**Acceptance Scenarios**:

1. **Given** all dependencies (database and vector store) are reachable, **When** `GET /health` is called, **Then** a `200 OK` response is returned with a body of `{"db":"ok","chroma":"ok"}`.
2. **Given** the database is unreachable, **When** `GET /health` is called, **Then** a `503 Service Unavailable` response is returned indicating the database is unhealthy.
3. **Given** the vector store is unreachable, **When** `GET /health` is called, **Then** a `503 Service Unavailable` response is returned indicating the vector store is unhealthy.
4. **Given** the application runtime environment, **When** the container orchestrator polls `/health` every 30 seconds, **Then** the health check completes within 10 seconds and returns an accurate status.

---

### User Story 7 — Transient Failures in AI Calls Are Automatically Retried (Priority: P3)

As an **end user**, I want the system to automatically retry transient failures when calling external AI services so that occasional network blips or rate-limit responses do not surface as errors, improving overall reliability.

**Why this priority**: Generative AI APIs experience occasional transient errors. Silent retries drastically reduce user-visible failures without changing the interface.

**Independent Test**: Simulate an AI API returning a `503` error; verify the system retries up to 3 times with exponential backoff before propagating the error.

**Acceptance Scenarios**:

1. **Given** an AI service call that fails on the first attempt with a transient error, **When** the retry logic is triggered, **Then** the system retries up to 3 times with increasing delays between attempts.
2. **Given** an AI service call that succeeds on the second attempt, **When** the retry completes, **Then** the successful result is returned to the caller transparently.
3. **Given** an AI service call that fails on all 3 attempts, **When** all retries are exhausted, **Then** the original exception is re-raised to the caller with a clear error.

---

### User Story 8 — The API Supports Multiple Concurrent Workers Without Configuration Changes (Priority: P3)

As a **DevOps engineer**, I want the API to be capable of running with multiple process workers in production so that the service can handle concurrent load without horizontal scaling of containers.

**Why this priority**: A single-process server is a bottleneck under any meaningful production load. Multi-worker capability is a standard production requirement.

**Independent Test**: Start the application using the production process manager command documented in the Dockerfile and verify multiple worker processes are spawned and each independently serves requests.

**Acceptance Scenarios**:

1. **Given** the production startup command is executed, **When** the service starts, **Then** at least 4 worker processes are active and each can independently resolve a health check request.
2. **Given** a worker process that crashes, **When** the crash occurs, **Then** the process manager automatically restarts the worker without affecting other running workers or in-flight requests.

---

### User Story 9 — Database Connections Are Efficiently Pooled and Resilient (Priority: P3)

As a **platform operator**, I want the application's database connection pool to be correctly sized and include resilience features so that connections are not exhausted under normal load and stale connections are automatically discarded.

**Why this priority**: Poorly configured connection pools are a leading cause of database-related outages at scale.

**Independent Test**: Under a load test generating 50 concurrent requests, observe database connection count remains bounded (≤ 30) and all requests succeed without connection errors.

**Acceptance Scenarios**:

1. **Given** 50 concurrent API requests, **When** each request interacts with the database, **Then** the total number of open database connections does not exceed the configured pool ceiling.
2. **Given** a database connection that has been idle long enough to be invalidated by the database server, **When** that connection is retrieved from the pool, **Then** it is tested and replaced automatically before being used.

---

### User Story 10 — Code Quality and Type Correctness Are Enforced Automatically (Priority: P3)

As a **developer**, I want pre-commit hooks and static type checking to run automatically before code is committed so that style violations, type errors, and common bugs are caught locally before reaching CI.

**Why this priority**: Shifting quality checks left to the developer machine reduces CI failures, code review friction, and production defects.

**Independent Test**: Install pre-commit hooks; introduce a deliberate style violation into a Python file; attempt to commit; verify the commit is blocked and the violation is reported.

**Acceptance Scenarios**:

1. **Given** pre-commit hooks are installed in a developer's local clone, **When** a commit is attempted with a linting violation, **Then** the commit is blocked and the violation is reported with the file name and line number.
2. **Given** a clean codebase, **When** `mypy` type checking is run against the backend package, **Then** zero type errors are reported.
3. **Given** a CI pipeline, **When** a pull request is submitted, **Then** both linting (ruff) and type checking (mypy) steps run and must pass before the PR can be merged.

---

### Edge Cases

- What happens when Redis is down at startup — should the API start in a degraded mode or refuse to start?
- How does the health check behave if the vector store responds slowly (near the 10-second timeout)?
- What happens if a migration is partially applied (e.g., interrupted mid-run) — will the next startup resume or fail?
- What happens when a Celery worker picks up a job for a `job_id` that no longer exists in the database?
- How does rate limiting behave behind a reverse proxy where all requests share the same IP?

---

## Requirements *(mandatory)*

### Functional Requirements

**Migration & Schema Management**

- **FR-001**: The system MUST automatically apply all pending database schema migrations to completion before the API begins accepting HTTP traffic on each startup.
- **FR-002**: The system MUST support sequential, versioned schema migrations where each migration is uniquely identified and tracked to prevent double-application.
- **FR-003**: The data-loading script MUST operate exclusively on existing tables and MUST NOT issue CREATE TABLE, ALTER TABLE, or any other DDL statements.

**Asynchronous Job Queue**

- **FR-004**: The predict endpoint MUST enqueue each inference request as a background job and return an immediate acknowledgement response with a job identifier.
- **FR-005**: The worker process MUST update the job status record to `processing` immediately upon dequeuing, before executing any inference logic.
- **FR-006**: The worker process MUST persist the inference result to the database on success, or persist a structured error message on failure, and update the final job status accordingly.
- **FR-007**: The predict endpoint MUST NOT directly execute any inference logic or block the HTTP response thread on inference computation.

**Rate Limiting**

- **FR-008**: The predict endpoint MUST enforce a maximum of 10 requests per IP address per 60-second rolling window.
- **FR-009**: Requests exceeding the rate limit MUST receive a `429 Too Many Requests` response with a human-readable message.
- **FR-010**: Rate limit counters MUST be tracked per unique client identifier (remote IP address by default).

**Structured Logging**

- **FR-011**: All log output from the backend application MUST be emitted as newline-delimited JSON to standard output.
- **FR-012**: Every log entry MUST include at minimum: `asctime` (ISO timestamp), `levelname`, `name` (logger name), and `message` fields.
- **FR-013**: All modules MUST obtain loggers via a single shared `get_logger(name)` factory function; no module may configure logging independently.

**Request Tracing**

- **FR-014**: The application MUST process every incoming HTTP request through middleware that assigns a unique trace identifier before routing.
- **FR-015**: The trace identifier MUST be sourced from the inbound `X-Request-ID` header if present; otherwise a new UUID must be generated.
- **FR-016**: The trace identifier MUST be included in the outbound response `X-Request-ID` header for every response.

**Live Health Check**

- **FR-017**: The `GET /health` endpoint MUST actively probe the database by executing a lightweight query (e.g., `SELECT 1`) on each call.
- **FR-018**: The `GET /health` endpoint MUST actively probe the vector store by invoking its heartbeat or ping API on each call.
- **FR-019**: The endpoint MUST return `200 OK` with `{"db":"ok","chroma":"ok"}` only when all dependency probes succeed.
- **FR-020**: The endpoint MUST return `503 Service Unavailable` with a body identifying the failing dependency when any probe fails.

**Retry Logic**

- **FR-021**: Calls to external AI services MUST be automatically retried up to 3 times on transient failure.
- **FR-022**: Retry attempts MUST use exponential backoff with a minimum of 2 seconds and a maximum of 10 seconds between attempts.
- **FR-023**: All retry attempts MUST be exhausted before propagating the exception to the caller.
- **FR-024**: HTTP client calls from the UI layer MUST include explicit connection and read timeouts to prevent indefinite hanging.

**Multi-Worker Support**

- **FR-025**: The production startup configuration MUST support launching multiple worker processes without additional configuration changes.
- **FR-026**: The production startup command MUST be documented within the backend container definition.

**Connection Pool Management**

- **FR-027**: The database engine MUST be configured with a minimum pool size, maximum overflow, keep-alive probing, and connection recycling interval.
- **FR-028**: The connection pool MUST automatically discard and replace any connection that fails a liveness test before returning it to application code.

**Code Quality Automation**

- **FR-029**: A pre-commit configuration MUST be present at the repository root and MUST include linting and formatting hooks that run before each commit.
- **FR-030**: A static type checking configuration MUST be present in the project manifest, targeting the backend package, with strict mode disabled but untyped function checking and missing import tolerance enabled.
- **FR-031**: Both pre-commit hooks and type checking MUST be executable via named Makefile targets for consistent developer and CI invocation.
- **FR-032**: CI pipeline MUST run type checking as a required step in the lint job.

### Key Entities

- **Migration Version**: A uniquely identified, ordered record of schema changes applied to the database. Tracks what has been applied and what is pending.
- **Inference Job**: A persisted record with a unique ID, current status (`queued`, `processing`, `completed`, `failed`), input payload, output result, and error message fields.
- **Rate Limit Window**: A time-bounded counter keyed by client identifier tracking the number of requests within a rolling period.
- **Request Trace ID**: A UUID assigned to each HTTP request lifecycle, propagated through middleware and included in both logs and response headers.
- **Health Probe Result**: A per-dependency outcome (ok / error) aggregated into a single health response payload.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The application reaches a healthy serving state from a fresh database without any manual operator intervention, 100% of the time.
- **SC-002**: A prediction request returns an acknowledgement response in under 500 milliseconds regardless of inference duration.
- **SC-003**: A client submitting 11 prediction requests in 60 seconds receives exactly 10 successful responses and 1 rejection — no more, no less.
- **SC-004**: 100% of application log lines produced during normal operation are valid, parseable JSON with all required fields present.
- **SC-005**: Every API response carries an `X-Request-ID` header with a consistent identifier that matches associated log entries.
- **SC-006**: The `/health` endpoint accurately reflects system state within 10 seconds: returning `200` when all dependencies are healthy and `503` immediately when any dependency is unreachable.
- **SC-007**: External AI service calls succeed on retry after a simulated first-attempt failure with no user-visible error.
- **SC-008**: The application serves concurrent requests from multiple workers under load without exhausting database connections (connection count ≤ 30 with 50 concurrent clients).
- **SC-009**: A commit containing a linting violation is blocked by pre-commit hooks with a clear report of the offending line.
- **SC-010**: Static type checking of the entire backend package completes with zero reported errors on the current codebase.

---

## Assumptions

- All changes are confined to Python source files and project configuration files; no changes to infrastructure provisioning, Kubernetes manifests, or cloud provider settings are in scope.
- The existing PostgreSQL and ChromaDB services defined in `docker-compose.yml` will remain the target data stores; no additional data stores are introduced.
- The Redis service required for the job queue is already defined or will be added to `docker-compose.yml` as a service entry (not a new infrastructure resource), consistent with the Python/config-only constraint.
- The Celery worker runs as a separate process/container launched from the same codebase; no new repository is required.
- Rate limiting uses the client's remote IP address as the default key; deployments behind a trusted reverse proxy that sets a forwarded-IP header will need an operator note but are out of scope for this specification.
- Pre-commit hooks are a developer-local tool; they are not enforced by the container image at runtime, only in CI and local environments.
- The `mypy` strict mode is not enabled; the goal is warning coverage without requiring full type annotation of the existing codebase.
- The multi-worker process manager documented in the Dockerfile is a production-mode configuration; the `make api` development target continues to use a single-process reload server.
- All dependency version constraints specified in the implementation notes are treated as minimum versions; exact pins are managed by the lock file generated after `uv sync`.
- The `db_init.py` data-loading responsibility (CSV → PostgreSQL) is preserved; its DDL responsibilities are fully replaced by the migration system.
