# Feature Specification: RAG System Hardening

**Feature Branch**: `007-rag-service-hardening`
**Created**: 2026-04-02
**Status**: Draft
**Input**: User description: "Harden the RAG system with service architecture, prompt versioning, caching, and dedicated explain endpoint."

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Credit Officer Requests an AI Explanation for a Prediction (Priority: P1)

A credit officer has submitted a loan application batch for scoring. After scoring completes, the officer navigates to the prediction result for a specific farmer and requests an AI-generated explanation of why a particular credit decision was made. The system retrieves relevant policy and feature-definition knowledge documents, constructs a versioned prompt, and returns a plain-language explanation within seconds.

**Why this priority**: This is the primary user-facing value of the RAG system. Every other capability (caching, versioning, auditing) exists to support the quality and reliability of this explanation.

**Independent Test**: Can be fully tested by submitting a prediction job, calling the explain endpoint with a valid `job_id` and `record_index`, and asserting that the response contains a non-empty explanation, the farmer UID, retrieved document IDs, and a prompt version.

**Acceptance Scenarios**:

1. **Given** a completed prediction job exists with at least one record, **When** an authorised user calls the explain endpoint with valid `job_id` and `record_index`, **Then** the system returns a 200 response containing a non-empty natural-language explanation, the farmer UID, a list of retrieved document IDs, the prompt version used, a cache-hit flag, and the latency in milliseconds.
2. **Given** the same inputs are submitted a second time within 24 hours, **When** the explain endpoint is called again, **Then** the system returns the same explanation with `cache_hit: true` and a significantly lower latency.
3. **Given** an invalid `job_id` is provided, **When** the endpoint is called, **Then** the system returns a 404 error with a clear message.

---

### User Story 2 — System Administrator Updates the Explanation Prompt (Priority: P2)

A data scientist or system administrator wants to improve explanation quality by updating the AI prompt template. They publish a new prompt version (e.g., v2) and configure the system to use it without redeploying the application. Existing cached explanations based on the old prompt remain valid and untouched.

**Why this priority**: Prompt versioning is essential for iterating on explanation quality in production without code deployments and without invalidating previously cached explanations inadvertently.

**Independent Test**: Can be tested by switching the configured prompt version environment variable to a new version, submitting an explain request, and verifying the response `prompt_version` field reflects the new version and a new cache entry is created separately from the old-version cache.

**Acceptance Scenarios**:

1. **Given** two prompt versions (v1 and v2) are available and the system is configured to use v2, **When** an explain request is made, **Then** the response `prompt_version` field equals `v2` and the cache key is different from the equivalent v1 request.
2. **Given** prompts are stored as versioned files, **When** the configured version does not exist, **Then** the system returns a clear configuration error rather than generating a malformed explanation.

---

### User Story 3 — Compliance Team Reviews the Explanation Audit Trail (Priority: P3)

A compliance officer needs to verify that all AI explanations generated for credit decisions are traceable. They query the audit log to confirm that each explanation records which documents were retrieved, what was generated, and how long it took.

**Why this priority**: Regulatory and internal governance requirements mandate that AI-assisted credit decisions are auditable and reproducible. The audit log underpins trust in the system.

**Independent Test**: Can be tested by generating an explanation, then querying the audit log table to confirm a new entry exists with the originating query, retrieved document IDs, generated text, and latency in milliseconds.

**Acceptance Scenarios**:

1. **Given** an explain request is processed (cache miss), **When** the audit log is queried, **Then** an entry exists containing the query text, retrieved document IDs, final explanation text, latency in milliseconds, and whether the result was served from cache.
2. **Given** an explain request is served from cache (cache hit), **When** the audit log is queried, **Then** a new entry is still recorded, marking `cache_hit: true` and the retrieval latency.
3. **Given** a document retrieval occurs independently of explain, **When** the audit log is queried, **Then** an entry captures the retrieval query, documents returned, and latency.

---

### Edge Cases

- What happens when no documents meet the similarity threshold? The system should return an explanation stating that insufficient knowledge context was available, rather than generating a hallucinated response.
- What happens when the in-memory or distributed cache is unavailable? The system should degrade gracefully — proceed without caching, generate the explanation, log a warning, and still record the audit entry.
- What happens when the AI generation service is temporarily unavailable? The system should return a 503 with a retry-after hint rather than a partial or empty explanation.
- What happens when `shap_dict` is empty or malformed? The system should validate input and return a 422 with a descriptive validation error.
- What happens if the cache key collides due to floating-point JSON serialisation differences? The canonical JSON serialisation of SHAP values must be deterministic to prevent false cache misses.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST expose a dedicated endpoint for requesting AI-generated explanations for credit predictions, accepting a job identifier, a record index within that job, and a model name.
- **FR-002**: The system MUST return a structured response from the explain endpoint that includes the farmer UID, prediction outcome, natural-language explanation, IDs of retrieved knowledge documents, a flag indicating whether the result was served from cache, the prompt version used, and the end-to-end latency.
- **FR-003**: The system MUST retrieve relevant knowledge documents using semantic similarity against documents categorised as feature definitions or policy rules, filtered by a minimum similarity threshold, and ordered by relevance.
- **FR-004**: The system MUST support multiple named prompt versions stored as external configuration files, with the active version controlled by a runtime configuration value rather than code changes.
- **FR-005**: The system MUST cache AI-generated explanations using a deterministic cache key derived from the prediction outcome, the SHAP feature-importance values (in canonical form), and the active prompt version, with a 24-hour time-to-live.
- **FR-006**: The system MUST serve a cached explanation immediately on a cache hit, without invoking the AI generation service, and indicate the cache hit in the response.
- **FR-007**: The system MUST write an audit log entry after every retrieval and every explanation generation event, recording the query, retrieved document identifiers, generated text (or null on retrieval-only events), latency in milliseconds, and cache-hit status.
- **FR-008**: The system MUST encapsulate all retrieval and explanation logic within a cohesive service component with clearly defined interfaces for retrieval and explanation operations, independent of the HTTP transport layer.
- **FR-009**: The system MUST include unit tests that exercise the retrieval and explanation service methods with all external dependencies mocked, including assertion of audit log writes.
- **FR-010**: The system MUST include integration tests that submit a prediction job, call the explain endpoint, and verify the explanation is non-empty and an audit log entry was created.
- **FR-011**: The explain endpoint MUST be registered under the `/v1/explain` path prefix and grouped under the "Explain" API tag.
- **FR-012**: The query used to retrieve knowledge documents MUST be constructed from the prediction outcome and the top SHAP feature-importance values in a consistent, reproducible format.

### Key Entities

- **Prediction Job**: Represents a completed batch scoring run; identified by a job ID and contains one or more scored farmer records.
- **Farmer Record**: A single scored entry within a prediction job; carries a farmer UID, a prediction outcome, and a SHAP feature-importance dictionary.
- **Retrieved Document**: A knowledge base entry (feature definition or policy rule) matched by semantic similarity; carries an ID, content text, category, and similarity score.
- **Explanation**: The AI-generated natural-language text that contextualises a credit prediction for a farmer, derived from retrieved documents and a versioned prompt.
- **Explanation Cache Entry**: A stored explanation keyed on a deterministic hash of the prediction, SHAP values, and prompt version; valid for 24 hours.
- **Prompt Version**: A named, externally stored template defining the structure and instructions given to the AI model for explanation generation.
- **Audit Log Entry**: A persistent record of every retrieval or explanation event, capturing timing, inputs, outputs, and cache status for regulatory traceability.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users receive a complete explanation response within 3 seconds for cache-miss requests and within 200 milliseconds for cache-hit requests under normal operating conditions.
- **SC-002**: Repeated explain requests with identical inputs within a 24-hour window return cached responses 100% of the time, avoiding redundant AI generation calls.
- **SC-003**: Every explain and retrieval event produces a corresponding audit log entry, achieving 100% audit coverage with zero silent failures.
- **SC-004**: Prompt version can be switched without redeploying the application, and the system correctly serves the updated prompt within one request cycle of the configuration change.
- **SC-005**: Unit test suite achieves full coverage of the service retrieval and explanation paths, including cache-hit, cache-miss, and audit-log write branches.
- **SC-006**: Integration tests confirm end-to-end explain flow succeeds (200 response with non-empty explanation) and audit log presence in fewer than 30 seconds of total test execution time.
- **SC-007**: The system degrades gracefully when the cache layer is unavailable — explanations are still generated and served, with a logged warning, and no 5xx response is returned to the client.

---

## Assumptions

- The prediction job store (database) is already accessible by the backend service; the spec does not assume a specific implementation, only that job and record data can be fetched by job ID and record index.
- The knowledge document corpus has already been ingested and is queryable by vector similarity; this spec covers retrieval and explanation, not document ingestion.
- Prompt version files follow a consistent naming convention (e.g., `v1`, `v2`) and are loaded at service startup or on first use; hot-reloading without restart is out of scope for this version.
- SHAP values are provided as a dictionary keyed by feature name with float values; canonical serialisation will sort keys alphabetically to ensure determinism.
- The AI generation service (Gemini) is reachable from the backend service and returns natural-language text; API key management and authentication are handled by the existing infrastructure configuration.
- Cross-encoder re-ranking of retrieved documents is treated as an optional enhancement; the baseline implementation uses similarity-score ordering only.
- Authentication and authorisation for the explain endpoint follow the same mechanism as existing API endpoints in the project; defining a new auth scheme is out of scope.
- Floating-point SHAP values are serialised with a fixed precision to prevent cache-key variance from rounding differences across environments.
- The distributed cache layer is already provisioned as part of the deployment infrastructure; the spec requires only integration with it, not provisioning.
