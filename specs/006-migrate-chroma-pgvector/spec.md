# Feature Specification: Migrate Vector Store from ChromaDB to PostgreSQL pgvector

**Feature Branch**: `006-migrate-chroma-pgvector`  
**Created**: 2026-04-01  
**Status**: Draft  
**Input**: User description: "Migrate the vector store from ChromaDB to PostgreSQL pgvector extension for transactional consistency and unified data layer."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Uninterrupted AI-Powered Credit Assessments (Priority: P1)

As a credit officer or applicant interacting with the Lersha Credit API, I need the AI assistant to retrieve contextually relevant policy rules and feature definitions to generate accurate credit recommendations — without service disruption after the migration.

**Why this priority**: The RAG (Retrieval-Augmented Generation) engine directly underpins credit decision quality. Any regression or unavailability of document retrieval blocks the core value proposition of the platform.

**Independent Test**: Can be fully tested by submitting a credit assessment query and verifying that the AI assistant returns a relevant, context-grounded answer sourced from retrieved documents, independently of any external vector store service.

**Acceptance Scenarios**:

1. **Given** the system has been migrated to the unified database, **When** a user submits a credit-related natural language query, **Then** the AI assistant retrieves the top relevant documents and incorporates them into a coherent, accurate response.
2. **Given** no highly similar documents exist above the relevance threshold, **When** a query is submitted, **Then** the system returns a gracefully degraded response and still processes the query (no crash or unhandled error).
3. **Given** the knowledge store contains categorised documents, **When** a query is submitted, **Then** only documents from approved categories are considered for retrieval.

---

### User Story 2 - Operators Can Ingest & Index Knowledge Documents (Priority: P2)

As a platform operator or DevOps engineer, I need a reliable way to load existing feature definitions and policy rules into the new unified document store so that document data is available from day one of the migration.

**Why this priority**: Without initial data ingestion, the AI assistant has no knowledge base to draw from. This must succeed before any end-user queries can be accurate.

**Independent Test**: Can be fully tested by running the data ingestion process against the existing document source, then confirming documents appear in the unified store with queryable metadata and similarity search capability.

**Acceptance Scenarios**:

1. **Given** a set of existing feature definition documents, **When** the ingestion process is run, **Then** all documents are stored with semantic embeddings and are immediately queryable.
2. **Given** a batch of 1000+ documents, **When** the ingestion process runs, **Then** it completes without timeout or data loss, with success/failure counts logged.
3. **Given** a document already exists in the store, **When** the same document is ingested again, **Then** the system handles the duplicate without failure (upsert or skip with log).

---

### User Story 3 - Retrieval Activity Is Auditable (Priority: P3)

As a compliance officer or platform auditor, I need to review logs of AI retrieval activity — including what was queried, what was retrieved, and what predictions were made — so that I can audit decision transparency and trace any anomalies.

**Why this priority**: Auditability supports regulatory compliance for credit decisions and helps diagnose AI behaviour post-migration. It is not a blocker for day-one operation but is required before production sign-off.

**Independent Test**: Can be tested independently by submitting a query, then inspecting the audit log to confirm the query text, retrieved document references, prediction outcome, model used, and response latency are all captured.

**Acceptance Scenarios**:

1. **Given** a user query is processed, **When** the AI engine retrieves documents and generates a response, **Then** an audit log entry is created capturing query, retrieved document references, prediction, model used, and latency.
2. **Given** no documents were retrieved (below threshold), **When** the audit log is inspected, **Then** the entry still records the query and the empty retrieval result.
3. **Given** multiple queries are processed, **When** audit logs are queried by date range or job ID, **Then** all matching entries are returned accurately.

---

### User Story 4 - Database Schema Is Version-Controlled and Repeatable (Priority: P4)

As a developer or DevOps engineer, I need all database structure changes (new tables, indexes, extensions) to be applied through the existing schema migration system so any environment can be brought to the correct state with a single command.

**Why this priority**: Ensures consistency across development, staging, and production environments. Without this, manual database state divergence becomes a risk.

**Independent Test**: Can be tested by running the migration on a clean database and confirming all tables, indexes, and extensions are present with the expected structure.

**Acceptance Scenarios**:

1. **Given** a fresh database instance, **When** the migration is applied, **Then** the semantic document table, audit log table, vector similarity index, and category index are all created without error.
2. **Given** the migration has already been applied, **When** it is run again, **Then** the operation is idempotent — no errors and no duplicate objects.
3. **Given** a rollback is triggered, **When** the migration is reverted, **Then** all created objects are cleanly removed without orphaned state.

---

### Edge Cases

- What happens when the similarity threshold filters out all candidate documents for a given query?
- How does the system handle documents whose embeddings were generated with a different model than the current query encoder?
- What occurs if the ingestion script is interrupted mid-batch — is partial ingestion recoverable or does it require a full re-run?
- How are duplicate documents (same content, different metadata) treated during ingestion?
- What happens if the vector extension is not available on the target database instance?
- How does retrieval behave when the document table is empty (cold start)?

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST store all semantic knowledge documents (feature definitions, policy rules) in the unified relational database with associated vector embeddings for similarity search.
- **FR-002**: The system MUST retrieve the top matching documents for a given query using vector similarity, filtered by document category and a minimum relevance threshold.
- **FR-003**: The system MUST log every retrieval event in an audit table, capturing the query text, retrieved document references, prediction outcome, model name, associated job identifier, generated response text, and latency.
- **FR-004**: The system MUST support batch ingestion of knowledge documents, processing large volumes reliably with per-run success and failure counts logged.
- **FR-005**: The system MUST apply all required database schema changes (new tables, indexes, extensions) through the version-controlled schema migration tooling, ensuring repeatability across environments.
- **FR-006**: The system MUST remove the dependency on the external ChromaDB service after successful migration, with all document storage and retrieval served by the unified database.
- **FR-007**: The system MUST expose configurable parameters for retrieval behaviour — specifically the maximum number of documents to retrieve per query and the minimum similarity threshold — without requiring code changes.
- **FR-008**: The system MUST maintain or improve current retrieval quality, such that at least 95% of queries that previously returned relevant results continue to do so after migration.
- **FR-009**: The vector similarity index MUST be created to support fast approximate nearest-neighbour search, keeping retrieval latency acceptable at production query volumes.
- **FR-010**: The ingestion process MUST compute semantic embeddings for each document at ingestion time using a consistent, fixed-dimension representation model (384 dimensions).
- **FR-011**: The infrastructure configuration MUST be updated to remove all ChromaDB-related volumes, mounts, and environment references after migration validation is confirmed.

### Key Entities

- **Knowledge Document**: A unit of domain knowledge (feature definition or policy rule) stored with its text content, category, title, metadata, and a semantic vector embedding. Uniquely identified by a system-generated document ID and a stable universally unique identifier.
- **Retrieval Audit Log**: A record of each document retrieval event, linking the query, retrieved document references, prediction, model, job context, generated text, and performance metrics. Used for compliance and debugging.
- **Document Category**: A classification label that scopes retrieval to relevant document types (e.g., feature_definition, policy_rule).
- **Semantic Embedding**: A fixed-length numerical vector representation of document or query text, used to compute content similarity without keyword matching.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All credit assessment queries return contextually relevant answers from the unified document store, with no degradation in answer quality compared to the pre-migration baseline.
- **SC-002**: Document retrieval completes in under 50 milliseconds for 95% of queries under normal production load after similarity indexing is applied.
- **SC-003**: The ingestion process successfully loads all existing knowledge documents in a single run with zero data loss in a test environment of at least 1,000 documents.
- **SC-004**: 100% of retrieval events are captured in the audit log with all required fields populated, verified over a test run of at least 50 queries.
- **SC-005**: The schema migration applies cleanly from zero state on a fresh database, and any environment can replicate the correct schema with a single upgrade command.
- **SC-006**: The system returns a valid (non-error) response when a query yields no documents above the similarity threshold — retrieval failure must not result in an unhandled exception.
- **SC-007**: All ChromaDB volumes, mounts, and service references are fully removed from infrastructure configuration files, with no residual references after successful validation.

---

## Assumptions

- The current ChromaDB collection contains all authoritative knowledge documents and is the source of truth for the initial data migration.
- The unified relational database (PostgreSQL) is already deployed and accessible to all services that previously accessed ChromaDB; no new database infrastructure provisioning is needed.
- The same semantic embedding model will be used at both ingestion time and query time, ensuring embedding-space consistency and accurate similarity comparisons.
- The embedding model produces 384-dimensional vectors; this dimension is fixed and not expected to change after initial deployment.
- The retrieval configuration (number of results, similarity threshold) will be managed via application configuration, making it adjustable without a code deployment.
- The schema migration tooling is already integrated into the deployment pipeline; applying the new migration requires only running the standard upgrade command.
- ChromaDB removal will proceed in two phases: (1) migration and validation, then (2) removal of the dependency and infrastructure — removal is contingent on validation success.
- No changes to the public-facing API contract are required; the migration is entirely internal to the AI retrieval layer.
- Infrastructure (Docker Compose or equivalent) manages service dependencies; cleanup of ChromaDB-related volumes and mounts is within scope.
- The ingestion script is a one-time operational tool executed by engineers; it does not require a user interface.
- The approximate nearest-neighbour index requires a minimum number of documents to be effective; ingestion must be completed before the index produces optimal results.
