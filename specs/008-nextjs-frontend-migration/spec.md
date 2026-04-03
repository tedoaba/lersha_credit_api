# Feature Specification: Migrate Streamlit UI to Modern Web Frontend

**Feature Branch**: `008-nextjs-frontend-migration`  
**Created**: 2026-04-03  
**Status**: Draft  
**Input**: User description: "Migrate the Streamlit UI to a Next.js 14 frontend with App Router, Zustand state, and TanStack Query."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Submit a Credit Prediction (Priority: P1)

A credit analyst or field officer visits the web interface and submits a batch of farmer records for credit scoring. They select the data source, initiate the prediction job, and receive confirmation that the job has been queued. The interface immediately begins showing live status updates without requiring a page refresh.

**Why this priority**: This is the core, revenue-generating workflow of the system. Without the ability to submit predictions and track their progress, the entire product has no value. All other user stories depend on a successful prediction having been filed.

**Independent Test**: Can be fully tested by opening the prediction form page, selecting a data source, clicking Submit, and verifying a job ID is returned and status polling begins — all without loading any other page.

**Acceptance Scenarios**:

1. **Given** an authenticated user is on the prediction form, **When** they select a data source and click Submit, **Then** the system queues the job, returns a unique job identifier, and begins displaying a real-time status indicator within two seconds.
2. **Given** a job is in a pending or processing state, **When** the user remains on the page, **Then** the status updates automatically every few seconds without any manual action from the user.
3. **Given** a network error occurs during submission, **When** the user is notified of the failure, **Then** the form remains populated so they can retry without re-entering data.

---

### User Story 2 - Browse and Inspect Prediction Results (Priority: P2)

After a prediction job completes, the analyst browses a structured results table listing all evaluated farmers from that batch. They can click any individual row to open a detailed view showing the credit score, contributing risk factors, a plain-language explanation of the decision, and any confidence metadata.

**Why this priority**: Viewing structured results transforms raw model output into actionable lending decisions. This is the primary deliverable users care about after submitting a job.

**Independent Test**: Can be tested independently by navigating to the results list page with a completed job ID and verifying rows are displayed, then clicking a row to verify the detail view shows score, factors, and explanation text.

**Acceptance Scenarios**:

1. **Given** a completed prediction job, **When** the user navigates to the results page, **Then** a table lists all evaluated farmers with their credit scores and status badges.
2. **Given** the results table is visible, **When** the user clicks a single farmer's row, **Then** a detail page opens showing the credit score, risk factor contributions (visualised as a ranked chart), and a plain-language AI-generated explanation.
3. **Given** a farmer's result has an AI explanation, **When** the detail page loads, **Then** the explanation text and its metadata (source documents, confidence level) are displayed clearly.
4. **Given** the results list contains many records, **When** the user scrolls or pages through the table, **Then** all records load without visible performance degradation.

---

### User Story 3 - Monitor the Dashboard at a Glance (Priority: P3)

A team lead or manager opens the dashboard to quickly understand the current state of the system — how many jobs are queued, how many have completed today, and whether any jobs have failed — without needing to navigate to individual result pages.

**Why this priority**: The dashboard provides operational awareness and reduces context-switching. It is a productivity multiplier but not required for basic prediction-to-decision workflows.

**Independent Test**: Can be tested by navigating to the dashboard root and verifying summary statistics and recent job activity are rendered without needing to submit a new job.

**Acceptance Scenarios**:

1. **Given** a user opens the dashboard, **When** the page loads, **Then** they see summary statistics (jobs pending, completed, failed) and a recent activity list updated within the last 60 seconds.
2. **Given** a job transitions from processing to completed while the dashboard is open, **When** the user does not refresh, **Then** the summary counts update to reflect the new state within 60 seconds.

---

### User Story 4 - Configure the API Key for Secure Access (Priority: P4)

A user or system administrator navigates to the Settings page to enter and persist their API key so that all subsequent requests from the browser are authenticated automatically without requiring the key to be re-entered each session.

**Why this priority**: Authentication is a prerequisite for any backend communication, but it is a one-time setup step. It has lower priority than workflows because it is completed once and then invisible to the user.

**Independent Test**: Can be tested by navigating to the Settings page, entering a key, refreshing the browser, and verifying the key is still present and used in subsequent API calls.

**Acceptance Scenarios**:

1. **Given** a user navigates to Settings, **When** they enter an API key and save it, **Then** the key is stored locally and all API calls from that session onwards include it automatically.
2. **Given** a stored API key exists, **When** the user closes and reopens the browser, **Then** the key is restored and no re-entry is required.
3. **Given** an invalid or missing API key is used, **When** a backend request is made, **Then** the user receives a clear error message prompting them to check their settings.

---

### Edge Cases

- What happens when a job ID referenced in a URL no longer exists in the backend (e.g., expired or deleted)?
- How does the interface behave if the backend API is unreachable when the user submits a prediction?
- What is displayed if a completed batch has zero results (empty dataset)?
- How are partially failed batches represented — where some farmers scored successfully and others errored?
- What happens if a user navigates directly to a result detail URL before the job has completed?
- How does the status polling behave when the browser tab is backgrounded or the device sleeps?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Users MUST be able to submit a prediction request by selecting a data source and initiating the job from a dedicated form page.
- **FR-002**: The system MUST display a real-time job status indicator (pending / processing / completed / failed) that updates automatically without requiring a page reload.
- **FR-003**: Users MUST be able to view a paginated or scrollable table of all farmer results associated with a completed prediction job.
- **FR-004**: Users MUST be able to open an individual farmer's result to see their credit score, ranked risk factor contributions, and an AI-generated plain-language explanation.
- **FR-005**: The AI explanation view MUST include contextual metadata such as source document references and a confidence indicator.
- **FR-006**: Users MUST be able to store an API key via the Settings page, and the key MUST persist across browser sessions without requiring re-entry.
- **FR-007**: All requests to the backend MUST include the stored API key as an authentication credential.
- **FR-008**: The dashboard MUST display a summary of job states (pending, completed, failed counts) and a recent activity feed that reflects data no older than 60 seconds.
- **FR-009**: The system MUST provide clear, user-friendly error messages for network failures, authentication errors, and missing data scenarios.
- **FR-010**: The frontend MUST be deployable as a containerised service that integrates with the existing backend and reverse proxy infrastructure.
- **FR-011**: Status polling MUST cease automatically once a job reaches a terminal state (completed or failed) to avoid unnecessary network load.
- **FR-012**: The interface MUST be navigable via a persistent top navigation bar accessible on every page.

### Key Entities

- **Prediction Job**: Represents a batch credit scoring request. Key attributes: unique identifier, submission timestamp, data source, current status (pending / processing / completed / failed), result count.
- **Farmer Result**: An individual scored record within a completed job. Key attributes: farmer identifier, credit score, risk tier, feature contribution values, explanation text.
- **Explanation**: An AI-generated narrative associated with a farmer result. Key attributes: explanation text, source documents cited, metadata (confidence, generation timestamp).
- **API Key**: A credential stored on the client device that authenticates all backend requests. Key attributes: key value, storage scope (persisted across sessions).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can submit a prediction and see a confirmed job status within 3 seconds of clicking Submit, under normal network conditions.
- **SC-002**: Job status updates appear on screen without manual refresh, with updates visible within 5 seconds of a backend state change, while the job is active.
- **SC-003**: Users can navigate from the results list to an individual farmer detail view in a single click, with the detail page fully loaded within 3 seconds.
- **SC-004**: The API key entered in Settings persists through a full browser close-and-reopen cycle with 100% reliability.
- **SC-005**: The frontend service starts successfully and is reachable within 60 seconds of container startup.
- **SC-006**: All pages remain usable on screens with a minimum viewport width of 768px (tablet and above).
- **SC-007**: 95% of users can complete the end-to-end workflow (submit → monitor → view results → read explanation) without encountering an unhandled error.
- **SC-008**: Building the frontend for production deployment completes without errors.

## Assumptions

- The existing backend REST API will continue to expose all endpoints that the current Streamlit interface consumes (prediction submission, job status, results retrieval, explanation retrieval).
- API authentication is enforced via a fixed API key passed in a request header — no user login, session management, or OAuth flow is required for the frontend itself.
- The primary target device is a desktop or laptop browser; mobile-only support is out of scope for the initial migration.
- The existing reverse proxy (Caddy) will be updated to route frontend and API traffic appropriately; no separate CDN or edge layer is required for the initial release.
- The Streamlit UI will remain available in its current form during the migration and will be decommissioned only after the new frontend is validated in production.
- Data fetched for the dashboard summary may have a staleness window of up to 60 seconds, which is acceptable for operational monitoring.
- The backend's response schemas are stable and versioned; no breaking changes are expected during the frontend development cycle.
- Source data formats and batch sizes used for predictions remain within the limits already supported by the backend.
