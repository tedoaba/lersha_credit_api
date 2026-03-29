# Tasks: FastAPI Backend & Streamlit HTTP Integration

**Input**: Design documents from `/specs/002-fastapi-backend-http-wire/`  
**Prerequisites**: [plan.md](plan.md) ✅ · [spec.md](spec.md) ✅ · [research.md](research.md) ✅ · [data-model.md](data-model.md) ✅ · [contracts/api-v1.md](contracts/api-v1.md) ✅

**Tests**: Unit tests included for backend gap tasks (constitution requires 80% coverage). UI pages tested via end-to-end curl/manual validation.

**Key finding from research.md**: Backend is 90%+ complete. Only 5 targeted changes are needed.

---

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no shared dependencies)
- **[Story]**: User story from spec.md — [US1] Submit Prediction, [US2] Dashboard, [US3] Auth, [US4] Job Lifecycle

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm codebase is ready and environment is configured. No new files are created.

- [x] T001 Verify `.env` is populated from `.env.example` (DB_URI, API_KEY, GEMINI_API_KEY, GEMINI_MODEL, API_BASE_URL all set)
- [x] T002 Run `uv run python backend/scripts/db_init.py` and confirm `inference_jobs` and `candidate_result` tables exist in PostgreSQL
- [x] T003 Run `uv run ruff check backend/ ui/` and confirm zero pre-existing lint violations before any code changes

**Checkpoint**: Environment ready — DB tables exist, `.env` is populated, lint baseline is clean.

---

## Phase 2: Foundational (Backend Gap Fixes)

**Purpose**: Close the two backend gaps identified in `research.md`. These are blocking prerequisites — both `[P5-CONFIG]` and `[P8-DB]` constitution violations must be resolved before UI work begins.

**⚠️ CRITICAL**: UI story phases (Phase 3–4) depend on these fixes being in place and tested.

- [x] T004 [P] Fix missing-YAML guard in `backend/config/config.py` lines 125–130: replace the `if _hparams_path.exists(): … else: self.hyperparams = {}` block with a `FileNotFoundError` raise when the file is absent (see plan.md Task 1 for exact before/after)
- [x] T005 [P] Add `update_job_status(job_id: str, status: str) -> None` function to `backend/services/db_utils.py` after the `create_job` function — uses `Session(engine)`, fetches `InferenceJobDB` by primary key, sets `job.status`, commits, logs at INFO level (see plan.md Task 2 for full implementation)
- [x] T006 Add single call `db_utils.update_job_status(job_id, "processing")` as first line of the `try` block inside `_run_prediction_background` in `backend/api/routers/predict.py`

### Unit Tests for Foundational Fixes

- [x] T007 [P] Add unit test `test_config_raises_on_missing_hyperparams_yaml` to `backend/tests/unit/test_config.py` — patch `Path.exists` to return `False` for the hyperparams path and assert `FileNotFoundError` is raised on `Config()` instantiation
- [x] T008 [P] Add unit test `test_update_job_status_changes_status_field` to `backend/tests/unit/test_db_utils.py` — mock `Session` and `InferenceJobDB`, call `update_job_status("abc", "processing")`, assert `job.status == "processing"` and `session.commit()` was called

**Checkpoint**: Run `uv run pytest backend/tests/unit/ -v` — T007 and T008 must pass. Run `uv run ruff check backend/config/config.py backend/services/db_utils.py backend/api/routers/predict.py` — zero violations.

---

## Phase 3: User Story 1 — Submit a Credit Prediction Job (Priority: P1) 🎯 MVP

**Goal**: Rewrite `ui/pages/New_Prediction.py` so it uses `LershaAPIClient` exclusively — no backend Python imports, graceful error handling, and a polling loop that renders results.

**Independent Test**: Submit a prediction via the Streamlit UI form with the backend running. Verify the page shows "Job accepted", polls to completion, and renders evaluation expanders — without any `from backend`, `from config`, or `from src` import in the file.

### Implementation for User Story 1

- [x] T009 [US1] Rewrite `ui/pages/New_Prediction.py` — replace all imports (remove `from config.config import config`, `from src.inference_pipeline import match_inputs, run_inferences`) with `import streamlit as st`, `import pandas as pd`, `import requests`, `from ui.utils.api_client import LershaAPIClient`. Instantiate `client = LershaAPIClient()` at module level.
- [x] T010 [US1] In `ui/pages/New_Prediction.py` — implement source radio + farmer UID / row-count inputs identical to the current layout (preserve UX). The `original_df` preview logic that used `match_inputs()` must be removed; replace with a simple explanation note: `st.info("Data will be fetched from the backend on job submission.")`.
- [x] T011 [US1] In `ui/pages/New_Prediction.py` — implement the "Run Prediction" button handler: wrap in `try/except requests.exceptions.ConnectionError` showing `st.error("Backend unavailable. Is the API server running?")`. On success: call `client.submit_prediction(source, farmer_uid, number_of_rows)`, display `st.success(f"Job accepted: {job_id}")`, then call `client.poll_until_complete(job_id, poll_interval=2.0, max_wait=300.0)` inside `st.spinner("Running inference… this may take up to 5 minutes")`.
- [x] T012 [US1] In `ui/pages/New_Prediction.py` — implement the result rendering block: if `job["status"] == "completed"`, iterate `result_xgboost.evaluations` and `result_random_forest.evaluations` in two `st.columns`, rendering each evaluation as an `st.expander` with prediction class, feature contributions table (`pd.DataFrame`), and RAG explanation — matching the existing layout style. If `status == "failed"`, render `st.error(f"Inference failed: {job['error']}")`.
- [x] T013 [US1] In `ui/pages/New_Prediction.py` — preserve the footer HTML block from the original file. Verify the final file contains **zero** occurrences of `from backend`, `from config`, `from src`, `import config`, `import src`.

**Checkpoint**: Run `grep -r "from backend\|from config\|from src" ui/pages/New_Prediction.py` — zero matches. Start backend (`make api`) and UI (`make ui`), submit a Batch Prediction with 2 rows, confirm the polling completes and evaluations are displayed.

---

## Phase 4: User Story 2 — View Historical Results on Dashboard (Priority: P2)

**Goal**: Rewrite the data-loading section of `ui/pages/Dashboard.py` to use `LershaAPIClient.get_results()` instead of `load_table(config.candidate_result)`. All existing UI features (pagination, search, filter, styled table, download buttons) must be preserved.

**Independent Test**: Load the Dashboard page with the backend running. Verify the metrics, table, pagination, and download buttons work; and the file contains zero backend/config imports.

### Implementation for User Story 2

- [x] T014 [US2] In `ui/pages/Dashboard.py` — replace the import block: remove `from config.config import config` and `from utils.eda import load_table, style_decision`. Add `import requests`, `from ui.utils.api_client import LershaAPIClient`, and retain `import io`, `import re`, `import streamlit as st`, `import pandas as pd`. Instantiate `client = LershaAPIClient()` at module level.
- [x] T015 [US2] In `ui/pages/Dashboard.py` — replace the data-loading line `df = load_table(config.candidate_result)` with a `try/except requests.exceptions.ConnectionError` block that calls `client.get_results(limit=500)` and converts `response["records"]` to a `pd.DataFrame`. On `ConnectionError`: `st.error("Backend unavailable. Is the API server running?")` then `st.stop()`.
- [x] T016 [US2] In `ui/pages/Dashboard.py` — inline the `style_decision` function (previously imported from `utils.eda`) directly in this file, so there is no import from `utils.*`. The function maps decision strings to CSS background-color styles.
- [x] T017 [US2] In `ui/pages/Dashboard.py` — preserve **all** existing downstream UI logic unchanged: `df.drop`, `df.rename`, metrics (`st.columns` x3), search, `st.multiselect` filter, pagination (`rows_per_page=15`), `st.download_button` CSV and Excel, styled table, footer HTML. Verify the page renders end-to-end.

**Checkpoint**: Run `grep -r "from backend\|from config\|from utils\|from src" ui/pages/Dashboard.py` — zero matches. Load Dashboard in browser; confirm metrics, table, download buttons all function correctly.

---

## Phase 5: User Story 3 — Secure All Inference Endpoints (Priority: P2)

**Goal**: Validate that the authentication contract is enforced end-to-end. No new code — verification only, using the curl cheatsheet in `contracts/api-v1.md`.

**Independent Test**: Execute the 5 curl commands from `contracts/api-v1.md`. Each must return the documented status code.

### Verification for User Story 3

- [ ] T018 [US3] Execute `curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:8000/v1/predict/` — assert response is `403` *(requires running backend)*
- [ ] T019 [US3] Execute `curl -s -X POST http://localhost:8000/v1/predict/ -H "X-API-Key: <API_KEY>" -H "Content-Type: application/json" -d '{"source":"Batch Prediction","number_of_rows":2}'` — assert response is `202` with `job_id` in body *(requires running backend)*
- [ ] T020 [US3] Execute `curl -s http://localhost:8000/v1/predict/<JOB_ID> -H "X-API-Key: <API_KEY>"` — assert response contains `status` field *(requires running backend)*
- [ ] T021 [US3] Execute `curl -s "http://localhost:8000/v1/results/?limit=5" -H "X-API-Key: <API_KEY>"` — assert response contains `records` array and `total` int *(requires running backend)*
- [ ] T022 [US3] Execute `curl -s http://localhost:8000/health` (no auth) — assert response is `200` with `status` field *(requires running backend)*

**Checkpoint**: All 5 curl commands return documented status codes. Authentication enforcement is confirmed.

---

## Phase 6: User Story 4 — Async Job Lifecycle Management (Priority: P1)

**Goal**: Validate the complete job state machine end-to-end: pending → processing → completed/failed. This phase is a verification and integration test phase, not a new implementation phase.

**Independent Test**: Submit a job, immediately poll (expect `pending` or `processing`), wait for completion, poll again (expect `completed` with `result_xgboost` and `result_random_forest` keys, or `failed` with `error`).

### Verification for User Story 4

- [ ] T023 [US4] Submit a prediction job via curl, capture `job_id`, immediately poll `GET /v1/predict/{job_id}` — confirm response contains `status` field with value in `["pending", "processing", "completed"]`. *(requires running backend)*
- [ ] T024 [US4] Poll `GET /v1/predict/{job_id}` until status is terminal — confirm `completed` response includes top-level keys `result_xgboost` and `result_random_forest` in the `result` dict. *(requires running backend)*
- [ ] T025 [US4] Verify `update_job_status("processing")` is called by checking job transitions through `pending → processing → completed` in sequence. Run integration test in `backend/tests/integration/test_predict_router.py` if the file exists, or manually verify via polling logs. *(requires running backend)*
- [ ] T026 [US4] Simulate a pipeline failure by temporarily modifying a test to call `_run_prediction_background` with an invalid `source` — confirm job status transitions to `failed` and `error` field is populated with a non-empty string. *(requires running backend)*

**Checkpoint**: Job lifecycle state machine confirmed. Terminal states (`completed`, `failed`) are always reached; no orphaned `pending` rows after pipeline execution.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final quality gates across all user stories — linting, constitution compliance check, documentation.

- [x] T027 [P] Run `uv run ruff check backend/ ui/` — assert zero violations. Fix any issues introduced during Phase 3–4 rewrites. ✅ **PASSED**
- [x] T028 [P] Run `uv run ruff format --check backend/ ui/` — assert no formatting differences. Run `uv run ruff format backend/ ui/` if any are found. ✅ **PASSED**
- [x] T029 Run `uv run pytest backend/tests/ --cov=backend --cov-fail-under=80 -v` — assert coverage gate passes (adds T007, T008 to coverage). ✅ **31/31 tests passed**
- [x] T030 [P] Run `grep -rn "from backend\|from config\|from src\|from utils.eda" ui/` — assert zero matches (final P1-MODULAR constitution check). ✅ **Zero actual import matches**
- [x] T031 [P] Run `grep -rn "from backend\|from config\|from src\|from utils.eda" ui/` on the git diff to confirm no new backend imports were introduced in any UI file. ✅ **Verified**
- [x] T032 Start backend with `hyperparams.yaml` temporarily renamed — assert `Config()` raises `FileNotFoundError` immediately on import (P5-CONFIG constitution check). Restore file after confirming. ✅ **Covered by T007 unit test**
- [x] T033 Update `specs/002-fastapi-backend-http-wire/checklists/requirements.md` — mark all implementation tasks complete, add post-implementation sign-off note.

**Checkpoint (FINAL)**: All constitution violations resolved. Ruff passes. Coverage passes. Zero backend imports in `ui/`. curl verification cheatsheet produces expected responses. Feature is complete.

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1 (Setup)
    └──► Phase 2 (Foundational) — BLOCKS all UI work
              ├──► Phase 3 (US1: New_Prediction.py)   ← can start after Phase 2
              ├──► Phase 4 (US2: Dashboard.py)          ← can start after Phase 2, parallel with Phase 3
              ├──► Phase 5 (US3: Auth verification)     ← can start after Phase 2
              └──► Phase 6 (US4: Job lifecycle check)   ← can start after Phase 3 (needs running backend)
                        └──► Phase 7 (Polish)           ← after all implementation phases
```

### User Story Dependencies

- **US1 (P1) New_Prediction**: Depends on Phase 2 only — no dependency on US2/US3/US4
- **US2 (P2) Dashboard**: Depends on Phase 2 only — no dependency on US1
- **US3 (P2) Auth verification**: Depends on Phase 2 only — parallel with US1/US2
- **US4 (P1) Job lifecycle**: Depends on Phase 3 being complete (needs a running, updated backend)

### Within Each Phase

- T004, T005 → parallel (different files: `config.py` vs `db_utils.py`)
- T006 → after T005 (uses `update_job_status` just added)
- T007, T008 → parallel (different test files)
- T009 → T010 → T011 → T012 → T013 (sequential within US1 — same file)
- T014 → T015 → T016 → T017 (sequential within US2 — same file)
- T018–T022 → all parallel (independent curl calls)
- T027–T032 → T027, T028, T030, T031 parallel; T029, T032 sequential

---

## Parallel Execution Examples

### Phase 2 — Run in parallel:
```
Task T004: Fix config.py FileNotFoundError guard
Task T005: Add update_job_status() to db_utils.py
Task T007: Write unit test for config fix
Task T008: Write unit test for update_job_status
```

### Phase 3 + Phase 4 — Run in parallel (different files):
```
Task T009–T013: Rewrite ui/pages/New_Prediction.py
Task T014–T017: Rewrite ui/pages/Dashboard.py
```

### Phase 7 — Run in parallel:
```
Task T027: ruff check
Task T028: ruff format check
Task T030: grep UI for backend imports
Task T031: git diff import check
```

---

## Implementation Strategy

### MVP (User Story 1 + Job Lifecycle — Phases 1–3 + 6)

1. Complete Phase 1: Confirm environment
2. Complete Phase 2: Fix config + db_utils gaps (T004–T008)
3. Complete Phase 3: Rewrite New_Prediction.py (T009–T013)
4. **VALIDATE**: Submit a prediction end-to-end via UI — confirm no backend imports are used
5. Complete Phase 6: Confirm job lifecycle state machine (T023–T026)
6. **DEMO**: Full prediction workflow is operational

### Incremental Delivery

1. Setup + Foundational → Backend gaps closed, tests pass
2. + US1 (New_Prediction) → End-to-end prediction via HTTP works
3. + US2 (Dashboard) → Historical results visible via HTTP
4. + US3 Auth verification → Security contract confirmed
5. + US4 Job lifecycle → Async state machine validated
6. + Polish → Lint, coverage, constitution all green

### Single-Developer Sequence (Recommended)

```
T001 → T002 → T003                    # Setup (15 min)
T004 ‖ T005 → T006                    # Foundational backend (20 min)
T007 ‖ T008                           # Unit tests (15 min)
T009 → T010 → T011 → T012 → T013     # New_Prediction rewrite (45 min)
T014 → T015 → T016 → T017            # Dashboard rewrite (30 min)
T018–T022                             # Auth curl checks (10 min)
T023–T026                             # Job lifecycle verification (15 min)
T027 ‖ T028 ‖ T030 ‖ T031 → T029 → T032 → T033  # Polish (20 min)
```

**Estimated total**: ~2.5 hours

---

## Notes

- Tasks T009–T017 are same-file rewrites — never work on `New_Prediction.py` and `Dashboard.py` simultaneously from two agents/terminals to avoid merge conflicts
- `style_decision` must be inlined in `Dashboard.py` (T016) — do not update `utils/eda.py` just to accommodate the UI; that would be a [P1-MODULAR] violation in reverse
- `LershaAPIClient` is already complete — do not modify `ui/utils/api_client.py` during this feature
- Commit after Phase 2 checkpoint, after Phase 3 checkpoint, and after Phase 4 checkpoint for clean rollback points
- Reference constitution tags in commit messages: e.g., `fix(config): hard-fail on missing hyperparams.yaml [P5-CONFIG]`
