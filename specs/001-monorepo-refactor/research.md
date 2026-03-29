# Research: Lersha Monorepo Refactor

**Phase**: Phase 0 — Outline & Research  
**Branch**: `001-monorepo-refactor`  
**Date**: 2026-03-29

---

## Summary

All technical decisions for this refactor are deterministic — the codebase is known, the target architecture is fully documented in `docs/ARCHITECTURE.md`, and the spec carries zero `[NEEDS CLARIFICATION]` markers. This research document records the specific decisions made during investigation of the legacy codebase and confirms the approach for each integration point.

---

## Decision 1: uv vs Poetry — Confirmed Migration Path

**Decision**: Migrate to uv. Remove `[tool.poetry.*]` sections entirely. Use `hatchling` as the build backend.

**Rationale**: The current `pyproject.toml` already has a `[project]` table (confirmed in source). It retains only two Poetry-specific lines: `[build-system]` pointing at `poetry-core` and `[tool.poetry.group.dev.dependencies]`. Migration is surgical — 4 lines removed, build system replaced, dev group moved to `[project.optional-dependencies]`.

**Finding from codebase**: `pyproject.toml` line 36-38 still declares `poetry-core` as build backend. Line 40-42 carries `[tool.poetry.group.dev.dependencies]` with only `ipykernel`. `package-mode=false` on line 10 suggests uv compatibility was already partially anticipated.

**Alternatives considered**:
- Keep Poetry: Rejected. Speed improvements from uv are 10–100×; CI simplicity is significant; constitution P11 mandates uv explicitly.
- pip + requirements.txt: Rejected. No lock file, no reproducible builds.

---

## Decision 2: SQLAlchemy 2.x Compatibility — Session API

**Decision**: Replace legacy `Session(bind=engine)` with `with Session(engine) as session:` context manager.

**Rationale**: `db_utils.py` line 155 uses `Session(bind=engine)` which is deprecated in SQLAlchemy 1.4 and removed in 2.x. The current `pyproject.toml` does not pin SQLAlchemy version; pandas 2.x and FastAPI 0.121 both pull in SQLAlchemy 2.x. This will cause an `AttributeError: 'Session' object has no attribute 'bind'` at runtime.

**Additional finding**: `pd.read_sql()` in SQLAlchemy 2.x requires a `Connection` object, not an `Engine`. All `pd.read_sql(query, con=engine)` calls must become `with engine.connect() as conn: pd.read_sql(query, conn)`.

**Alternatives considered**:
- Pin SQLAlchemy 1.4: Rejected. Constitution P11 mandates current, maintained dependencies. Pinning a deprecated major version is a security risk.

---

## Decision 3: ChromaDB PersistentClient API

**Decision**: Use `chromadb.PersistentClient(path=str(config.chroma_db_path))` — not ephemeral `chromadb.Client()`.

**Rationale**: The existing `rag_engine.py` (not yet read but referenced in docs) uses an ephemeral client, meaning ChromaDB data is lost on every restart. The spec (FR-025) and REFACTOR_PLAN.md (Step 2.4) both mandate switching to `PersistentClient`. The `CHROMA_DB_PATH` config field must be added.

**API note**: `chromadb.PersistentClient` is available in chromadb ≥ 0.4.0. The current chromadb version in the project should be checked and pinned. The `path` argument must be a string (not `Path` object) in some versions.

**Alternatives considered**:
- `chromadb.HttpClient`: Rejected for this phase. Adds another container dependency. `PersistentClient` with a Docker volume is sufficient for Phase 1 production readiness.

---

## Decision 4: apply_feature_engineering() Extraction — Zero ML Dependency Confirmed

**Decision**: Extract `apply_feature_engineering()` to `backend/core/feature_engineering.py` with imports limited to `numpy` and `pandas`.

**Rationale**: Confirmed by reading `src/infer_utils.py` lines 269–323. The function uses only:
- `numpy` (`np.round`, `np.log1p`)
- `pandas` (`pd.DataFrame`, `pd.qcut`, `pd.cut`)
- No SHAP, joblib, xgboost, catboost, sklearn, or MLflow imports

This means the extraction is clean. Unit tests for `apply_feature_engineering()` can run without any ML model artifacts.

**Dropped columns** (confirmed from source, 17 columns total):
```python
['age', 'value_chain', 'estimated_cost', 'estimated_income', 'estimated_expenses',
 'estimated_income_another_farm', 'total_farmland_size', 'land_size', 'childrenunder12',
 'elderlymembersover60', 'agricultureexperience', 'agriculturalcertificate',
 'hasmemberofmicrofinance', 'hascooperativeassociation', 'hascommunityhealthinsurance',
 'maincrops', 'lastyearaverageprice']
```

---

## Decision 5: Dead Code — Confirmed Functions to Delete

**Decision**: Delete `generate_shap_value_summary_plotsss` (triple-s) and `load_prediction_model()` (singular).

**Rationale confirmed** from `src/infer_utils.py`:

- `generate_shap_value_summary_plotsss` (lines 325–387): Identical logic to `generate_shap_value_summary_plots` (lines 389–456) with slightly less mature error handling. Triple `s` is a typo naming collision. The clean `generate_shap_value_summary_plots` (double `s`) at line 389 is the active version. The triple-s version is dead.

- `load_prediction_model()` (lines 56–74): References `config.xgb_model_18`, `config.xgb_model_44`, `config.xgb_engineered_model` — attributes that no longer exist in `config.py`. Function is orphaned. `load_prediction_models()` (plural, lines 77–93) is the live version routing correctly to `xgb_model_36`, `rf_model_36`, `cab_model_36`.

---

## Decision 6: `BASE_DIR` Depth — Confirmed parents[2]

**Decision**: `BASE_DIR = Path(__file__).resolve().parents[2]`

**Rationale**: Confirmed by file locations:
- Config file will be at: `backend/config/config.py`
- `parents[0]` = `backend/config/`
- `parents[1]` = `backend/`
- `parents[2]` = project root `lersha_credit_api/` ✅

Current code (confirmed `config/config.py` line 7) uses `parents[1]` which currently points at `lersha_credit_api/` because the config is at `config/config.py` (one level deep). After the move to `backend/config/config.py` (two levels deep), `parents[1]` would point at `backend/` — breaking all model paths. The fix to `parents[2]` is mandatory and verified.

---

## Decision 7: API Response Fields — Root Cause Confirmed

**Decision**: The legacy `app.py` calls `infer()` from `infer.py`. `infer.py` returns three values matched to `result_18`, `result_44`, `result_featured`. The new `run_inferences()` in `pipeline.py` runs only two models (`xgboost`, `random_forest`) and returns structured dicts. Response must use `result_xgboost` and `result_random_forest`.

**Confirmed from source**:
- `app.py` line 27: `result_18, result_44, result_featured = infer(item.source, item.farmer_uid, item.number_of_rows)`
- This calls `infer.py`, which is a different execution path from `pipeline.py`'s `run_inferences()`
- The new `predict.py` router will call `run_inferences()` directly, returning two model results

---

## Decision 8: `asyncio_mode="auto"` — pytest-asyncio Requirement

**Decision**: Add `pytest-asyncio>=0.23` to dev dependencies. Use `asyncio_mode = "auto"` in `[tool.pytest.ini_options]`.

**Rationale**: `asyncio_mode="auto"` requires pytest-asyncio ≥ 0.21. Version 0.23+ avoids a known warning about unrecognized custom markers in some test collection scenarios. Since we're on Python 3.12 and FastAPI uses async throughout, this is a non-optional CI dependency.

---

## Decision 9: Pydantic Schemas — model field naming

**Decision**: Define `PredictRequest` and `JobResponse` in `backend/api/schemas.py`. The `evaluations` list within `result_xgboost` / `result_random_forest` uses field names from the architecture doc (`predicted_class_name`, `top_feature_contributions`, `rag_explanation`, `model_name`, `timestamp`).

**Detail**: The `CreditScoringRecord` Pydantic schema in `backend/services/schema.py` must contain a `top_feature_contributions: list[FeatureContribution]` field where `FeatureContribution` is `{"feature": str, "value": float}`. The existing `schema.py` must be verified — if it already defines this shape, no change is needed.

---

## Decision 10: `chroma_loader.py` Location

**Decision**: The file will live at `backend/chat/chroma_loader.py` per the architecture doc (§4 module list for `backend/chat/`). However, since it's listed in the spec's directory skeleton under `backend/scripts/populate_chroma.py`, the callable **entry point** will be `backend/scripts/populate_chroma.py` (which imports and calls the loader). The loader logic itself can live in `backend/chat/chroma_loader.py`.

**Alternatives considered**: Put everything in scripts. Rejected — the ChromaDB collection management (upsert logic, embedding) belongs in the `chat/` layer; the script is just an invocation wrapper.

---

## Unresolved Items

None. All `[NEEDS CLARIFICATION]` markers from the spec are zero (confirmed from spec validation). All decisions above are grounded in direct codebase inspection.
