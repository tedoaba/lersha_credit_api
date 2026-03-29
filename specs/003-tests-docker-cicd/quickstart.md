# Quick Start: Feature 003 — Test Suite, Docker & CI/CD

**Branch**: `003-tests-docker-cicd`

---

## Prerequisites

- Python 3.12
- `uv` (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker & Docker Compose
- PostgreSQL 16 running locally (or via Docker)

---

## Local Development Setup

### 1. Install dependencies

```bash
make install
# Equivalent: uv sync --extra dev
```

### 2. Set environment variables

Copy the example env file and fill in your values:

```bash
cp .env.example .env
# Edit .env:
#   API_KEY=your-secret-key
#   DB_URI=postgresql://lersha:lersha@localhost:5432/lersha
#   GEMINI_API_KEY=your-gemini-key
```

### 3. Initialise the database

```bash
make setup-db
# Equivalent: uv run python backend/scripts/db_init.py
```

---

## Running Tests Locally

### Prerequisites: test database

Create the test database before running integration tests:

```bash
createdb test_lersha
# Or via psql:
# psql -U postgres -c "CREATE DATABASE test_lersha;"
```

Export the test DB URI:

```bash
export DB_URI=postgresql://lersha:lersha@localhost:5432/test_lersha
export API_KEY=ci-test-key
export GEMINI_API_KEY=your-gemini-key   # mocked in unit tests, needed for integration
```

### Run all tests

```bash
make test
# Equivalent: uv run pytest backend/tests/ -v
```

### Run with coverage report

```bash
make coverage
# Equivalent: uv run pytest backend/tests/ --cov=backend --cov-report=html --cov-report=term-missing
# HTML report: open htmlcov/index.html
```

### Run only unit tests

```bash
uv run pytest backend/tests/unit/ -v
```

### Run only integration tests

```bash
uv run pytest backend/tests/integration/ -v
```

---

## Code Quality

### Lint

```bash
make lint
# Equivalent: uv run ruff check backend/ ui/
```

### Format

```bash
make format
# Equivalent: uv run ruff format backend/ ui/
```

### Check format without applying

```bash
make check-format
# Equivalent: uv run ruff format --check backend/ ui/
```

### Run all CI quality gates locally

```bash
make ci-quality
# Equivalent: make lint && make check-format
```

---

## Docker

### Build images

```bash
make docker-build
# Builds:
#   lersha-backend:latest  (from backend/Dockerfile)
#   lersha-ui:latest       (from ui/Dockerfile)
```

### Start the full stack

```bash
make docker-up
# Equivalent: docker compose up -d
# Services:
#   postgres  → localhost:5432
#   backend   → localhost:8000
#   ui        → localhost:8501
#   mlflow    → localhost:5000
```

### Verify the backend is healthy

```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy", ...}
```

### Verify authentication

```bash
# Without key → 403
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"source": "Batch Prediction", "number_of_rows": 1}'

# With key → 202 + job_id
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"source": "Batch Prediction", "number_of_rows": 1}'
```

### Stop the stack

```bash
make docker-down
```

---

## CI/CD

The GitHub Actions pipeline runs automatically on every push and pull request.

**Pipeline**: `.github/workflows/ci.yml`

| Job | Trigger | Runs |
|-----|---------|------|
| `lint` | push / PR | ruff check, ruff format --check, mypy |
| `test` | push / PR | pytest with postgres:16 service; fails if coverage < 80% |
| `build` | after lint + test pass | docker build backend + ui |

**Required repository secrets** (set in GitHub → Settings → Secrets):
- `GEMINI_API_KEY` — used by integration tests that exercise the live pipeline
- `DB_URI` — set automatically by the postgres:16 service container in CI

---

## Cleanup

```bash
make clean
# Removes: __pycache__/, .pytest_cache/, htmlcov/, .coverage, .ruff_cache/
```
