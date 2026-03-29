# Quickstart: FastAPI Backend & Streamlit UI (Local Dev)

**Branch**: `002-fastapi-backend-http-wire`

---

## Prerequisites

- Python 3.12 (via `.venv`)
- PostgreSQL running (local or Docker)
- `.env` file created from `.env.example`
- `uv` installed

---

## 1. Environment Setup

```bash
# Copy and edit the env file
cp .env.example .env
# Edit .env — set DB_URI, API_KEY, GEMINI_API_KEY, GEMINI_MODEL at minimum

# Install all dependencies
uv sync
```

---

## 2. Database Initialisation

```bash
# Create all tables (inference_jobs, candidate_result, etc.)
uv run python backend/scripts/db_init.py
```

---

## 3. Start the FastAPI Backend

```bash
# Dev mode with hot reload
uv run uvicorn backend.main:app --reload --port 8000

# Or via Makefile
make api
```

Verify: `curl http://localhost:8000/health`

---

## 4. Start the Streamlit UI

In a separate terminal:

```bash
uv run streamlit run ui/Introduction.py --server.port 8501

# Or via Makefile
make ui
```

Open: `http://localhost:8501`

---

## 5. Verify End-to-End

```bash
# Step 1: Confirm health
curl http://localhost:8000/health

# Step 2: Submit a prediction (replace key with your API_KEY value)
curl -s -X POST http://localhost:8000/v1/predict/ \
  -H "X-API-Key: your-secret-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"source":"Batch Prediction","number_of_rows":2}'
# → {"job_id":"<uuid>","status":"accepted"}

# Step 3: Poll job status
curl -s http://localhost:8000/v1/predict/<JOB_ID> \
  -H "X-API-Key: your-secret-api-key-here"
# → {"status":"completed","result":{...}}

# Step 4: Fetch results
curl -s "http://localhost:8000/v1/results/?limit=5" \
  -H "X-API-Key: your-secret-api-key-here"
```

---

## 6. Docker Compose (Full Stack)

```bash
# Start all services
docker-compose up --build

# Stop
docker-compose down
```

Services:
- `backend`: FastAPI on port 8000
- `ui`: Streamlit on port 8501
- `db`: PostgreSQL on port 5432
