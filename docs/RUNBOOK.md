# Lersha Credit Scoring — Operational Runbook

## Quick Reference

| Service | Port | Health Check |
|---------|------|-------------|
| FastAPI | 8006 | `GET /health` |
| Next.js | 3007 | `GET /` |
| PostgreSQL | 5432 | `pg_isready` |
| Redis | 6379 | `redis-cli ping` |
| Celery Worker | — | `celery -A backend.worker inspect ping` |

---

## Scaling Workers

**Check queue depth:**
```bash
# Redis queue length
docker exec redis redis-cli LLEN celery

# Celery active/reserved tasks
docker exec celery celery -A backend.worker inspect active
docker exec celery celery -A backend.worker inspect reserved
```

**Scale up:**
```bash
# Add more workers (Docker Compose)
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale celery=4

# Or increase concurrency per worker
docker exec celery celery -A backend.worker worker --concurrency=8
```

---

## Debugging a Failed Job

```bash
# 1. Find the job
docker exec postgres psql -U lersha -d lersha -c \
  "SELECT job_id, status, error, created_at FROM inference_jobs WHERE job_id = '<JOB_ID>';"

# 2. Check worker logs (filter by job_id)
docker logs celery 2>&1 | grep '<JOB_ID>'

# 3. Check if results were partially saved
docker exec postgres psql -U lersha -d lersha -c \
  "SELECT COUNT(*) FROM candidate_result WHERE job_id = '<JOB_ID>';"

# 4. Retry a failed job (manual)
docker exec api python -c "
from backend.worker import run_inference_task
from backend.services.db_utils import get_job
job = get_job('<JOB_ID>')
# Re-dispatch
run_inference_task.delay('<JOB_ID>', job['result'] or {})
"
```

---

## Redis Cache Management

```bash
# Check RAG cache size
docker exec redis redis-cli DBSIZE

# List RAG cache keys
docker exec redis redis-cli KEYS "rag:explain:*" | head -20

# Flush only RAG cache (preserving Celery state)
docker exec redis redis-cli --scan --pattern "rag:explain:*" | xargs docker exec -i redis redis-cli DEL

# Flush entire Redis (WARNING: clears Celery broker too)
docker exec redis redis-cli FLUSHDB
```

---

## Database Operations

### Backup

Automated backups run every 6 hours (docker-compose.prod.yml backup service).

```bash
# Manual backup
docker exec postgres pg_dump -U lersha lersha | gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz
```

### Restore

```bash
# Stop the API and worker first
docker compose stop api celery

# Restore from backup
gunzip -c backup_20260406.sql.gz | docker exec -i postgres psql -U lersha lersha

# Re-run migrations (in case backup is older than latest migration)
docker exec api alembic -c backend/alembic.ini upgrade head

# Restart services
docker compose start api celery
```

### Apply Migrations

```bash
# Check current migration version
docker exec api alembic -c backend/alembic.ini current

# Apply pending migrations
docker exec api alembic -c backend/alembic.ini upgrade head

# Rollback one migration
docker exec api alembic -c backend/alembic.ini downgrade -1
```

---

## Gemini / RAG Issues

### Gemini API Down

**Symptoms:** Jobs complete but `rag_explanation` shows `[RAG unavailable]`.

```bash
# Check Gemini error rate in logs
docker logs api 2>&1 | grep "RAG explanation failed" | tail -20

# Check audit log for cache hit rate
docker exec postgres psql -U lersha -d lersha -c \
  "SELECT cache_hit, COUNT(*) FROM rag_audit_log WHERE created_at > NOW() - INTERVAL '1 hour' GROUP BY cache_hit;"
```

**Mitigation:**
- Predictions still work (SHAP values are computed locally)
- Explanations degrade to `[RAG unavailable]` — not a hard failure
- Monitor Gemini status at https://status.cloud.google.com/

### pgvector Retrieval Returns No Results

```bash
# Check document count
docker exec postgres psql -U lersha -d lersha -c "SELECT COUNT(*) FROM rag_documents;"

# Check similarity threshold (may be too high)
# Default: 0.75 — lower if needed in hyperparams.yaml
```

---

## Connection Pool Exhaustion

**Symptoms:** `FATAL: too many connections for role "lersha"` in PostgreSQL logs.

```bash
# Check active connections
docker exec postgres psql -U lersha -d lersha -c \
  "SELECT count(*), state FROM pg_stat_activity WHERE datname = 'lersha' GROUP BY state;"

# Check max connections
docker exec postgres psql -U lersha -d lersha -c "SHOW max_connections;"
```

**Fix:** The engine is now cached via `@lru_cache` (one pool per process). If still hitting limits:
- Reduce `pool_size` in `db_utils.py` (default: 10)
- Increase PostgreSQL `max_connections` in `postgresql.conf`
- Reduce Gunicorn/Celery worker count

---

## Log Analysis

```bash
# All errors in the last hour
docker logs api --since 1h 2>&1 | python -c "
import sys, json
for line in sys.stdin:
    try:
        obj = json.loads(line)
        if obj.get('levelname') == 'ERROR':
            print(obj['asctime'], obj['message'])
    except: pass
"

# Inference latency (from worker logs)
docker logs celery 2>&1 | grep "completed successfully" | tail -10

# Request trace by ID
docker logs api 2>&1 | grep '<REQUEST_ID>'
docker logs celery 2>&1 | grep '<REQUEST_ID>'
```

---

## Health Check Failures

If `/health` returns 503:

```bash
# Check which component is down
curl -s http://localhost:8006/health | python -m json.tool

# Restart individual services
docker compose restart postgres   # If DB is down
docker compose restart redis      # If Redis is down
docker compose restart api        # If API is unresponsive
```
