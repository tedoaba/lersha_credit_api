"""Database utilities for the Lersha Credit Scoring backend.

All PostgreSQL interactions are centralised here. SQLAlchemy 2.x patterns
are used throughout (``with engine.connect() as conn:`` and
``with Session(engine) as session:``).

Bug fixes applied in monorepo refactor (2026-03-29):
  - fetch_raw_data: replaced full-table scan + Python filter with
    parameterised ``WHERE farmer_uid = :uid`` server-side query.
  - fetch_multiple_raw_data: replaced full-table scan + df.sample() with
    ``ORDER BY RANDOM() LIMIT :n`` server-side sampling.
  - save_batch_evaluations: migrated from deprecated Session(bind=engine)
    to SQLAlchemy 2.x Session(engine) context manager.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from backend.config.config import config
from backend.logger.logger import get_logger
from backend.services.db_model import CreditScoringRecordDB, InferenceJobDB
from backend.services.schema import CreditScoringRecord

logger = get_logger(__name__)


# ── Engine factory ──────────────────────────────────────────────────────────


# TODO: Refactor to a module-level singleton via functools.lru_cache (or a
# lazy-init _engine variable) so the connection pool is truly shared across
# all calls within a worker process.  Currently each call creates a new engine
# instance with its own pool, which limits pooling effectiveness under load.
def db_engine():
    """Create and return a pooled SQLAlchemy engine using ``config.db_uri``.

    Pool settings (PostgreSQL only — SQLite uses the default single-thread pool):
      pool_size=10      : up to 10 persistent connections per engine instance
      max_overflow=20   : up to 20 additional connections under peak load
      pool_pre_ping=True: validate connections before use (recovers after DB restart)
      pool_recycle=3600 : recycle connections after 1 hour to avoid stale sockets
    """
    uri = config.db_uri
    # Pool settings only apply to PostgreSQL; SQLite (e.g. in tests) uses defaults
    if uri.startswith("postgresql"):
        return create_engine(
            uri,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
    return create_engine(uri)


# ── Data load ───────────────────────────────────────────────────────────────


def load_data_to_database(csv_path: str, table_name: str) -> None:
    """Load a CSV file into the database, replacing any existing data.

    Args:
        csv_path: Path to the CSV file.
        table_name: Target database table name.
    """
    logger.info("Loading CSV '%s' → table '%s'", csv_path, table_name)
    df = pd.read_csv(csv_path)
    engine = db_engine()
    df.to_sql(name=table_name, con=engine, if_exists="replace", index=False)
    logger.info("CSV loaded successfully (%d rows)", len(df))


# ── Read helpers ────────────────────────────────────────────────────────────


def get_data_from_database(table_name: str) -> pd.DataFrame:
    """Fetch all rows from a table.

    Args:
        table_name: Name of the table to query.

    Returns:
        pd.DataFrame: All rows from the table.
    """
    logger.info("Fetching all data from '%s'", table_name)
    engine = db_engine()
    query = text(f"SELECT * FROM {table_name}")  # noqa: S608 — table name from config, not user input
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    logger.info("Fetched %d rows", len(df))
    return df


def fetch_rows(table_name: str, filters: dict | None = None) -> pd.DataFrame:
    """Fetch rows with optional column-equality filters.

    Args:
        table_name: Name of the table to query.
        filters: Optional ``{column: value}`` dict. All conditions are ANDed.

    Returns:
        pd.DataFrame: Matching rows.
    """
    engine = db_engine()
    if not filters:
        query = text(f"SELECT * FROM {table_name}")  # noqa: S608
        with engine.connect() as conn:
            return pd.read_sql(query, conn)

    where_clause = " AND ".join([f"{col} = :{col}" for col in filters])
    query = text(f"SELECT * FROM {table_name} WHERE {where_clause}")  # noqa: S608
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params=filters)


def fetch_raw_data(table_name: str, filters: str) -> pd.DataFrame:
    """Fetch a single farmer's raw data row by farmer_uid.

    FIXED: Uses a server-side parameterised WHERE clause instead of a
    full-table scan followed by Python-side filtering.

    Args:
        table_name: Database table containing raw farmer records.
        filters: The ``farmer_uid`` value to match.

    Returns:
        pd.DataFrame: Rows where ``farmer_uid = filters``.
    """
    logger.info("Fetching raw data for farmer_uid='%s' from '%s'", filters, table_name)
    engine = db_engine()
    query = text(f"SELECT * FROM {table_name} WHERE farmer_uid = :uid")  # noqa: S608
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"uid": filters})
    logger.info("Found %d row(s) for farmer_uid='%s'", len(df), filters)
    return df


def fetch_multiple_raw_data(
    table_name: str,
    n_rows: int = 3,
    *,
    gender: str | None = None,
    age_min: int | None = None,
    age_max: int | None = None,
) -> pd.DataFrame:
    """Fetch a random sample of raw farmer rows via database-level RANDOM().

    FIXED: Uses ``ORDER BY RANDOM() LIMIT :n`` instead of a full-table
    scan followed by ``df.sample()``. This is significantly more efficient
    on large tables.

    Args:
        table_name: Database table to sample from.
        n_rows: Number of rows to return.
        gender: Optional gender filter (case-insensitive).
        age_min: Optional minimum age (inclusive).
        age_max: Optional maximum age (inclusive).

    Returns:
        pd.DataFrame: ``n_rows`` randomly sampled rows.
    """
    logger.info("Fetching %d random rows from '%s' (gender=%s, age=%s-%s)", n_rows, table_name, gender, age_min, age_max)
    engine = db_engine()

    where_clauses: list[str] = []
    params: dict[str, Any] = {"n": n_rows}

    if gender:
        where_clauses.append("LOWER(gender) = LOWER(:gender)")
        params["gender"] = gender
    if age_min is not None:
        where_clauses.append("age >= :age_min")
        params["age_min"] = age_min
    if age_max is not None:
        where_clauses.append("age <= :age_max")
        params["age_max"] = age_max

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    query = text(f"SELECT * FROM {table_name} {where_sql} ORDER BY RANDOM() LIMIT :n")  # noqa: S608

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)
    logger.info("Fetched %d rows", len(df))
    return df


def search_farmers(query: str, limit: int = 10) -> list[dict]:
    """Search farmers by name or UID for autocomplete.

    Args:
        query: Search term (matched via ILIKE on first_name, middle_name, last_name, farmer_uid).
        limit: Maximum results to return.

    Returns:
        list[dict]: Matching farmer records with uid and name fields.
    """
    engine = db_engine()
    table_name = config.farmer_data_all
    sql = text(
        f"SELECT farmer_uid, first_name, middle_name, last_name "  # noqa: S608
        f"FROM {table_name} "
        f"WHERE first_name ILIKE :q OR middle_name ILIKE :q "
        f"OR last_name ILIKE :q OR farmer_uid ILIKE :q "
        f"ORDER BY first_name LIMIT :lim"
    )
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"q": f"%{query}%", "lim": limit})
    return df.to_dict(orient="records")


def fetch_and_filter_columns(table_name: str, columns_to_select: list, column_order: list) -> pd.DataFrame:
    """Fetch specific columns from a table in a defined order.

    Args:
        table_name: Source database table.
        columns_to_select: Columns to keep from the full row set.
        column_order: Final explicit column ordering.

    Returns:
        pd.DataFrame: Filtered and reordered dataset.
    """
    engine = db_engine()
    query = text(f"SELECT * FROM {table_name}")  # noqa: S608
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    df = df[columns_to_select]
    df = df[column_order]
    return df


def get_row_by_number(table: str, row_number: int, order_by: str = "age") -> pd.DataFrame:
    """Fetch a single row by its 0-based offset after ORDER BY.

    Args:
        table: Source database table.
        row_number: 0-based row index.
        order_by: Column to use for ordering.

    Returns:
        pd.DataFrame: Single-row DataFrame.
    """
    engine = db_engine()
    query = text(f"SELECT * FROM {table} ORDER BY {order_by} LIMIT 1 OFFSET :offset")  # noqa: S608
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"offset": row_number})


# ── Write helpers ───────────────────────────────────────────────────────────


def save_batch_evaluations(input_df: pd.DataFrame, evaluation_results: list) -> bool:
    """Persist a batch of evaluation records to the candidate_result table.

    Args:
        input_df: Original raw farmer DataFrame (provides name/UID fields).
        evaluation_results: List of result dicts from ``run_inferences``.

    Returns:
        bool: ``True`` on successful commit.

    Raises:
        Exception: Re-raises any exception after rolling back the session.
    """
    engine = db_engine()
    with Session(engine) as session:
        try:
            for i, result in enumerate(evaluation_results):
                row = input_df.iloc[i]

                record = CreditScoringRecord(
                    farmer_uid=str(row.get("farmer_uid", "")),
                    first_name=row.get("first_name"),
                    middle_name=row.get("middle_name"),
                    last_name=row.get("last_name"),
                    predicted_class_name=result["predicted_class_name"],
                    top_feature_contributions=result["top_feature_contributions"],
                    rag_explanation=result["rag_explanation"],
                    model_name=result["model_name"],
                    timestamp=datetime.utcnow(),
                )

                db_row = CreditScoringRecordDB(
                    farmer_uid=record.farmer_uid,
                    first_name=record.first_name,
                    middle_name=record.middle_name,
                    last_name=record.last_name,
                    predicted_class_name=record.predicted_class_name,
                    top_feature_contributions=[fc.dict() for fc in record.top_feature_contributions],
                    rag_explanation=record.rag_explanation,
                    model_name=record.model_name,
                    timestamp=record.timestamp,
                )
                session.add(db_row)

            session.commit()
            logger.info("Saved %d evaluation records", len(evaluation_results))
            return True
        except Exception as exc:
            session.rollback()
            logger.error("Failed to save batch evaluations: %s", exc, exc_info=True)
            raise


# ── Inference job CRUD ──────────────────────────────────────────────────────


def create_job(job_id: str) -> None:
    """Insert a new inference job row with status 'pending'.

    Args:
        job_id: UUID string for the new job.
    """
    engine = db_engine()
    with Session(engine) as session:
        job = InferenceJobDB(
            job_id=job_id,
            status="pending",
            result=None,
            error=None,
            created_at=datetime.utcnow(),
        )
        session.add(job)
        session.commit()
    logger.info("Created inference job '%s'", job_id)


def update_job_status(job_id: str, status: str) -> None:
    """Update the status field of an inference job.

    Used to mark a job as ``'processing'`` before the pipeline result is
    available, enabling accurate state visibility during polling.

    Args:
        job_id: UUID string of the job to update.
        status: New status value — one of ``'pending'``, ``'processing'``,
            ``'completed'``, ``'failed'``.
    """
    engine = db_engine()
    with Session(engine) as session:
        job = session.get(InferenceJobDB, job_id)
        if job:
            job.status = status
            session.commit()
    logger.info("Job '%s' status updated to '%s'", job_id, status)


def update_job_result(job_id: str, result: dict) -> None:
    """Mark an inference job as completed and store its result payload.

    Args:
        job_id: UUID string of the job to update.
        result: Dict containing ``result_xgboost`` and ``result_random_forest`` keys.
    """
    engine = db_engine()
    with Session(engine) as session:
        job = session.get(InferenceJobDB, job_id)
        if job:
            job.status = "completed"
            job.result = result
            job.completed_at = datetime.utcnow()
            session.commit()
    logger.info("Job '%s' marked completed", job_id)


def update_job_error(job_id: str, error: str) -> None:
    """Mark an inference job as failed and record the error message.

    Args:
        job_id: UUID string of the job to update.
        error: Human-readable error message or exception string.
    """
    engine = db_engine()
    with Session(engine) as session:
        job = session.get(InferenceJobDB, job_id)
        if job:
            job.status = "failed"
            job.error = error
            job.completed_at = datetime.utcnow()
            session.commit()
    logger.warning("Job '%s' marked failed: %s", job_id, error)


def get_job(job_id: str) -> dict[str, Any] | None:
    """Retrieve a single inference job by its ID.

    Args:
        job_id: UUID string of the job.

    Returns:
        dict with ``job_id``, ``status``, ``result``, ``error`` keys,
        or ``None`` if the job does not exist.
    """
    engine = db_engine()
    with Session(engine) as session:
        job = session.get(InferenceJobDB, job_id)
        if not job:
            return None
        return {
            "job_id": job.job_id,
            "status": job.status,
            "result": job.result,
            "error": job.error,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
        }


def get_all_results(limit: int = 500, model_name: str | None = None) -> list[dict]:
    """Fetch evaluation records from the candidate_result table.

    Args:
        limit: Maximum number of records to return (default 500).
        model_name: Optional filter by model name.

    Returns:
        list[dict]: Serialisable list of evaluation records.
    """
    engine = db_engine()
    farmer_table = config.farmer_data_all
    base = (
        f"SELECT cr.*, "  # noqa: S608
        f"COALESCE(cr.first_name, fd.first_name) AS first_name, "
        f"COALESCE(cr.middle_name, fd.middle_name) AS middle_name, "
        f"COALESCE(cr.last_name, fd.last_name) AS last_name, "
        f"fd.gender "
        f"FROM candidate_result cr "
        f"LEFT JOIN {farmer_table} fd ON cr.farmer_uid = fd.farmer_uid"
    )
    if model_name:
        query = text(f"{base} WHERE cr.model_name = :mn ORDER BY cr.timestamp DESC LIMIT :lim")
        params: dict = {"mn": model_name, "lim": limit}
    else:
        query = text(f"{base} ORDER BY cr.timestamp DESC LIMIT :lim")
        params = {"lim": limit}

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)
    return df.to_dict(orient="records")


def get_results_paginated(
    *,
    page: int = 1,
    per_page: int = 20,
    search: str | None = None,
    decision: str | None = None,
    gender: str | None = None,
    model_name: str | None = None,
) -> dict:
    """Fetch evaluation records with pagination, search, and filters.

    Joins candidate_result with farmer_data_all to get gender.

    Returns:
        dict with ``total``, ``page``, ``per_page``, ``records`` keys.
    """
    engine = db_engine()
    farmer_table = config.farmer_data_all

    where_clauses: list[str] = []
    params: dict[str, Any] = {}

    if search:
        where_clauses.append(
            "(COALESCE(cr.first_name, fd.first_name) ILIKE :search "
            "OR COALESCE(cr.last_name, fd.last_name) ILIKE :search "
            "OR cr.farmer_uid ILIKE :search)"
        )
        params["search"] = f"%{search}%"
    if decision:
        where_clauses.append("cr.predicted_class_name = :decision")
        params["decision"] = decision
    if model_name:
        where_clauses.append("cr.model_name = :model_name")
        params["model_name"] = model_name
    if gender:
        where_clauses.append("LOWER(fd.gender) = LOWER(:gender)")
        params["gender"] = gender

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    # Count query
    count_sql = text(
        f"SELECT COUNT(*) FROM candidate_result cr "  # noqa: S608
        f"LEFT JOIN {farmer_table} fd ON cr.farmer_uid = fd.farmer_uid "
        f"{where_sql}"
    )
    # Data query with pagination
    offset = (page - 1) * per_page
    data_sql = text(
        f"SELECT cr.*, "  # noqa: S608
        f"COALESCE(cr.first_name, fd.first_name) AS first_name, "
        f"COALESCE(cr.middle_name, fd.middle_name) AS middle_name, "
        f"COALESCE(cr.last_name, fd.last_name) AS last_name, "
        f"fd.gender "
        f"FROM candidate_result cr "
        f"LEFT JOIN {farmer_table} fd ON cr.farmer_uid = fd.farmer_uid "
        f"{where_sql} "
        f"ORDER BY cr.timestamp DESC LIMIT :per_page OFFSET :offset"
    )
    params["per_page"] = per_page
    params["offset"] = offset

    with engine.connect() as conn:
        total = conn.execute(count_sql, params).scalar() or 0
        df = pd.read_sql(data_sql, conn, params=params)

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "records": df.to_dict(orient="records"),
    }


def get_analytics_summary() -> dict:
    """Aggregate counts for the dashboard: by decision and by gender.

    Joins candidate_result with farmer_data_all to get gender.
    """
    engine = db_engine()
    farmer_table = config.farmer_data_all

    by_decision_sql = text(
        "SELECT predicted_class_name, COUNT(*) as count "
        "FROM candidate_result GROUP BY predicted_class_name"
    )
    by_gender_decision_sql = text(
        f"SELECT COALESCE(fd.gender, 'Unknown') as gender, "  # noqa: S608
        f"cr.predicted_class_name, COUNT(*) as count "
        f"FROM candidate_result cr "
        f"LEFT JOIN {farmer_table} fd ON cr.farmer_uid = fd.farmer_uid "
        f"GROUP BY fd.gender, cr.predicted_class_name"
    )
    recent_sql = text(
        f"SELECT cr.*, fd.gender FROM candidate_result cr "  # noqa: S608
        f"LEFT JOIN {farmer_table} fd ON cr.farmer_uid = fd.farmer_uid "
        f"ORDER BY cr.timestamp DESC LIMIT 10"
    )

    with engine.connect() as conn:
        # Total
        total_row = conn.execute(text("SELECT COUNT(*) FROM candidate_result")).scalar() or 0

        # By decision
        decision_df = pd.read_sql(by_decision_sql, conn)
        by_decision = dict(zip(decision_df["predicted_class_name"], decision_df["count"], strict=True))

        # By gender x decision
        gd_df = pd.read_sql(by_gender_decision_sql, conn)
        by_gender: dict[str, dict[str, int]] = {}
        for _, row in gd_df.iterrows():
            g = str(row["gender"])
            d = str(row["predicted_class_name"])
            by_gender.setdefault(g, {})[d] = int(row["count"])

        # Recent
        recent_df = pd.read_sql(recent_sql, conn)
        recent = recent_df.to_dict(orient="records")

    return {
        "total": int(total_row),
        "by_decision": by_decision,
        "by_gender": by_gender,
        "recent": recent,
    }


def get_recent_jobs(limit: int = 20) -> list[dict]:
    """Fetch the most recent inference jobs.

    Args:
        limit: Maximum number of jobs to return.

    Returns:
        list[dict]: Serialisable list of job records.
    """
    engine = db_engine()
    query = text(
        "SELECT job_id, status, error, created_at, completed_at "
        "FROM inference_jobs ORDER BY created_at DESC LIMIT :lim"
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"lim": limit})
    return df.to_dict(orient="records")
