"""pgvector Knowledge Base Population Script.

Migrates all feature-definition documents from the inline FEATURE_DEFINITIONS
catalogue into the rag_documents PostgreSQL table using batch upserts via
pgvector-compatible embeddings.

This script replaces backend/scripts/populate_chroma.py (archived to
backup/scripts/populate_chroma.py) as part of 006-migrate-chroma-pgvector.

Usage:
    uv run python -m backend.scripts.populate_pgvector

Prerequisites:
    - PostgreSQL running with pgvector extension enabled (migration 003 applied)
    - DB_URI environment variable pointing to a live database
    - Sentence-transformers model cached locally (all-MiniLM-L6-v2)

Expected output:
    [INFO] Starting pgvector knowledge base population...
    [INFO] Loading sentence-transformer model 'all-MiniLM-L6-v2'...
    [INFO] Model loaded. Starting embedding and ingestion for 34 documents...
    [OK]   Batch 1/1 — ingested 34 documents.
    [OK]   Ingested 34 documents in 1 batch(es). Failures: 0.
"""

from __future__ import annotations

import sys
import uuid
from datetime import UTC, datetime

from sentence_transformers import SentenceTransformer
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.config.config import config
from backend.services.db_model import RagDocumentDB
from backend.services.db_utils import db_engine

# ── Document catalogue ─────────────────────────────────────────────────────────
# Source: migrated from populate_chroma.py FEATURE_DEFINITIONS list.
# Each entry: (feature_name, description, category)
FEATURE_DEFINITIONS: list[tuple[str, str, str]] = [
    ("gender", "Farmer's gender: Male or Female — affects household income patterns.", "feature_definition"),
    (
        "age_group",
        "Age group derived from age: Young (0-20), Early_Middle (21-35), Late_Middle (36-45), Senior (46+).",
        "feature_definition",
    ),
    (
        "family_size",
        "Total number of family members — affects income-per-member and loan capacity.",
        "feature_definition",
    ),
    ("typeofhouse", "Type of dwelling: permanent, semi-permanent, or temporary.", "feature_definition"),
    ("asset_ownership", "Whether the farmer owns productive assets (1=yes, 0=no).", "feature_definition"),
    (
        "water_reserve_access",
        "Access to reliable water reserve for irrigation (1=yes, 0=no).",
        "feature_definition",
    ),
    (
        "output_storage_type",
        "Storage facility type for harvest output: warehouse, silo, none.",
        "feature_definition",
    ),
    (
        "decision_making_role",
        "Farmer's role in household financial decisions: primary, secondary, joint.",
        "feature_definition",
    ),
    (
        "hasrusacco",
        "Member of a Rural Savings and Credit Cooperative (RUSACCO) (1=yes, 0=no).",
        "feature_definition",
    ),
    ("haslocaledir", "Has a local cooperative membership (1=yes, 0=no).", "feature_definition"),
    (
        "primaryoccupation",
        "Main livelihood activity: farming, animal husbandry, mixed, other.",
        "feature_definition",
    ),
    (
        "holdsleadershiprole",
        "Holds a leadership role in a community organisation (1=yes, 0=no).",
        "feature_definition",
    ),
    ("land_title", "Holds a formal land title document (1=yes, 0=no).", "feature_definition"),
    ("rented_farm_land", "Size of rented farm land in hectares.", "feature_definition"),
    ("own_farmland_size", "Size of farmer's own farmland in hectares.", "feature_definition"),
    ("family_farmland_size", "Total family-controlled farmland size in hectares.", "feature_definition"),
    ("flaw", "Presence of observed land/crop defects or quality issues (1=yes, 0=no).", "feature_definition"),
    (
        "farm_mechanization",
        "Level of mechanization: manual, semi-mechanized, fully mechanized.",
        "feature_definition",
    ),
    (
        "agriculture_experience",
        "Log-transformed years of agricultural experience (log1p of raw value).",
        "feature_definition",
    ),
    (
        "institutional_support_score",
        "Sum of 4 binary institutional flags: microfinance, cooperative, agri-cert, health-insurance.",
        "feature_definition",
    ),
    ("farmsizehectares", "Total operated farm area in hectares.", "feature_definition"),
    ("seedtype", "Seed type used: improved, traditional, hybrid.", "feature_definition"),
    ("seedquintals", "Quantity of seed used in quintals (100 kg per quintal).", "feature_definition"),
    ("expectedyieldquintals", "Expected harvest yield in quintals.", "feature_definition"),
    ("saleableyieldquintals", "Quantity of harvest intended for market sale in quintals.", "feature_definition"),
    ("ureafertilizerquintals", "Urea fertilizer used in quintals.", "feature_definition"),
    ("dapnpsfertilizerquintals", "DAP/NPS fertilizer used in quintals.", "feature_definition"),
    (
        "input_intensity",
        "Input-to-land ratio: (seeds + urea + DAP) / farmsize — proxy for farming intensity.",
        "feature_definition",
    ),
    (
        "yield_per_hectare",
        "Expected yield per operated hectare — key productivity indicator.",
        "feature_definition",
    ),
    (
        "income_per_family_member",
        "Total estimated income divided by family size — welfare indicator.",
        "feature_definition",
    ),
    (
        "total_estimated_income",
        "Sum of primary farm income and income from other farm activities.",
        "feature_definition",
    ),
    (
        "total_estimated_cost",
        "Sum of production expenses and estimated total costs.",
        "feature_definition",
    ),
    (
        "net_income",
        "Total income minus total cost — primary creditworthiness indicator.",
        "feature_definition",
    ),
    ("decision", "Target label: Eligible, Review, or Not Eligible.", "feature_definition"),
]

# ── Ingestion configuration ────────────────────────────────────────────────────
BATCH_SIZE = 1_000  # Rows per INSERT batch — safe for all typical corpus sizes


def populate_pgvector(batch_size: int = BATCH_SIZE) -> int:
    """Embed and upsert all FEATURE_DEFINITIONS into rag_documents.

    Uses ``INSERT ... ON CONFLICT (doc_id) DO UPDATE`` so the script is
    fully idempotent — re-running it does not create duplicate rows.

    Args:
        batch_size: Number of documents to insert per database round-trip.

    Returns:
        int: Total number of documents successfully ingested.

    Raises:
        SystemExit: Exits with code 1 if any batch fails to insert.
    """
    print("[INFO] Starting pgvector knowledge base population...")  # noqa: T201
    print(f"[INFO] Loading sentence-transformer model '{config.embedder_model}'...")  # noqa: T201

    # Load model ONCE outside the loop to avoid repeated HuggingFace cache hits
    embedder = SentenceTransformer(config.embedder_model)
    total_docs = len(FEATURE_DEFINITIONS)

    print(f"[INFO] Model loaded. Starting embedding and ingestion for {total_docs} documents...")  # noqa: T201

    # Pre-compute all embeddings in a single batch call (most efficient)
    contents = [defn for _, defn, _ in FEATURE_DEFINITIONS]
    all_embeddings = embedder.encode(contents, show_progress_bar=False).tolist()

    engine = db_engine()
    now = datetime.now(UTC)
    successes = 0
    failures = 0
    n_batches = 0

    # Chunk documents into batches
    for batch_start in range(0, total_docs, batch_size):
        batch_slice = FEATURE_DEFINITIONS[batch_start : batch_start + batch_size]
        batch_embeddings = all_embeddings[batch_start : batch_start + batch_size]
        n_batches += 1
        batch_label = f"Batch {n_batches}"

        rows = []
        for (feature_name, content, category), embedding in zip(batch_slice, batch_embeddings, strict=False):
            rows.append(
                {
                    "doc_id": uuid.uuid5(uuid.NAMESPACE_DNS, f"lersha.rag.{feature_name}"),
                    "category": category,
                    "title": feature_name.replace("_", " ").title(),
                    "content": content,
                    "embedding": embedding,
                    "metadata": {"feature_name": feature_name},
                    "created_at": now,
                    "updated_at": now,
                }
            )

        try:
            with Session(engine) as session:
                stmt = pg_insert(RagDocumentDB.__table__).values(rows)
                upsert_stmt = stmt.on_conflict_do_update(
                    index_elements=["doc_id"],
                    set_={
                        "content": stmt.excluded.content,
                        "embedding": stmt.excluded.embedding,
                        "metadata": stmt.excluded["metadata"],
                        "updated_at": stmt.excluded.updated_at,
                    },
                )
                session.execute(upsert_stmt)
                session.commit()
                successes += len(rows)
                print(f"[OK]   {batch_label} — ingested {len(rows)} documents.")  # noqa: T201
        except SQLAlchemyError as exc:
            failures += 1
            print(f"[ERROR] {batch_label} failed: {exc}", file=sys.stderr)  # noqa: T201

    total_batches = n_batches - failures
    print(  # noqa: T201
        f"[OK]   Ingested {successes} documents in {total_batches} batch(es). Failures: {failures}."
    )

    if failures > 0:
        sys.exit(1)

    return successes


if __name__ == "__main__":
    populate_pgvector()
