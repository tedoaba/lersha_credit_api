"""pgvector Knowledge Base Population Script.

Ingests two document sources into the rag_documents PostgreSQL table:

1. **Feature definitions** (inline catalogue) — 34 terse ML feature descriptions
   (category: ``feature_definition``).
2. **Domain knowledge** (markdown files) — rich context from
   ``backend/data/rag-data/`` covering data point methodology, crop reference
   data, survey mappings, and eligibility decision framework
   (category: ``domain_knowledge``).

Markdown files are split into section-based chunks (one chunk per ``## ``
heading) to stay within the sentence-transformer embedding sweet spot
(~256 tokens).  Each chunk is assigned a stable UUID5 doc_id for idempotent
upserts.

This script replaces backend/scripts/populate_chroma.py (archived to
backup/scripts/populate_chroma.py) as part of 006-migrate-chroma-pgvector.

Usage:
    uv run python -m backend.scripts.populate_pgvector

Prerequisites:
    - PostgreSQL running with pgvector extension enabled (migration 003 applied)
    - DB_URI environment variable pointing to a live database
    - Sentence-transformers model cached locally (all-MiniLM-L6-v2)
"""

from __future__ import annotations

import re
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.config.config import BASE_DIR, config
from backend.services.db_model import RagDocumentDB
from backend.services.db_utils import db_engine

RAG_DATA_DIR = BASE_DIR / "backend" / "data" / "rag-data"

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

# ── Markdown chunking ─────────────────────────────────────────────────────────

_H2_SPLIT = re.compile(r"(?=^## )", re.MULTILINE)


def chunk_markdown(filepath: Path, base_dir: Path) -> list[tuple[str, str, str, str]]:
    """Split a markdown file into section-based chunks.

    Each chunk gets the H1 title prepended for context.  Empty sections
    (heading only, no body text) are skipped.

    Returns:
        List of ``(doc_id_seed, title, content, category)`` tuples.
    """
    text = filepath.read_text(encoding="utf-8")
    relative = filepath.relative_to(base_dir).as_posix()

    # Extract H1 title (first line starting with "# ")
    h1_match = re.match(r"^#\s+(.+)", text)
    h1_title = h1_match.group(1).strip() if h1_match else filepath.stem.replace("-", " ").title()

    sections = _H2_SPLIT.split(text)
    chunks: list[tuple[str, str, str, str]] = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract section heading
        heading_match = re.match(r"^##\s+(.+)", section)
        if heading_match:
            heading = heading_match.group(1).strip()
            body = section[heading_match.end() :].strip()
        else:
            # Preamble before any H2 (contains H1 + intro text)
            heading = "Introduction"
            body = section

        if not body:
            continue

        # Prepend H1 title for context
        chunk_content = f"{h1_title} — {heading}\n\n{body}"
        doc_id_seed = f"lersha.rag.md.{relative}#{heading}"
        title = f"{h1_title} > {heading}"

        chunks.append((doc_id_seed, title, chunk_content, "domain_knowledge"))

    return chunks


def load_markdown_documents(rag_dir: Path) -> list[tuple[str, str, str, str]]:
    """Walk ``rag_dir`` and chunk all ``.md`` files.

    Returns:
        List of ``(doc_id_seed, title, content, category)`` tuples.
    """
    all_chunks: list[tuple[str, str, str, str]] = []
    md_files = sorted(rag_dir.rglob("*.md"))

    for md_file in md_files:
        all_chunks.extend(chunk_markdown(md_file, rag_dir))

    return all_chunks


# ── Ingestion configuration ────────────────────────────────────────────────────
BATCH_SIZE = 1_000  # Rows per INSERT batch — safe for all typical corpus sizes


def _upsert_documents(
    engine: object,
    documents: list[dict],
    batch_size: int,
    label: str,
) -> tuple[int, int]:
    """Batch-upsert ``documents`` into rag_documents.

    Returns:
        ``(successes, failures)`` counts.
    """
    successes = 0
    failures = 0

    for n_batch, batch_start in enumerate(range(0, len(documents), batch_size), 1):
        batch = documents[batch_start : batch_start + batch_size]
        batch_label = f"{label} batch {n_batch}"

        try:
            with Session(engine) as session:
                stmt = pg_insert(RagDocumentDB.__table__).values(batch)
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
                successes += len(batch)
                print(f"[OK]   {batch_label} — ingested {len(batch)} documents.")  # noqa: T201
        except SQLAlchemyError as exc:
            failures += 1
            print(f"[ERROR] {batch_label} failed: {exc}", file=sys.stderr)  # noqa: T201

    return successes, failures


def populate_pgvector(batch_size: int = BATCH_SIZE) -> int:
    """Embed and upsert feature definitions and domain knowledge into rag_documents.

    Ingests two sources:
    1. Inline ``FEATURE_DEFINITIONS`` (category: ``feature_definition``)
    2. Markdown files from ``backend/data/rag-data/`` (category: ``domain_knowledge``)

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
    print(f"[INFO] Using Ollama embedder '{config.embedder_model}' at {config.ollama_host}...")  # noqa: T201

    ollama = httpx.Client(base_url=config.ollama_host, timeout=120.0)

    # max_chars = 2000  # Safe limit for mxbai-embed-large (512-token context)

    def _embed_batch(texts: list[str]) -> list[list[float]]:
        """Embed texts via Ollama /api/embed, one at a time.

        Truncates each text to ``max_chars`` characters to stay within
        the embedding model's context window.
        """
        all_embs: list[list[float]] = []
        for text_item in texts:
            resp = ollama.post(
                "/api/embed",
                json={
                    "model": config.embedder_model,
                    "input": [text_item],
                    "options": {"num_ctx": 8192},
                },
            )
            resp.raise_for_status()
            all_embs.append(resp.json()["embeddings"][0])
        return all_embs

    # ── Source 1: Inline feature definitions ──────────────────────────────
    feature_contents = [defn for _, defn, _ in FEATURE_DEFINITIONS]

    # ── Source 2: Markdown domain knowledge ───────────────────────────────
    md_chunks = load_markdown_documents(RAG_DATA_DIR)
    md_contents = [content for _, _, content, _ in md_chunks]

    total_docs = len(FEATURE_DEFINITIONS) + len(md_chunks)
    print(  # noqa: T201
        f"[INFO] Embedding {len(FEATURE_DEFINITIONS)} feature definitions "
        f"+ {len(md_chunks)} domain knowledge chunks ({total_docs} total)..."
    )

    # Pre-compute all embeddings via Ollama (batch call)
    all_contents = feature_contents + md_contents
    all_embeddings = _embed_batch(all_contents)

    feature_embeddings = all_embeddings[: len(FEATURE_DEFINITIONS)]
    md_embeddings = all_embeddings[len(FEATURE_DEFINITIONS) :]

    engine = db_engine()
    now = datetime.now(UTC)

    # Build feature definition rows
    feature_rows = []
    for (feature_name, content, category), embedding in zip(FEATURE_DEFINITIONS, feature_embeddings, strict=False):
        feature_rows.append(
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

    # Build domain knowledge rows
    md_rows = []
    for (doc_id_seed, title, content, category), embedding in zip(md_chunks, md_embeddings, strict=False):
        md_rows.append(
            {
                "doc_id": uuid.uuid5(uuid.NAMESPACE_DNS, doc_id_seed),
                "category": category,
                "title": title,
                "content": content,
                "embedding": embedding,
                "metadata": {"source": "rag-data-markdown"},
                "created_at": now,
                "updated_at": now,
            }
        )

    # Upsert both sources
    feat_ok, feat_fail = _upsert_documents(engine, feature_rows, batch_size, "Feature definitions")
    md_ok, md_fail = _upsert_documents(engine, md_rows, batch_size, "Domain knowledge")

    total_ok = feat_ok + md_ok
    total_fail = feat_fail + md_fail
    print(  # noqa: T201
        f"[OK]   Ingested {total_ok} documents "
        f"({feat_ok} feature_definition + {md_ok} domain_knowledge). "
        f"Failures: {total_fail}."
    )

    if total_fail > 0:
        sys.exit(1)

    return total_ok


if __name__ == "__main__":
    populate_pgvector()
