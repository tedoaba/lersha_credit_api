"""ChromaDB population script.

Run once to populate the credit_features collection with feature definitions
that will be used by the RAG explanation pipeline.

Usage:
    uv run python backend/scripts/populate_chroma.py
"""
import chromadb
from chromadb.utils import embedding_functions

from backend.config.config import config
from backend.logger.logger import get_logger

logger = get_logger(__name__)

FEATURE_DEFINITIONS = [
    ("gender", "Farmer's gender: Male or Female — affects household income patterns."),
    ("age_group", "Age group derived from age: Young (0-20), Early_Middle (21-35), Late_Middle (36-45), Senior (46+)."),
    ("family_size", "Total number of family members — affects income-per-member and loan capacity."),
    ("typeofhouse", "Type of dwelling: permanent, semi-permanent, or temporary."),
    ("asset_ownership", "Whether the farmer owns productive assets (1=yes, 0=no)."),
    ("water_reserve_access", "Access to reliable water reserve for irrigation (1=yes, 0=no)."),
    ("output_storage_type", "Storage facility type for harvest output: warehouse, silo, none."),
    ("decision_making_role", "Farmer's role in household financial decisions: primary, secondary, joint."),
    ("hasrusacco", "Member of a Rural Savings and Credit Cooperative (RUSACCO) (1=yes, 0=no)."),
    ("haslocaledir", "Has a local cooperative membership (1=yes, 0=no)."),
    ("primaryoccupation", "Main livelihood activity: farming, animal husbandry, mixed, other."),
    ("holdsleadershiprole", "Holds a leadership role in a community organisation (1=yes, 0=no)."),
    ("land_title", "Holds a formal land title document (1=yes, 0=no)."),
    ("rented_farm_land", "Size of rented farm land in hectares."),
    ("own_farmland_size", "Size of farmer's own farmland in hectares."),
    ("family_farmland_size", "Total family-controlled farmland size in hectares."),
    ("flaw", "Presence of observed land/crop defects or quality issues (1=yes, 0=no)."),
    ("farm_mechanization", "Level of mechanization: manual, semi-mechanized, fully mechanized."),
    ("agriculture_experience", "Log-transformed years of agricultural experience (log1p of raw value)."),
    ("institutional_support_score", "Sum of 4 binary institutional flags: microfinance, cooperative, agri-cert, health-insurance."),
    ("farmsizehectares", "Total operated farm area in hectares."),
    ("seedtype", "Seed type used: improved, traditional, hybrid."),
    ("seedquintals", "Quantity of seed used in quintals (100 kg per quintal)."),
    ("expectedyieldquintals", "Expected harvest yield in quintals."),
    ("saleableyieldquintals", "Quantity of harvest intended for market sale in quintals."),
    ("ureafertilizerquintals", "Urea fertilizer used in quintals."),
    ("dapnpsfertilizerquintals", "DAP/NPS fertilizer used in quintals."),
    ("input_intensity", "Input-to-land ratio: (seeds + urea + DAP) / farmsize — proxy for farming intensity."),
    ("yield_per_hectare", "Expected yield per operated hectare — key productivity indicator."),
    ("income_per_family_member", "Total estimated income divided by family size — welfare indicator."),
    ("total_estimated_income", "Sum of primary farm income and income from other farm activities."),
    ("total_estimated_cost", "Sum of production expenses and estimated total costs."),
    ("net_income", "Total income minus total cost — primary creditworthiness indicator."),
    ("decision", "Target label: Eligible, Review, or Not Eligible."),
]


def populate_chroma() -> None:
    """Upsert all feature definitions into the credit_features ChromaDB collection."""
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.embedder_model,
    )

    client = chromadb.PersistentClient(path=str(config.chroma_db_path))
    collection = client.get_or_create_collection(
        name="credit_features",
        embedding_function=embedder,
    )

    ids = [f"feature_{i}" for i in range(len(FEATURE_DEFINITIONS))]
    documents = [defn for _, defn in FEATURE_DEFINITIONS]
    metadatas = [{"feature_name": name} for name, _ in FEATURE_DEFINITIONS]

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    logger.info("ChromaDB: upserted %d feature definitions into 'credit_features'", len(FEATURE_DEFINITIONS))


if __name__ == "__main__":
    logger.info("Populating ChromaDB collection 'credit_features'...")
    populate_chroma()
    logger.info("ChromaDB population complete")
