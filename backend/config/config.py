"""Configuration singleton for the Lersha Credit Scoring backend.

All runtime settings are read from environment variables with safe defaults.
Import pattern: from backend.config.config import config

BASE_DIR resolves to the repository root (parents[2] from this file at
backend/config/config.py → parents[0]=backend/config, [1]=backend, [2]=root).

Secret reading priority (per [P9-SEC] and [P5-CONFIG]):
  1. Docker secrets file at /run/secrets/{name}  (production)
  2. Environment variable fallback               (development / CI)
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# Repository root — two levels up from backend/config/config.py
BASE_DIR = Path(__file__).resolve().parents[2]


def _read_secret(name: str, env_var: str) -> str | None:
    """Read a secret from a Docker secrets file, falling back to an environment variable.

    In production the secret is mounted at /run/secrets/{name} by Docker Compose.
    In development the environment variable set in .env is used instead.

    Args:
        name: Docker secret name, maps to the file path /run/secrets/{name}.
        env_var: Name of the fallback environment variable.

    Returns:
        The secret value (whitespace-stripped), or None if neither source is set.
    """
    secret_path = Path(f"/run/secrets/{name}")
    if secret_path.exists():
        return secret_path.read_text(encoding="utf-8").strip()
    return os.getenv(env_var)


class Config:
    """Central configuration object. Instantiated once at module load."""

    def __init__(self) -> None:
        # ── Directory paths ────────────────────────────────────────────────
        self.output_dir = os.getenv("OUTPUT_DIR", str(BASE_DIR / "output"))
        self.model_dir = os.getenv("MODEL_DIR", str(BASE_DIR / "backend" / "models"))
        self.log_dir = os.getenv("LOG_DIR", str(BASE_DIR / "logs"))
        self.log_file = os.getenv("LOG_FILE", str(BASE_DIR / "logs" / "credit_scoring_model.log"))
        self.run_id = os.getenv("RUN_ID", "default_run")

        # ── Feature / target columns ───────────────────────────────────────
        self.target_column = os.getenv("TARGET_COLUMN", "decision")
        self.id_column = os.getenv("ID_COLUMN", "farmer_uid")

        self.columns_36 = [
            "gender",
            "age_group",
            "family_size",
            "typeofhouse",
            "asset_ownership",
            "water_reserve_access",
            "output_storage_type",
            "decision_making_role",
            "hasrusacco",
            "haslocaledir",
            "primaryoccupation",
            "holdsleadershiprole",
            "land_title",
            "rented_farm_land",
            "own_farmland_size",
            "family_farmland_size",
            "flaw",
            "farm_mechanization",
            "agriculture_experience",
            "institutional_support_score",
            "farmsizehectares",
            "seedtype",
            "seedquintals",
            "expectedyieldquintals",
            "saleableyieldquintals",
            "ureafertilizerquintals",
            "dapnpsfertilizerquintals",
            "input_intensity",
            "yield_per_hectare",
            "income_per_family_member",
            "total_estimated_income",
            "total_estimated_cost",
            "net_income",
        ]

        # ── Model file paths ───────────────────────────────────────────────
        # FIXED: each model now reads from its OWN environment variable.
        # Previously rf_model_36 and cab_model_36 both incorrectly read XGB_MODEL_36.
        self.xgb_model_36 = os.getenv(
            "XGB_MODEL_36",
            str(BASE_DIR / "backend" / "models" / "xgboost_36_credit_score.pkl"),
        )
        self.rf_model_36 = os.getenv(
            "RF_MODEL_36",
            str(BASE_DIR / "backend" / "models" / "random_forest_36_credit_score.pkl"),
        )
        self.cab_model_36 = os.getenv(
            "CAB_MODEL_36",
            str(BASE_DIR / "backend" / "models" / "catboost_36_credit_score.pkl"),
        )

        # ── Feature / label pickle paths ───────────────────────────────────
        self.feature_column_36 = os.getenv(
            "FEATURE_COLUMN_36",
            str(BASE_DIR / "backend" / "models" / "36_feature_columns.pkl"),
        )
        self.target_column_36 = os.getenv(
            "TARGET_COLUMN_36",
            str(BASE_DIR / "backend" / "models" / "36_label_classes.pkl"),
        )

        # ── Data file paths ────────────────────────────────────────────────
        self.testing_csv_path = os.getenv("CSV_PATH", str(BASE_DIR / "backend" / "data" / "testing_dataset_final.csv"))
        self.csv_general = os.getenv("CSV_GENERAL", str(BASE_DIR / "backend" / "data" / "merged_dataset_with_name.csv"))

        # ── Database ───────────────────────────────────────────────────────
        self.db_uri = os.getenv("DB_URI")
        if not self.db_uri:
            raise ValueError("DB_URI environment variable not set")
        self.db_table = os.getenv("DB_TABLE")
        self.candidate_raw_data_table = os.getenv("CANDIDATE_RAW_DATA_TABLE", "candidate_raw_data_table")
        self.candidate_result = os.getenv("CANDIDATE_RESULT", "candidate_result")
        self.farmer_data_all = os.getenv("FARMER_DATA_ALL")
        if not self.farmer_data_all:
            raise ValueError("FARMER_DATA_ALL environment variable not set")

        # ── LLM / Embeddings ───────────────────────────────────────────────────────
        # Legacy path: used by rag_engine.get_rag_explanation() only.
        # New code should use prompt_version + prompt_dir via RagService.
        self.prompt_path = os.getenv("PROMPT_PATH", str(BASE_DIR / "backend" / "prompts" / "prompts.yaml"))
        self.gemini_model_id = os.getenv("GEMINI_MODEL")
        self.embedder_model = os.getenv("EMBEDDER_MODEL", "all-MiniLM-L6-v2")

        # ── RAG Prompt Versioning ──────────────────────────────────────────────
        # PROMPT_VERSION selects which vN.yaml file is loaded from prompt_dir.
        # Default: "v1" (backend/prompts/v1.yaml).
        # To switch versions without a code change, set PROMPT_VERSION=v2 and
        # ensure backend/prompts/v2.yaml exists before restarting.
        self.prompt_version: str = os.getenv("PROMPT_VERSION", "v1")
        self.prompt_dir: Path = BASE_DIR / "backend" / "prompts"

        # ── Output / SHAP ──────────────────────────────────────────────────
        self.shap_path = os.getenv("SHAP_PATH", str(BASE_DIR / "output" / "shap_summary"))

        # ── MLflow ─────────────────────────────────────────────────────────
        self.model_name = os.getenv("MODEL_NAME", "credit_scoring")
        self.model_stage = os.getenv("STAGE_NAME", "Staging")
        self.mlflow_experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Credit Scoring Model")
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", (BASE_DIR / "mlruns").as_uri())
        self.catboost_dir = os.getenv("CATBOOST_DIR", str(BASE_DIR / "mlruns" / "catboost"))
        self.training_tag = os.getenv("TRAINING_TAG", "training")
        self.inference_tag = os.getenv("INFERENCE_TAG", "inference")

        # ── Redis / Celery ─────────────────────────────────────────────────────
        self.redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")

        # ── API security ───────────────────────────────────────────────────────────
        # Reads from /run/secrets/api_key (Docker prod) or API_KEY env var (dev/CI)
        self.api_key = _read_secret("api_key", "API_KEY")
        if not self.api_key:
            raise ValueError("API_KEY environment variable (or Docker secret 'api_key') not set")

        # ── Required LLM keys ──────────────────────────────────────────────
        # Reads from /run/secrets/gemini_api_key (Docker prod) or GEMINI_API_KEY env var
        self.gemini_api_key = _read_secret("gemini_api_key", "GEMINI_API_KEY")
        if not self.gemini_model_id:
            raise ValueError("GEMINI_MODEL environment variable not set")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable (or Docker secret 'gemini_api_key') not set")

        # ── Hyperparameters from YAML ──────────────────────────────────────
        _hparams_path = BASE_DIR / "backend" / "config" / "hyperparams.yaml"
        if not _hparams_path.exists():
            raise FileNotFoundError(
                f"Required hyperparameters file not found: {_hparams_path}. "
                "Ensure backend/config/hyperparams.yaml exists before starting the application."
            )
        with open(_hparams_path, encoding="utf-8") as f:
            self.hyperparams: dict = yaml.safe_load(f) or {}

        # ── Ensure required directories exist ─────────────────────────────
        for directory in (self.output_dir, self.model_dir, self.log_dir):
            Path(directory).mkdir(parents=True, exist_ok=True)


config = Config()
