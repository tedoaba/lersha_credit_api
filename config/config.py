import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]

class Config:

    def __init__(self):
        self.output_dir = os.getenv("OUTPUT_DIR", BASE_DIR / "output/")
        self.model_dir = os.getenv("MODEL_DIR", BASE_DIR / "models")
        self.log_dir = os.getenv("LOG_DIR", BASE_DIR / "logs")
        self.log_file = os.getenv("LOG_FILE", BASE_DIR / "logs" / "credit_scoring_model.log")
        self.run_id = os.getenv("RUN_ID", "default_run")

        self.target_column = os.getenv("TARGET_COLUMN", "decision")
        self.id_column = os.getenv("ID_COLUMN", "farmer_uid")
        
        self.columns_34 = ['gender', 'age_group', 'family_size', 'typeofhouse',
                            'asset_ownership', 'water_reserve_access', 'output_storage_type',
                            'decision_making_role', 'hasrusacco', 'haslocaledir',
                            'primaryoccupation', 'holdsleadershiprole', 'land_title',
                            'rented_farm_land', 'own_farmland_size', 'family_farmland_size', 'flaw',
                            'farm_mechanization', 'agriculture_experience',
                            'institutional_support_score', 'farmsizehectares', 'seedtype',
                            'seedquintals', 'expectedyieldquintals', 'saleableyieldquintals',
                            'ureafertilizerquintals', 'dapnpsfertilizerquintals', 'input_intensity',
                            'yield_per_hectare', 'income_per_family_member',
                            'total_estimated_income', 'total_estimated_cost', 'net_income',
                            'decision']

        self.xgb_model_34 = os.getenv("XGB_MODEL_34", BASE_DIR / "models" / "xgboost_34_credit_score.pkl")
        self.rf_model_34 = os.getenv("XGB_MODEL_34", BASE_DIR / "models" / "random_forest_34_credit_score.pkl")
        self.cab_model_34 = os.getenv("XGB_MODEL_34", BASE_DIR / "models" / "catboost_34_credit_score.pkl")

        self.testing_csv_path = os.getenv("CSV_PATH", BASE_DIR / "data" / "testing_dataset_final.csv")
        self.csv_general = os.getenv("CSV_GENERAL", BASE_DIR / "data" / "merged_dataset_with_name.csv")

        self.feature_column_34 = os.getenv("FEATURE_COLUMN_34", BASE_DIR / "models" / "34_feature_columns.pkl")
        self.target_column_34 = os.getenv("TARGET_COLUMN_34", BASE_DIR / "models" / "34_label_classes.pkl")

        self.db_uri = os.getenv("DB_URI")
        self.db_table = os.getenv("DB_TABLE")

        self.candidate_raw_data_table = os.getenv("CANDIDATE_RAW_DATA_TABLE", "candidate_raw_data_table")
        self.candidate_result = os.getenv("CANDIDATE_RESULT", "candidate_result")
        self.farmer_data_all = os.getenv("FARMER_DATA_ALL")

        self.prompt_path = os.getenv("PROMPT_PATH", BASE_DIR / "prompts" / "prompts.yaml")
        self.gemini_model_id = os.getenv("GEMINI_MODEL")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.embedder_model = os.getenv("EMBEDDER_MODEL", "all-MiniLM-L6-v2")

        self.shap_path = os.getenv("SHAP_PATH", BASE_DIR / "output" / "shap_summary")

        self.model_name = os.getenv("MODEL_NAME", "credit_scoring")
        self.model_stage = os.getenv("STAGE_NAME", "Staging") # "Production" or "Staging"
        self.mlflow_experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "Credit Scoring Model")
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI",(BASE_DIR / "mlruns").as_uri())
        self.catboost_dir = os.getenv("CATBOOST_DIR", BASE_DIR / "mlruns" / "catboost")
        self.training_tag = os.getenv("TRAINING_TAG", "training")
        self.inference_tag = os.getenv("INFERENCE_TAG", "inference")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        if not self.gemini_model_id:
            raise ValueError("GEMINI_MODEL environment variable not set")

        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")


config = Config()
