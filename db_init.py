

from services.db_utils import load_data_to_database, create_candidate_result_table
from src.logger import get_logger
from config.config import config

logger = get_logger(__name__)


if __name__ == "__main__":
    logger.info("Loading to Database initialized...")
    load_data_to_database(config.csv_general, config.farmer_data_all)
    load_data_to_database(config.testing_csv_path, config.candidate_raw_data_table)

    logger.info("Creating Prediction result table...")
    create_candidate_result_table(config.candidate_result)
