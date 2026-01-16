import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from services.schema import CreditScoringRecord
from services.db_model import CreditScoringRecordDB
from config.config import config
from src.logger import get_logger

logger = get_logger(__name__)


def db_engine():
    return create_engine(config.db_uri)

def create_candidate_result_table(table_name):
    engine = create_engine(config.db_uri)

    create_table_query = text(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,

        farmer_uid VARCHAR(100) NOT NULL,
        first_name VARCHAR(100) NOT NULL,
        middle_name VARCHAR(100) NOT NULL,
        last_name VARCHAR(100) NOT NULL,


        predicted_class_name VARCHAR(100) NOT NULL,
        top_feature_contributions JSONB NOT NULL,
        rag_explanation TEXT NOT NULL,
        model_name TEXT NOT NULL,

        timestamp TIMESTAMPTZ NOT NULL
    );
    """)

    with engine.connect() as connection:
        connection.execute(create_table_query)
        connection.commit()

    logger.info("Table created successfully.")


def load_data_to_database(csv_path, table_name):
    logger.info("Loading data from CSV to database...")
    df = pd.read_csv(csv_path)

    engine = create_engine(config.db_uri)

    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="replace",
        index=False
    )

    logger.info("CSV Dataset Successfully Uploaded to Database!")


def get_data_from_database(table_name):
    logger.info("Fetching data from database...")
    engine = db_engine()
    query = text(f"SELECT * FROM {table_name}")
    df = pd.read_sql(query, con=engine)
    logger.info("Data fetch complete!")
    return df


def fetch_rows(table_name, filters=None):
    """
    filters: dict of {column: value} or None
    """
    engine = db_engine()

    if not filters:
        query = text(f"SELECT * FROM {table_name}")
        return pd.read_sql(query, con=engine)

    where_clause = " AND ".join([f"{col} = :{col}" for col in filters])
    query = text(f"SELECT * FROM {config.db_table} WHERE {where_clause}")
    return pd.read_sql(query, con=engine, params=filters)


def fetch_raw_data(table_name, filters):
    engine = db_engine()
    query = text(f"SELECT * FROM {table_name}") 
    df = pd.read_sql(query, con=engine)
    df = df[df["farmer_uid"] == filters]
    return df

def fetch_multiple_raw_data(table_name, n_rows=3):
    engine = db_engine()

    query = text(f"SELECT * FROM {table_name}") 
    df = pd.read_sql(query, con=engine)

    sample_df = df.sample(n=n_rows, replace=False, random_state=None)

    return sample_df


def fetch_and_filter_columns(table_name, columns_to_select, column_order):
    """
    Fetch all data from a table, filter specific columns, reorder them,
    and return as a pandas DataFrame.

    Args:
        db_path (str): Path to the SQLite database.
        table_name (str): Name of the table to query.
        columns_to_select (list): Columns to keep from the table.
        column_order (list): Final explicit order of columns to return.

    Returns:
        pandas.DataFrame: Filtered and reordered dataset.
    """
    engine = db_engine()

    try:
        query = text(f"SELECT * FROM {table_name}") 
        df = pd.read_sql(query, con=engine)

        df = df[columns_to_select]

        df = df[column_order]

        return df

    finally:
        engine.close()


def get_row_by_number(table, row_number, order_by="age"):
    """
    row_number: 0-based index
    """
    engine = db_engine()
    query = f"""
        SELECT *
        FROM {table}
        ORDER BY {order_by}
        LIMIT 1 OFFSET {row_number};
    """
    return pd.read_sql(query, con=engine)


def save_batch_evaluations(input_df, evaluation_results: list):
    """
    Save only selected input columns + model outputs.
    Selected columns: farmer_uid, age, gender
    """

    engine = db_engine()
    session = Session(bind=engine)

    try:
        for i, result in enumerate(evaluation_results):

            row = input_df.iloc[i]

            farmer_uid = str(row.get("farmer_uid"))
            first_name = row.get("first_name")
            middle_name = row.get("middle_name")
            last_name = row.get("last_name")

            # Build unified Pydantic record
            record = CreditScoringRecord(
                farmer_uid=farmer_uid,
                first_name=first_name,
                middle_name=middle_name,
                last_name=last_name,


                predicted_class_name=result["predicted_class_name"],
                top_feature_contributions=result["top_feature_contributions"],
                rag_explanation=result["rag_explanation"],
                model_name=result["model_name"],

                timestamp=datetime.utcnow()
            )

            # Convert to SQLAlchemy record
            db_row = CreditScoringRecordDB(
                farmer_uid=record.farmer_uid,
                first_name=record.first_name,
                middle_name=record.middle_name,
                last_name=record.last_name,


                predicted_class_name=record.predicted_class_name,
                top_feature_contributions=[fc.dict() for fc in record.top_feature_contributions],
                rag_explanation=record.rag_explanation,
                model_name=record.model_name,

                timestamp=record.timestamp
            )

            session.add(db_row)

        session.commit()
        return True

    except Exception as e:
        session.rollback()
        raise e

    finally:
        session.close()
