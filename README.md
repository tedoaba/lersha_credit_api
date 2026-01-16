# credit_scoring

## Install Libraries

```bash
poetry update
poetry run pip install chromadb sentence_transformers 
poetry run pip install -r requirements.txt

```

### Run Model Training

- Uncomment imports and the function call `load_data_to_database` in `train.py` file and upload data to database
- then comment the training code.

```bash

poetry run python main.py

```

### Model Inference via FastAPI

```bash

poetry run uvicorn app:app --reload

```

### Run Mlflow Server

```bash

poetry run mlflow ui --backend-store-uri mlruns
poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db


```
