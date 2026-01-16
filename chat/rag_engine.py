
import json
import yaml
from dotenv import load_dotenv
from chromadb import Client as ChromaClient
from chromadb.utils import embedding_functions
from google.genai import Client as GeminiClient

from config.config import config

load_dotenv()

embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=config.embedder_model
)


if not config.gemini_model_id:
    raise ValueError("GEMINI_MODEL environment variable not set")

if not config.gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set")

gemini_client = GeminiClient(api_key=config.gemini_api_key)


client = ChromaClient()
collection = client.get_or_create_collection("credit_features")


def retrieve_docs(query: str, k: int = 5):
    """
    Retrieve top-k similar documents from Chroma DB for the given query.
    """
    query_embedding = embedder([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    documents = results.get("documents", [[]])[0]

    if not documents:
        return []

    return [str(doc) for doc in documents]

def get_rag_explanation(prediction: str, shap_dict: dict) -> str:
    shap_json = json.dumps(shap_dict, indent=2)

    query_text = f"Model predicted: {prediction}\nSHAP contributions: {shap_json}"
    retrieved_docs = retrieve_docs(query_text, k=5)
    context = "\n".join(retrieved_docs) if retrieved_docs else "No relevant feature definitions found in the database."

    with open(config.prompt_path, "r", encoding="utf-8") as f:
        rag_prompt = yaml.safe_load(f)

    prompt = f"""
            SYSTEM:
            {rag_prompt['system']}

            INSTRUCTIONS:
            {rag_prompt['instructions']}

            RULES:
            {rag_prompt['rules']}

            OUTPUT:
            {rag_prompt['output']}

            USER INPUT:
            prediction = {prediction}
            shap_json = {shap_json}
            context = {context}

            RESPONSE:
            Generate the paragraphs now.
                """

    response = gemini_client.models.generate_content(
        model=config.gemini_model_id, 
        contents=prompt
    )

    if hasattr(response, "text") and response.text:
        explanation = response.text
    elif hasattr(response, "candidates"):
        explanation = response.candidates[0].content.parts[0].text
    else:
        explanation = str(response)

    return explanation.strip()

if __name__ == "__main__":
    sample_prediction = "High Risk"
    sample_shap = {
        "age": -0.25,
        "income": 0.15,
        "loan_to_value": -0.30,
        "employment_length": 0.10
    }

    explanation = get_rag_explanation(sample_prediction, sample_shap)
    print("RAG Explanation:\n", explanation)
