import difflib
import random
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import ollama
import weaviate

WEAVIATE_HOST = "localhost"
WEAVIATE_HTTP_PORT = 8090
WEAVIATE_GRPC_PORT = 50051

COLLECTION_NAME = "MobilePhones"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

MODELS = [
    ("Qwen2.5", "qwen2.5:0.5b"),
    ("Llama3.2", "llama3.2:1b"),
    ("Gemma3", "gemma3:270m"),
]


embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


client = weaviate.connect_to_local(
    host=WEAVIATE_HOST,
    port=WEAVIATE_HTTP_PORT,
    grpc_port=WEAVIATE_GRPC_PORT
)

collection = client.collections.get(COLLECTION_NAME)


def ensure_models_available():
    try:
        installed = {
            m["model"] if isinstance(m, dict) else m.model
            for m in ollama.list().get("models", [])
        }
    except Exception:
        installed = set()

    for display_name, model_name in MODELS:
        if model_name not in installed:
            print(f"[startup] Pulling missing model: {model_name} ({display_name})")
            try:
                ollama.pull(model_name)
            except Exception as e:
                print(f"[startup] Failed to pull {model_name}: {e}")


def ensure_collection_exists():
    if not client.collections.exists(COLLECTION_NAME):
        print(
            f"[startup] WARNING: collection '{COLLECTION_NAME}' does not exist yet. "
            f"Run ingest.py before calling /recommend, or every query will fail."
        )



@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_collection_exists()
    ensure_models_available()
    yield
    print("[shutdown] Closing Weaviate connection cleanly...")
    client.close()


app = FastAPI(
    title="Mobile Recommendation System",
    version="1.0",
    lifespan=lifespan
)



def retrieve(query, n_results=20):

    query_embedding = embedding_model.encode(query)

    response = collection.query.near_vector(
        near_vector=query_embedding.tolist(),
        limit=n_results
    )

    return response.objects


def get_unique_phones(results, max_phones=5):

    phones = []
    seen_models = set()

    for obj in results:

        meta = obj.properties
        model_name = meta["model"]

        if model_name not in seen_models:
            seen_models.add(model_name)
            phones.append(meta)

        if len(phones) == max_phones:
            break

    return phones


def ask_model(model_name, phones, query):
    shuffled_phones = phones.copy()
    rng = random.Random(f"{model_name}::{query}")
    rng.shuffle(shuffled_phones)

    context = ""
    phone_names = []

    for phone in shuffled_phones:

        phone_names.append(phone["model"])

        context += f"""
Company: {phone['company']}
Model: {phone['model']}
RAM: {phone['ram']}
Processor: {phone['processor']}
Battery: {phone['battery']}
Price: {phone['price']}
Launch Year: {phone['launch_year']}
"""

    options_list = "\n".join(f"- {name}" for name in phone_names)

    prompt = f"""You are a smartphone recommendation expert.

Example:
User Query: longest battery life
Available Phones:
- Galaxy M14
- iPhone 13
Answer: Galaxy M14

Now do the same for this real request.

User Query:
{query}

Available Phones:

{context}

You must choose EXACTLY ONE phone from this list of model names, copied exactly as written:

{options_list}

Answer with ONLY the model name on a single line. No explanation, no punctuation,
no extra words.
"""

    try:

        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0,
                "num_predict": 30
            }
        )

        raw_text = response["message"]["content"].strip()
        recommendation = raw_text.split("\n")[0].strip().strip('"').strip("'")

        if not recommendation:
            return "No Recommendation", "Empty response from model"

        return recommendation, None

    except Exception as e:
        traceback.print_exc()
        return "No Recommendation", str(e)

def get_phone_details(model_name, phones):
    """
    Returns (details_dict, match_type).
    match_type is one of: "exact", "fuzzy", "fallback", "none"

    "fallback" means the model's raw output did not match any retrieved
    phone — the entry shown is just the top vector-search result, NOT the
    model's actual choice. Keep an eye on this field; if a model shows
    "fallback" frequently, it's not really doing the recommendation task.
    """

    # 1. Exact / substring match first (handles the easy case)
    for phone in phones:
        if model_name.lower() in phone["model"].lower() \
                or phone["model"].lower() in model_name.lower():
            return dict(phone), "exact"

    # 2. Fuzzy match — catches cases where the LLM slightly rewords
    #    the model name (e.g. adds "5G", drops a hyphen, etc.)
    candidate_names = [phone["model"] for phone in phones]
    close = difflib.get_close_matches(model_name, candidate_names, n=1, cutoff=0.4)

    if close:
        matched_name = close[0]
        for phone in phones:
            if phone["model"] == matched_name:
                return dict(phone), "fuzzy"

    # 3. Fallback — never silently return "Unknown" for everything.
    if phones:
        return dict(phones[0]), "fallback"

    return {
        "company": "Unknown",
        "model": model_name,
        "ram": "Unknown",
        "battery": "Unknown",
        "processor": "Unknown",
        "launch_year": "Unknown",
        "price": "Unknown"
    }, "none"


@app.get("/")
def home():

    total_records = collection.aggregate.over_all(total_count=True).total_count

    return {
        "project": "Mobile Recommendation System",
        "vector_database": "Weaviate",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llms": [display_name for display_name, _ in MODELS],
        "total_records": total_records
    }



@app.get("/recommend")
def recommend(query: str):

    results = retrieve(query)
    phones = get_unique_phones(results)

    recommendations = []

    for display_name, model_name in MODELS:

        raw_recommendation, error = ask_model(model_name, phones, query)
        phone_details, match_type = get_phone_details(raw_recommendation, phones)

        entry = {
            "recommended_by": display_name,
            "model": phone_details.get("model"),
            "company": phone_details.get("company"),
            "ram": phone_details.get("ram"),
            "battery": phone_details.get("battery"),
            "processor": phone_details.get("processor"),
            "launch_year": phone_details.get("launch_year"),
            "price": phone_details.get("price"),
            "raw_model_output": raw_recommendation,
            "match_type": match_type,
        }

        if error:
            entry["error"] = error

        recommendations.append(entry)

    top_recommendation = recommendations[0]
    other_recommendations = recommendations[1:]

    total_records = collection.aggregate.over_all(total_count=True).total_count

    return {
        "user_query": query,
        "chunking_details": {
            "chunking_method": "Row Based Chunking",
            "total_records": total_records,
            "retrieved_chunks": len(phones)
        },
        "top_recommendation": top_recommendation,
        "other_model_recommendations": other_recommendations
    }