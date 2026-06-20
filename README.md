# Mobile Recommendation RAG System

A Retrieval-Augmented Generation (RAG) based Mobile Recommendation System that recommends smartphones using semantic search, vector embeddings, and multiple Large Language Models (LLMs).

## Features

* Semantic mobile search using embeddings
* Weaviate Vector Database for storage and retrieval
* Multi-LLM recommendation comparison
* FastAPI REST API
* Row-Based Chunking
* Metadata-aware retrieval
* Natural language query support

## Tech Stack

* Python
* FastAPI
* Weaviate
* Ollama
* Sentence Transformers
* Pandas

## Models Used

### Embedding Model

* BAAI/bge-small-en-v1.5

### LLM Models

* Qwen2.5 (0.5B)
* Llama3.2 (1B)
* Gemma3 (270M)

## Architecture

Dataset (CSV)
→ Row-Based Chunking
→ Embedding Generation
→ Weaviate Vector Database
→ Semantic Retrieval
→ LLM Recommendation Engine
→ FastAPI Response

## Project Workflow

1. Load mobile dataset
2. Create row-based chunks
3. Generate embeddings using BGE-Small
4. Store vectors in Weaviate
5. Retrieve relevant mobile phones using semantic search
6. Compare recommendations from multiple LLMs
7. Return structured recommendation output

## API Endpoint

### Get Mobile Recommendation

```http
GET /recommend?query=best gaming phone under 50000
```

### Example Query

```text
Best Android phone with good battery and camera
```

## Sample Output

```json
{
  "user_query": "Best gaming phone",
  "top_recommendation": {
    "recommended_by": "Qwen2.5",
    "company": "Samsung",
    "model": "Galaxy S24 Ultra",
    "ram": "12GB",
    "battery": "5000mAh",
    "processor": "Snapdragon 8 Gen 3"
  }
}
```
