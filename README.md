---
title: disease-rag
emoji: 🩺
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
suggested_hardware: cpu-basic
---

# RAG Disease Improvement

A Vietnamese medical RAG project that builds a disease-information pipeline from raw web pages to grounded answers with source references.

## What This Project Builds

This repository creates a complete RAG workflow:

1. collect disease article URLs
2. scrape and clean article content
3. split documents into chunks
4. embed chunks into vectors
5. retrieve relevant chunks with hybrid search
6. rerank the candidates
7. generate answers from retrieved evidence
8. expose the system through an API and a frontend

## Step-by-Step RAG Pipeline

### 1. Crawl the disease index

The first step collects canonical disease article links from the Tam Anh Hospital A-Z disease page.

- Script: `data_prep/crawl_disease_index.py`
- Input: the A-Z index page
- Output: `Data/processed/health_disease_links.csv`

Command:
`python data_prep/crawl_disease_index.py --save-html`

### 2. Scrape disease article pages

After collecting URLs, the project downloads each disease page and extracts the main article content while skipping boilerplate sections.

- Script: `data_prep/scape_disease.py`
- Input: `Data/processed/health_disease_links.csv`
- Output: `Data/processed/health_disease_content.jsonl`

This step also keeps useful fields such as title, URL, article text, and section structure for later chunking.

Command:
`python data_prep/scape_disease.py`

### 3. Convert raw articles into chunkable text

The scraped JSONL is transformed into structured content that can be split into smaller retrieval units.

- Main logic: `src/chunking/title_chunker.py`
- Alternative chunking: `src/chunking/token_chunker.py`

Two chunking strategies are supported:

- `title`: split by section headings
- `token`: split by token windows with overlap

The default pipeline uses title-based chunking because it preserves the semantic structure of medical articles better.

### 4. Embed the chunks

Each chunk is converted into a dense vector using `multilingual-e5-small`.

- Pipeline: `src/pipelines/embed_health_disease.py`
- Embedding model wrapper: `src/embeddings/e5_multilingual.py`
- Output chunks: `Data/processed/health_disease_chunks.jsonl`
- Output embeddings: `Data/vectorstore/health_disease_e5_small_embeddings.npy`
- Output metadata: `Data/vectorstore/health_disease_e5_small_metadata.jsonl`

Command:
`python src/pipelines/embed_health_disease.py --chunk-mode title`

### 5. Build hybrid retrieval

At query time, the system does not rely on only one search method.
It combines:

- vector retrieval for semantic similarity
- BM25 retrieval for keyword matching
- candidate merging across both methods

Core file:
- `src/retrieval/hybrid_retriever.py`

This design helps the system catch both meaning-based matches and exact-term matches, which is useful for medical questions.

### 6. Rerank the retrieved candidates

The hybrid retriever returns a candidate set, then a cross-encoder reranker sorts those candidates again using a stronger relevance signal.

This improves final retrieval quality before answer generation.

Implemented in:
- `src/retrieval/hybrid_retriever.py`

### 7. Generate a grounded answer

Once the top chunks are selected, the generator builds a prompt from those chunks and asks Gemini to answer only from the provided context.

- Generator: `src/generation/gemini_llm.py`
- Service layer: `src/services/rag_service.py`

The output contains:

- the final answer
- retrieved source chunks
- source titles and URLs

### 8. Serve the RAG system

The finished RAG pipeline is exposed through:

- FastAPI backend: `backend/main.py`
- Frontend app: `frontend/`

The backend handles `/api/ask` and `/api/health`.
The frontend lets users ask medical questions, inspect chunks, and view source references.

## Main Artifacts

- `Data/processed/health_disease_links.csv`
- `Data/processed/health_disease_content.jsonl`
- `Data/processed/health_disease_chunks.jsonl`
- `Data/vectorstore/health_disease_e5_small_embeddings.npy`
- `Data/vectorstore/health_disease_e5_small_metadata.jsonl`

## Minimal Local Run

If you only want to run the finished system:

Backend:
`python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000`

Frontend:
`cd frontend && npm run dev`

Frontend: `http://localhost:3000`  
Backend docs: `http://localhost:8000/docs`

## Deploy Notes

If the frontend is deployed on Vercel and the API is deployed on Render, set these environment variables:

Vercel:
- `VITE_API_BASE_URL=https://your-render-service.onrender.com`

Render:
- `RAG_BACKEND_CORS_ORIGINS=https://your-vercel-site.vercel.app`
- Optional for preview deployments: `RAG_BACKEND_CORS_ORIGIN_REGEX=https://.*\.vercel\.app`

Without `VITE_API_BASE_URL`, the frontend now uses `localhost:8000` only in local dev. In production it calls the same origin, so a separate Render backend still needs `VITE_API_BASE_URL`.

## Hugging Face Spaces

This repository can now run as a single Docker Space:

- FastAPI serves `/api/health` and `/api/ask`
- the built Vite frontend is served from the same origin
- no Vercel or Render split is required, so there is no CORS setup

Required for the Space:

- SDK: `Docker`
- App port: `7860`
- Secret: `GEMINI_API_KEY`

Runtime files needed in the Space repository:

- `Data/vectorstore/health_disease_e5_small_embeddings.npy`
- `Data/vectorstore/health_disease_e5_small_metadata.jsonl`

If those files are still ignored locally, add them explicitly when pushing to the Space repository:

`git add -f Data/vectorstore/health_disease_e5_small_embeddings.npy Data/vectorstore/health_disease_e5_small_metadata.jsonl`
