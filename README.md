# RAG Disease Improvement

A Vietnamese medical RAG project for disease information lookup.

The project:
- crawls disease article links from Tam Anh Hospital
- scrapes and cleans article content
- chunks and embeds the text with `multilingual-e5-small`
- retrieves results with vector search + BM25 + reranking
- serves answers and sources through a FastAPI backend and a Vite frontend

## Project Flow

1. Crawl the A-Z disease index:
   `python data_prep/crawl_disease_index.py --save-html`
2. Scrape disease pages into JSONL:
   `python data_prep/scape_disease.py`
3. Build chunks, embeddings, and metadata:
   `python src/pipelines/embed_health_disease.py --chunk-mode title`
4. Start the backend API:
   `python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000`
5. Start the frontend:
   `cd closer-landing-page && npm run dev`

## Run Locally

### Backend

1. Install Python dependencies:
   `pip install -r requirements.txt`
2. Optional: set `GEMINI_API_KEY` if you want answer generation.
3. Run:
   `python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000`

### Frontend

1. Install Node dependencies:
   `cd closer-landing-page && npm install`
2. Run:
   `npm run dev`

Frontend: `http://localhost:3000`  
Backend docs: `http://localhost:8000/docs`

## Data Outputs

- `Data/processed/health_disease_links.csv`
- `Data/processed/health_disease_content.jsonl`
- `Data/processed/health_disease_chunks.jsonl`
- `Data/vectorstore/health_disease_e5_small_embeddings.npy`
- `Data/vectorstore/health_disease_e5_small_metadata.jsonl`
