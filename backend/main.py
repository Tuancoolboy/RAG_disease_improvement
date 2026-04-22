from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import BACKEND_CORS_ORIGINS, BACKEND_HOST, BACKEND_PORT  # noqa: E402
from src.services import get_default_rag_service  # noqa: E402


class AskRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000)
    vector_top_k: int = Field(default=20, ge=1, le=100)
    bm25_top_k: int = Field(default=20, ge=1, le=100)
    rerank_top_k: int = Field(default=8, ge=1, le=20)
    include_answer: bool = True


app = FastAPI(
    title="Disease RAG API",
    version="1.0.0",
    description="Hybrid retrieval + cross-encoder rerank + Gemini answer API for the disease corpus.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=BACKEND_CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/ask")
def ask(request: AskRequest) -> dict:
    service = get_default_rag_service()
    try:
        return service.ask(
            request.query,
            vector_top_k=request.vector_top_k,
            bm25_top_k=request.bm25_top_k,
            rerank_top_k=request.rerank_top_k,
            include_answer=request.include_answer,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True)
