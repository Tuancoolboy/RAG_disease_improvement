from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.config import (
    GEMINI_API_KEY,
    HEALTH_METADATA_JSONL,
    HEALTH_VECTORSTORE_EMBEDDINGS,
    HYBRID_BM25_TOP_K,
    HYBRID_RERANK_TOP_K,
    HYBRID_VECTOR_TOP_K,
)
from src.embeddings import E5MultilingualEmbedder
from src.generation import GeminiLLM
from src.retrieval import HybridRetriever, SearchResult


class RAGService:
    def __init__(
        self,
        *,
        metadata_path: Path = HEALTH_METADATA_JSONL,
        embeddings_path: Path = HEALTH_VECTORSTORE_EMBEDDINGS,
        gemini_api_key: str = GEMINI_API_KEY,
    ) -> None:
        self.metadata_path = metadata_path
        self.embeddings_path = embeddings_path
        self.embedder = E5MultilingualEmbedder()
        self.retriever = HybridRetriever(
            metadata_path=metadata_path,
            embeddings_path=embeddings_path,
            embedder=self.embedder,
        )
        self.gemini = GeminiLLM(api_key=gemini_api_key)

    def search(
        self,
        query: str,
        *,
        vector_top_k: int = HYBRID_VECTOR_TOP_K,
        bm25_top_k: int = HYBRID_BM25_TOP_K,
        rerank_top_k: int = HYBRID_RERANK_TOP_K,
    ) -> list[SearchResult]:
        return self.retriever.search(
            query,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            rerank_top_k=rerank_top_k,
        )

    def generate_answer(self, query: str, results: list[SearchResult]) -> str:
        return self.gemini.generate_answer(query, results)

    def ask(
        self,
        query: str,
        *,
        vector_top_k: int = HYBRID_VECTOR_TOP_K,
        bm25_top_k: int = HYBRID_BM25_TOP_K,
        rerank_top_k: int = HYBRID_RERANK_TOP_K,
        include_answer: bool = True,
    ) -> dict:
        results = self.search(
            query,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
            rerank_top_k=rerank_top_k,
        )
        answer = self.generate_answer(query, results) if include_answer else None
        return {
            "query": query,
            "answer": answer,
            "retrieved_count": len(results),
            "sources": [self.serialize_result(index, result) for index, result in enumerate(results, start=1)],
        }

    def serialize_result(self, rank: int, result: SearchResult) -> dict:
        preview = result.chunk_body.replace("\n", " ").strip()
        if len(preview) > 280:
            preview = preview[:277].rstrip() + "..."

        return {
            "rank": rank,
            "chunk_id": result.chunk_id,
            "chunk_title": result.chunk_title,
            "chunk_body": result.chunk_body,
            "preview": preview,
            "source_title": result.source_title,
            "source_url": result.source_url,
            "chunk_index": result.chunk_index,
            "rerank_score": result.rerank_score,
            "vector_score": result.vector_score,
            "bm25_score": result.bm25_score,
            "vector_rank": result.vector_rank,
            "bm25_rank": result.bm25_rank,
        }


@lru_cache(maxsize=1)
def get_default_rag_service() -> RAGService:
    return RAGService()
