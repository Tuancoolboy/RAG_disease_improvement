from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import (
    CROSS_ENCODER_BATCH_SIZE,
    CROSS_ENCODER_MAX_TOKENS,
    CROSS_ENCODER_MODEL_NAME,
    HEALTH_METADATA_JSONL,
    HEALTH_VECTORSTORE_EMBEDDINGS,
    HYBRID_BM25_TOP_K,
    HYBRID_RERANK_TOP_K,
    HYBRID_VECTOR_TOP_K,
)
from src.embeddings import E5MultilingualEmbedder


def normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def tokenize_text(text: str) -> list[str]:
    return re.findall(r"\w+", normalize_text(text), flags=re.UNICODE)


def chunk_document_text(metadata: dict) -> str:
    chunk_title = (metadata.get("chunk_title") or "").strip()
    chunk_body = (metadata.get("chunk_body") or "").strip()
    if chunk_title and chunk_body:
        return f"{chunk_title}\n\n{chunk_body}"
    return chunk_title or chunk_body


@dataclass(slots=True)
class SearchResult:
    chunk_id: str
    chunk_title: str
    chunk_body: str
    source_title: str
    source_url: str
    chunk_index: int
    score: float
    rerank_score: float
    vector_score: float
    bm25_score: float
    vector_rank: int | None
    bm25_rank: int | None
    rerank_rank: int
    metadata: dict


class BM25Retriever:
    def __init__(
        self,
        metadata_records: list[dict],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.metadata_records = metadata_records
        self.k1 = k1
        self.b = b

        self.doc_tokens = [tokenize_text(chunk_document_text(record)) for record in metadata_records]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / max(1, len(self.doc_lengths))

        self.term_freqs: list[dict[str, int]] = []
        self.doc_freqs: dict[str, int] = {}
        for tokens in self.doc_tokens:
            term_freq: dict[str, int] = {}
            for token in tokens:
                term_freq[token] = term_freq.get(token, 0) + 1
            self.term_freqs.append(term_freq)
            for token in set(tokens):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.num_docs = len(metadata_records)
        self.idf = {
            token: math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))
            for token, df in self.doc_freqs.items()
        }

    def search(self, query: str, *, top_k: int = HYBRID_BM25_TOP_K) -> list[tuple[int, float]]:
        query_tokens = tokenize_text(query)
        if not query_tokens:
            return []

        scores = np.zeros(self.num_docs, dtype=np.float32)
        for idx, term_freq in enumerate(self.term_freqs):
            doc_len = self.doc_lengths[idx]
            score = 0.0
            for token in query_tokens:
                tf = term_freq.get(token, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(token, 0.0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(1.0, self.avgdl))
                score += idf * numerator / denominator
            scores[idx] = score

        ranked_indices = np.argsort(scores)[::-1]
        results: list[tuple[int, float]] = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append((int(idx), score))
            if len(results) >= top_k:
                break
        return results


class VectorRetriever:
    def __init__(
        self,
        embeddings: np.ndarray,
        metadata_records: list[dict],
        embedder: E5MultilingualEmbedder,
    ) -> None:
        self.embeddings = embeddings.astype(np.float32)
        self.metadata_records = metadata_records
        self.embedder = embedder

    def search(self, query: str, *, top_k: int = HYBRID_VECTOR_TOP_K) -> list[tuple[int, float]]:
        query_embedding = self.embedder.embed_query(query).astype(np.float32)
        scores = self.embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]


class CrossEncoderReranker:
    def __init__(
        self,
        *,
        model_name: str = CROSS_ENCODER_MODEL_NAME,
        max_length: int = CROSS_ENCODER_MAX_TOKENS,
        batch_size: int = CROSS_ENCODER_BATCH_SIZE,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or self._pick_device()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _pick_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []

        scores: list[float] = []
        for start in range(0, len(documents), self.batch_size):
            batch_docs = documents[start:start + self.batch_size]
            encoded = self.tokenizer(
                [query] * len(batch_docs),
                batch_docs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with torch.no_grad():
                logits = self.model(**encoded).logits
                if logits.ndim == 2 and logits.shape[1] > 1:
                    batch_scores = logits[:, -1]
                else:
                    batch_scores = logits.view(-1)

            scores.extend(batch_scores.detach().cpu().tolist())

        return [float(score) for score in scores]


class HybridRetriever:
    def __init__(
        self,
        *,
        metadata_path: Path = HEALTH_METADATA_JSONL,
        embeddings_path: Path = HEALTH_VECTORSTORE_EMBEDDINGS,
        embedder: E5MultilingualEmbedder | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.metadata_path = metadata_path
        self.embeddings_path = embeddings_path
        self.metadata_records = self._load_metadata(metadata_path)
        self.embeddings = np.load(embeddings_path)
        if self.embeddings.shape[0] != len(self.metadata_records):
            raise ValueError("Embeddings row count does not match metadata row count.")

        self.embedder = embedder or E5MultilingualEmbedder()
        self.reranker = reranker or CrossEncoderReranker()
        self.vector_retriever = VectorRetriever(self.embeddings, self.metadata_records, self.embedder)
        self.bm25_retriever = BM25Retriever(self.metadata_records)

    def _load_metadata(self, path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def search(
        self,
        query: str,
        *,
        vector_top_k: int = HYBRID_VECTOR_TOP_K,
        bm25_top_k: int = HYBRID_BM25_TOP_K,
        rerank_top_k: int = HYBRID_RERANK_TOP_K,
    ) -> list[SearchResult]:
        vector_hits = self.vector_retriever.search(query, top_k=vector_top_k)
        bm25_hits = self.bm25_retriever.search(query, top_k=bm25_top_k)

        vector_map = {idx: score for idx, score in vector_hits}
        bm25_map = {idx: score for idx, score in bm25_hits}
        vector_rank_map = {idx: rank for rank, (idx, _) in enumerate(vector_hits, start=1)}
        bm25_rank_map = {idx: rank for rank, (idx, _) in enumerate(bm25_hits, start=1)}

        candidate_indices = list(dict.fromkeys([idx for idx, _ in vector_hits] + [idx for idx, _ in bm25_hits]))
        if not candidate_indices:
            return []

        candidate_docs = [chunk_document_text(self.metadata_records[idx]) for idx in candidate_indices]
        rerank_scores = self.reranker.score(query, candidate_docs)

        results: list[SearchResult] = []
        for idx, candidate_idx in enumerate(candidate_indices):
            metadata = self.metadata_records[candidate_idx]
            chunk_title = (metadata.get("chunk_title") or "").strip()
            chunk_body = (metadata.get("chunk_body") or "").strip()
            rerank_score = float(rerank_scores[idx])

            results.append(
                SearchResult(
                    chunk_id=metadata.get("chunk_id", ""),
                    chunk_title=chunk_title,
                    chunk_body=chunk_body,
                    source_title=metadata.get("source_title", ""),
                    source_url=metadata.get("source_url", ""),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    score=rerank_score,
                    rerank_score=rerank_score,
                    vector_score=float(vector_map.get(candidate_idx, 0.0)),
                    bm25_score=float(bm25_map.get(candidate_idx, 0.0)),
                    vector_rank=vector_rank_map.get(candidate_idx),
                    bm25_rank=bm25_rank_map.get(candidate_idx),
                    rerank_rank=0,
                    metadata=metadata,
                )
            )

        results.sort(key=lambda item: item.rerank_score, reverse=True)
        for rank, item in enumerate(results, start=1):
            item.rerank_rank = rank
        return results[:rerank_top_k]
