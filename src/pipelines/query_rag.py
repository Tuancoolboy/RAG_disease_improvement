#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import (
    HEALTH_METADATA_JSONL,
    HEALTH_VECTORSTORE_EMBEDDINGS,
    HYBRID_BM25_TOP_K,
    HYBRID_RERANK_TOP_K,
    HYBRID_VECTOR_TOP_K,
)
from src.services import RAGService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid RAG search with vector + BM25 + reranking.")
    parser.add_argument("--query", required=True, help="User question")
    parser.add_argument("--metadata", type=Path, default=HEALTH_METADATA_JSONL, help="Metadata JSONL path")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=HEALTH_VECTORSTORE_EMBEDDINGS,
        help="Embeddings NPY path",
    )
    parser.add_argument("--vector-top-k", type=int, default=HYBRID_VECTOR_TOP_K, help="Vector retriever top-k")
    parser.add_argument("--bm25-top-k", type=int, default=HYBRID_BM25_TOP_K, help="BM25 retriever top-k")
    parser.add_argument("--rerank-top-k", type=int, default=HYBRID_RERANK_TOP_K, help="Final rerank top-k")
    parser.add_argument("--no-answer", action="store_true", help="Only print retrieved chunks")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    service = RAGService(
        metadata_path=args.metadata,
        embeddings_path=args.embeddings,
    )
    results = service.search(
        args.query,
        vector_top_k=args.vector_top_k,
        bm25_top_k=args.bm25_top_k,
        rerank_top_k=args.rerank_top_k,
    )

    print(f"Query: {args.query}")
    print(f"Retrieved: {len(results)}")
    for idx, item in enumerate(results, start=1):
        preview = item.chunk_body[:220].replace("\n", " ")
        print(
            f"{idx}. rerank={item.rerank_score:.4f} | "
            f"vector={item.vector_score:.4f} | bm25={item.bm25_score:.4f}"
        )
        print(f"   title: {item.chunk_title}")
        print(f"   chunk_id: {item.chunk_id}")
        print(f"   preview: {preview}")

    if args.no_answer:
        return

    answer = service.generate_answer(args.query, results)
    print("\nAnswer:\n")
    print(answer)


if __name__ == "__main__":
    main()
