#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.chunking import TokenChunker
from src.embeddings import E5MultilingualEmbedder

INPUT_JSONL = REPO_ROOT / "Data" / "processed" / "health_disease_content.jsonl"
CHUNKS_OUTPUT = REPO_ROOT / "Data" / "processed" / "health_disease_chunks.jsonl"
EMBEDDINGS_OUTPUT = REPO_ROOT / "Data" / "vectorstore" / "health_disease_e5_small_embeddings.npy"
METADATA_OUTPUT = REPO_ROOT / "Data" / "vectorstore" / "health_disease_e5_small_metadata.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk body_text rows from health_disease_content.jsonl and embed them with multilingual-e5-small.",
    )
    parser.add_argument("--input", type=Path, default=INPUT_JSONL, help="Input health_disease JSONL")
    parser.add_argument("--chunks-output", type=Path, default=CHUNKS_OUTPUT, help="Chunk JSONL output")
    parser.add_argument(
        "--embeddings-output",
        type=Path,
        default=EMBEDDINGS_OUTPUT,
        help="NumPy embeddings output (.npy)",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=METADATA_OUTPUT,
        help="Metadata JSONL aligned with embedding rows",
    )
    parser.add_argument(
        "--model-name",
        default="intfloat/multilingual-e5-small",
        help="Hugging Face embedding model",
    )
    parser.add_argument("--max-tokens", type=int, default=512, help="Max model input tokens")
    parser.add_argument("--overlap", type=int, default=64, help="Token overlap between chunks")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    return parser.parse_args()


def iter_source_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            body_text = (record.get("body_text") or "").strip()
            if not body_text:
                continue
            yield line_no, record, body_text


def build_chunk_records(input_path: Path, chunker: TokenChunker) -> list[dict]:
    chunk_records: list[dict] = []

    for line_no, record, body_text in iter_source_rows(input_path):
        source_token_count = chunker.count_tokens(body_text)
        chunks = chunker.split(body_text)

        for chunk_index, chunk in enumerate(chunks):
            chunk_records.append(
                {
                    "chunk_id": f"{record['slug']}::chunk::{chunk_index}",
                    "source_line": line_no,
                    "source_title": record.get("title", ""),
                    "source_slug": record.get("slug", ""),
                    "source_url": record.get("url", ""),
                    "source_path": record.get("path", ""),
                    "published_date": record.get("published_date", ""),
                    "doctor_review": record.get("doctor_review", ""),
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                    "source_body_token_count": source_token_count,
                    "chunk_token_count": chunk.token_count,
                    "was_chunked": len(chunks) > 1,
                    "text": chunk.text,
                }
            )

    return chunk_records


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input JSONL not found: {args.input}")

    embedder = E5MultilingualEmbedder(
        model_name=args.model_name,
        max_length=args.max_tokens,
    )
    chunker = TokenChunker(
        embedder.tokenizer,
        max_input_tokens=args.max_tokens,
        overlap=args.overlap,
        text_prefix=embedder.passage_prefix,
    )

    chunk_records = build_chunk_records(args.input, chunker)
    if not chunk_records:
        raise RuntimeError("No body_text rows were found to embed.")

    texts = [record["text"] for record in chunk_records]
    embeddings = embedder.embed_passages(texts, batch_size=args.batch_size)

    args.embeddings_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.embeddings_output, embeddings)
    save_jsonl(chunk_records, args.chunks_output)
    save_jsonl(chunk_records, args.metadata_output)

    total_docs = len({record["source_line"] for record in chunk_records})
    total_chunks = len(chunk_records)
    chunked_docs = sum(1 for record in chunk_records if record["chunk_index"] == 0 and record["was_chunked"])

    print(f"Documents: {total_docs}")
    print(f"Chunks: {total_chunks}")
    print(f"Chunked documents: {chunked_docs}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Chunks JSONL: {args.chunks_output}")
    print(f"Metadata JSONL: {args.metadata_output}")
    print(f"Embeddings NPY: {args.embeddings_output}")


if __name__ == "__main__":
    main()
