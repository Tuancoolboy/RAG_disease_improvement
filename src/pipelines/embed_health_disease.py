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

from src.chunking import TitleChunker, TokenChunker
from src.config import (
    E5_BATCH_SIZE,
    E5_MAX_TOKENS,
    E5_MODEL_NAME,
    E5_OVERLAP,
    HEALTH_CHUNKS_JSONL,
    HEALTH_CONTENT_JSONL,
    HEALTH_METADATA_JSONL,
    HEALTH_VECTORSTORE_EMBEDDINGS,
)
from src.embeddings import E5MultilingualEmbedder

INPUT_JSONL = HEALTH_CONTENT_JSONL
CHUNKS_OUTPUT = HEALTH_CHUNKS_JSONL
EMBEDDINGS_OUTPUT = HEALTH_VECTORSTORE_EMBEDDINGS
METADATA_OUTPUT = HEALTH_METADATA_JSONL


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
        default=E5_MODEL_NAME,
        help="Hugging Face embedding model",
    )
    parser.add_argument(
        "--chunk-mode",
        choices=("title", "token"),
        default="title",
        help="Chunk by section title or by token window.",
    )
    parser.add_argument("--max-tokens", type=int, default=E5_MAX_TOKENS, help="Max model input tokens")
    parser.add_argument("--overlap", type=int, default=E5_OVERLAP, help="Token overlap between chunks")
    parser.add_argument("--batch-size", type=int, default=E5_BATCH_SIZE, help="Embedding batch size")
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


def build_chunk_records(input_path: Path, chunker, *, chunk_mode: str, max_tokens: int) -> list[dict]:
    chunk_records: list[dict] = []

    for line_no, record, body_text in iter_source_rows(input_path):
        source_token_count = chunker.count_tokens(body_text)
        if chunk_mode == "title":
            record_sections = record.get("sections") or []
            if record_sections:
                chunks = []
                source_title = record.get("title", "")
                for idx, section in enumerate(record_sections):
                    section_title = (section.get("section_title") or source_title).strip()
                    section_content = (section.get("content") or "").strip()
                    if section_title == source_title and section_content:
                        chunk_text = "\n\n".join([source_title, section_content]).strip()
                    elif section_content:
                        chunk_text = "\n\n".join([source_title, section_title, section_content]).strip()
                    else:
                        chunk_text = "\n\n".join([source_title, section_title]).strip()

                    token_count = chunker.count_tokens(chunk_text)
                    chunks.append({
                        "chunk_id": f"{record['slug']}::section::{idx}",
                        "chunk_index": idx,
                        "section_title": section_title,
                        "section_block_count": None,
                        "chunk_token_count": token_count,
                        "text": chunk_text,
                    })
            else:
                chunks = []
                for chunk in chunker.split(record.get("title", ""), body_text):
                    chunks.append({
                        "chunk_id": f"{record['slug']}::section::{chunk.section_index}",
                        "chunk_index": chunk.section_index,
                        "section_title": chunk.section_title,
                        "section_block_count": chunk.block_count,
                        "chunk_token_count": chunk.token_count,
                        "text": chunk.text,
                    })

            for chunk in chunks:
                chunk_records.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "source_line": line_no,
                        "source_title": record.get("title", ""),
                        "source_slug": record.get("slug", ""),
                        "source_url": record.get("url", ""),
                        "source_path": record.get("path", ""),
                        "published_date": record.get("published_date", ""),
                        "doctor_review": record.get("doctor_review", ""),
                        "chunk_index": chunk["chunk_index"],
                        "chunk_count": len(chunks),
                        "section_title": chunk["section_title"],
                        "section_block_count": chunk["section_block_count"],
                        "source_body_token_count": source_token_count,
                        "chunk_token_count": chunk["chunk_token_count"],
                        "chunk_mode": chunk_mode,
                        "was_chunked": len(chunks) > 1,
                        "exceeds_embedding_limit": chunk["chunk_token_count"] > max_tokens,
                        "text": chunk["text"],
                    }
                )
        else:
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
                        "chunk_mode": chunk_mode,
                        "was_chunked": len(chunks) > 1,
                        "exceeds_embedding_limit": chunk.token_count > max_tokens,
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
    if args.chunk_mode == "title":
        chunker = TitleChunker(embedder.tokenizer)
    else:
        chunker = TokenChunker(
            embedder.tokenizer,
            max_input_tokens=args.max_tokens,
            overlap=args.overlap,
            text_prefix=embedder.passage_prefix,
        )

    chunk_records = build_chunk_records(
        args.input,
        chunker,
        chunk_mode=args.chunk_mode,
        max_tokens=args.max_tokens,
    )
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
    chunks_over_limit = sum(1 for record in chunk_records if record["exceeds_embedding_limit"])

    print(f"Documents: {total_docs}")
    print(f"Chunks: {total_chunks}")
    print(f"Chunked documents: {chunked_docs}")
    print(f"Chunk mode: {args.chunk_mode}")
    print(f"Chunks over embedding limit: {chunks_over_limit}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Chunks JSONL: {args.chunks_output}")
    print(f"Metadata JSONL: {args.metadata_output}")
    print(f"Embeddings NPY: {args.embeddings_output}")


if __name__ == "__main__":
    main()
