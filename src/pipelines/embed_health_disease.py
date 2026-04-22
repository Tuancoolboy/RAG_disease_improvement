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


def build_embedding_text(chunk_title: str, chunk_body: str) -> str:
    chunk_title = (chunk_title or "").strip()
    chunk_body = (chunk_body or "").strip()
    if chunk_title and chunk_body:
        return "\n\n".join([chunk_title, chunk_body]).strip()
    return chunk_title or chunk_body


def count_model_input_tokens(tokenizer, text: str, *, text_prefix: str) -> int:
    prefixed_text = f"{text_prefix}{text.strip()}"
    return len(
        tokenizer(
            prefixed_text,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )["input_ids"]
    )


def split_text_by_token_window(tokenizer, text: str, *, max_tokens: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    total_tokens = len(input_ids)

    if total_tokens <= max_tokens:
        return [text]

    effective_overlap = min(overlap, max(0, max_tokens - 1))
    step = max(1, max_tokens - effective_overlap)
    chunks: list[str] = []
    start = 0

    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        chunk_text = text[char_start:char_end].strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= total_tokens:
            break
        start += step

    return chunks


def split_section_for_embedding(
    tokenizer,
    *,
    chunk_title: str,
    chunk_body: str,
    max_tokens: int,
    overlap: int,
    text_prefix: str,
) -> list[dict]:
    embedding_text = build_embedding_text(chunk_title, chunk_body)
    model_token_count = count_model_input_tokens(tokenizer, embedding_text, text_prefix=text_prefix)

    if model_token_count <= max_tokens or not chunk_body:
        return [{
            "chunk_title": chunk_title,
            "chunk_body": chunk_body,
            "embedding_text": embedding_text,
            "chunk_token_count": model_token_count,
            "section_chunk_index": 0,
            "section_chunk_count": 1,
        }]

    fixed_prefix_text = f"{text_prefix}{chunk_title.strip()}\n\n"
    fixed_token_count = len(
        tokenizer(
            fixed_prefix_text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )["input_ids"]
    )
    special_token_count = tokenizer.num_special_tokens_to_add(pair=False)
    available_body_tokens = max_tokens - fixed_token_count - special_token_count

    if available_body_tokens <= 0:
        return [{
            "chunk_title": chunk_title,
            "chunk_body": chunk_body,
            "embedding_text": embedding_text,
            "chunk_token_count": model_token_count,
            "section_chunk_index": 0,
            "section_chunk_count": 1,
        }]

    body_chunks = split_text_by_token_window(
        tokenizer,
        chunk_body,
        max_tokens=available_body_tokens,
        overlap=overlap,
    )
    section_chunk_count = len(body_chunks)
    split_chunks: list[dict] = []

    for idx, body_part in enumerate(body_chunks):
        split_embedding_text = build_embedding_text(chunk_title, body_part)
        split_chunks.append({
            "chunk_title": chunk_title,
            "chunk_body": body_part,
            "embedding_text": split_embedding_text,
            "chunk_token_count": count_model_input_tokens(
                tokenizer,
                split_embedding_text,
                text_prefix=text_prefix,
            ),
            "section_chunk_index": idx,
            "section_chunk_count": section_chunk_count,
        })

    return split_chunks


def build_chunk_records(
    input_path: Path,
    chunker,
    *,
    chunk_mode: str,
    max_tokens: int,
    overlap: int,
    tokenizer,
    text_prefix: str,
) -> list[dict]:
    chunk_records: list[dict] = []

    for line_no, record, body_text in iter_source_rows(input_path):
        source_token_count = chunker.count_tokens(body_text)
        if chunk_mode == "title":
            record_sections = record.get("sections") or []
            raw_chunks: list[dict] = []
            if record_sections:
                source_title = record.get("title", "")
                for idx, section in enumerate(record_sections):
                    section_title = (section.get("section_title") or source_title).strip()
                    chunk_body = (section.get("content") or "").strip()
                    raw_chunks.append({
                        "section_index": idx,
                        "chunk_title": section_title,
                        "chunk_body": chunk_body,
                        "section_block_count": None,
                    })
            else:
                for chunk in chunker.split(record.get("title", ""), body_text):
                    raw_chunks.append({
                        "section_index": chunk.section_index,
                        "chunk_title": chunk.section_title,
                        "chunk_body": chunk.body_text,
                        "section_block_count": chunk.block_count,
                    })

            chunks: list[dict] = []
            for raw_chunk in raw_chunks:
                split_chunks = split_section_for_embedding(
                    tokenizer,
                    chunk_title=raw_chunk["chunk_title"],
                    chunk_body=raw_chunk["chunk_body"],
                    max_tokens=max_tokens,
                    overlap=overlap,
                    text_prefix=text_prefix,
                )
                for split_chunk in split_chunks:
                    chunks.append({
                        "chunk_id": (
                            f"{record['slug']}::section::{raw_chunk['section_index']}::part::"
                            f"{split_chunk['section_chunk_index']}"
                            if split_chunk["section_chunk_count"] > 1
                            else f"{record['slug']}::section::{raw_chunk['section_index']}"
                        ),
                        "section_index": raw_chunk["section_index"],
                        "chunk_title": split_chunk["chunk_title"],
                        "chunk_body": split_chunk["chunk_body"],
                        "section_block_count": raw_chunk["section_block_count"],
                        "chunk_token_count": split_chunk["chunk_token_count"],
                        "embedding_text": split_chunk["embedding_text"],
                        "section_chunk_index": split_chunk["section_chunk_index"],
                        "section_chunk_count": split_chunk["section_chunk_count"],
                    })

            for chunk_index, chunk in enumerate(chunks):
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
                        "chunk_index": chunk_index,
                        "chunk_count": len(chunks),
                        "section_index": chunk["section_index"],
                        "section_title": chunk["chunk_title"],
                        "section_chunk_index": chunk["section_chunk_index"],
                        "section_chunk_count": chunk["section_chunk_count"],
                        "chunk_title": chunk["chunk_title"],
                        "chunk_body": chunk["chunk_body"],
                        "section_block_count": chunk["section_block_count"],
                        "source_body_token_count": source_token_count,
                        "chunk_token_count": chunk["chunk_token_count"],
                        "chunk_mode": chunk_mode,
                        "was_chunked": len(chunks) > 1,
                        "exceeds_embedding_limit": chunk["chunk_token_count"] > max_tokens,
                        "text": chunk["chunk_body"],
                        "embedding_text": chunk["embedding_text"],
                    }
                )
        else:
            split_chunks = split_section_for_embedding(
                tokenizer,
                chunk_title=record.get("title", ""),
                chunk_body=body_text,
                max_tokens=max_tokens,
                overlap=overlap,
                text_prefix=text_prefix,
            )
            for chunk_index, chunk in enumerate(split_chunks):
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
                        "chunk_count": len(split_chunks),
                        "section_index": 0,
                        "source_body_token_count": source_token_count,
                        "chunk_token_count": chunk["chunk_token_count"],
                        "chunk_mode": chunk_mode,
                        "was_chunked": len(split_chunks) > 1,
                        "exceeds_embedding_limit": chunk["chunk_token_count"] > max_tokens,
                        "section_title": chunk["chunk_title"],
                        "section_chunk_index": chunk["section_chunk_index"],
                        "section_chunk_count": chunk["section_chunk_count"],
                        "chunk_title": chunk["chunk_title"],
                        "chunk_body": chunk["chunk_body"],
                        "text": chunk["chunk_body"],
                        "embedding_text": chunk["embedding_text"],
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
        overlap=args.overlap,
        tokenizer=embedder.tokenizer,
        text_prefix=embedder.passage_prefix,
    )
    if not chunk_records:
        raise RuntimeError("No body_text rows were found to embed.")

    texts = [record["embedding_text"] for record in chunk_records]
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
