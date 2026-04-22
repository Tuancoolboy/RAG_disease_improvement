from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase


@dataclass(slots=True)
class TextChunk:
    text: str
    token_count: int
    start_token: int
    end_token: int


class TokenChunker:
    """Chunk text by model tokens while preserving original character spans."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_input_tokens: int = 512,
        overlap: int = 64,
        text_prefix: str = "passage: ",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.overlap = overlap
        self.text_prefix = text_prefix

        prefix_tokens = self.tokenizer(
            text_prefix,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        self.max_body_tokens = max(1, max_input_tokens - len(prefix_tokens) - special_tokens)

        if overlap >= self.max_body_tokens:
            raise ValueError(
                f"overlap={overlap} must be smaller than usable body tokens={self.max_body_tokens}"
            )

    def count_tokens(self, text: str) -> int:
        return len(
            self.tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
        )

    def split(self, text: str) -> list[TextChunk]:
        text = text.strip()
        if not text:
            return []

        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        total_tokens = len(input_ids)

        if total_tokens <= self.max_body_tokens:
            return [
                TextChunk(
                    text=text,
                    token_count=total_tokens,
                    start_token=0,
                    end_token=total_tokens,
                )
            ]

        chunks: list[TextChunk] = []
        step = self.max_body_tokens - self.overlap
        start = 0

        while start < total_tokens:
            end = min(start + self.max_body_tokens, total_tokens)
            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]
            chunk_text = text[char_start:char_end].strip()

            if chunk_text:
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        token_count=end - start,
                        start_token=start,
                        end_token=end,
                    )
                )

            if end >= total_tokens:
                break

            start += step

        return chunks
