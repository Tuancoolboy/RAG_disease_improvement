from __future__ import annotations

import re
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

from src.config import TITLE_CHUNK_MAX_HEADING_CHARS, TITLE_CHUNK_MAX_HEADING_WORDS


@dataclass(slots=True)
class SectionChunk:
    section_title: str
    text: str
    token_count: int
    block_count: int
    section_index: int


class TitleChunker:
    """Split body_text into chunks using in-article section headings."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_heading_words: int = TITLE_CHUNK_MAX_HEADING_WORDS,
        max_heading_chars: int = TITLE_CHUNK_MAX_HEADING_CHARS,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_heading_words = max_heading_words
        self.max_heading_chars = max_heading_chars

    def count_tokens(self, text: str) -> int:
        return len(
            self.tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
        )

    def split(self, source_title: str, body_text: str) -> list[SectionChunk]:
        blocks = [block.strip() for block in body_text.split("\n\n") if block.strip()]
        if not blocks:
            return []

        sections: list[tuple[str, list[str]]] = []
        current_title = source_title.strip()
        current_blocks: list[str] = []

        for idx, block in enumerate(blocks):
            next_block = blocks[idx + 1] if idx + 1 < len(blocks) else ""
            if self._is_heading(block, next_block):
                if current_blocks:
                    sections.append((current_title, current_blocks))
                current_title = block
                current_blocks = [block]
            else:
                current_blocks.append(block)

        if current_blocks:
            sections.append((current_title, current_blocks))

        return [
            SectionChunk(
                section_title=section_title,
                text=self._build_chunk_text(source_title, section_title, section_blocks),
                token_count=self.count_tokens(
                    self._build_chunk_text(source_title, section_title, section_blocks)
                ),
                block_count=len(section_blocks),
                section_index=section_index,
            )
            for section_index, (section_title, section_blocks) in enumerate(sections)
        ]

    def _build_chunk_text(self, source_title: str, section_title: str, section_blocks: list[str]) -> str:
        if not section_blocks:
            return source_title.strip()

        if section_title == source_title:
            return "\n\n".join([source_title.strip(), *section_blocks]).strip()

        body_blocks = section_blocks[1:] if section_blocks and section_blocks[0] == section_title else section_blocks
        parts = [source_title.strip(), section_title.strip(), *body_blocks]
        return "\n\n".join(part for part in parts if part).strip()

    def _is_heading(self, block: str, next_block: str) -> bool:
        text = block.strip()
        if not text:
            return False
        if text.startswith("- "):
            return False
        if len(text) > self.max_heading_chars:
            return False

        words = text.split()
        if len(words) > self.max_heading_words:
            return False

        if text.lower().startswith("nguồn tham khảo"):
            return True

        if re.match(r"^\d+\.\s+", text):
            return True

        if text.endswith((".", "!", ",")):
            return False

        if not next_block:
            return False

        next_words = next_block.split()
        if next_block.startswith("- "):
            return True

        if len(next_words) > len(words):
            return True

        return False
