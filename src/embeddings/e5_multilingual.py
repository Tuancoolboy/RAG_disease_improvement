from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.config import E5_MAX_TOKENS, E5_MODEL_NAME, E5_PASSAGE_PREFIX


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class E5MultilingualEmbedder:
    """Wrapper around intfloat/multilingual-e5-small."""

    def __init__(
        self,
        model_name: str = E5_MODEL_NAME,
        *,
        device: str | None = None,
        max_length: int = E5_MAX_TOKENS,
        passage_prefix: str = E5_PASSAGE_PREFIX,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.passage_prefix = passage_prefix
        self.device = device or self._pick_device()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def _pick_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def embed_passages(self, texts: Iterable[str], *, batch_size: int = 32) -> np.ndarray:
        all_embeddings: list[np.ndarray] = []
        batch: list[str] = []

        for text in texts:
            batch.append(f"{self.passage_prefix}{text.strip()}")
            if len(batch) >= batch_size:
                all_embeddings.append(self._embed_batch(batch))
                batch = []

        if batch:
            all_embeddings.append(self._embed_batch(batch))

        if not all_embeddings:
            return np.empty((0, 384), dtype=np.float32)

        return np.vstack(all_embeddings).astype(np.float32)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = average_pool(outputs.last_hidden_state, encoded["attention_mask"])
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()
