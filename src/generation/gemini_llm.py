from __future__ import annotations

import json
import time
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from src.config import (
    GEMINI_API_BASE_URL,
    GEMINI_API_KEY,
    GEMINI_API_TIMEOUT,
    GEMINI_MODEL_NAME,
    GEMINI_RETRY_BASE_DELAY,
    GEMINI_RETRY_MAX_ATTEMPTS,
    GEMINI_TEMPERATURE,
)
from src.retrieval import SearchResult


class GeminiLLM:
    def __init__(
        self,
        *,
        api_key: str = GEMINI_API_KEY,
        model_name: str = GEMINI_MODEL_NAME,
        temperature: float = GEMINI_TEMPERATURE,
        api_base_url: str = GEMINI_API_BASE_URL,
        timeout: int = GEMINI_API_TIMEOUT,
        max_attempts: int = GEMINI_RETRY_MAX_ATTEMPTS,
        retry_base_delay: float = GEMINI_RETRY_BASE_DELAY,
    ) -> None:
        self.api_key = api_key.strip()
        self.model_name = model_name
        self.temperature = temperature
        self.api_base_url = api_base_url.rstrip("/")
        self.timeout = timeout
        self.max_attempts = max(1, max_attempts)
        self.retry_base_delay = max(0.0, retry_base_delay)

    def generate_answer(self, question: str, retrieved_chunks: Iterable[SearchResult]) -> str:
        if not self.api_key:
            raise ValueError(
                "Missing Gemini API key. Set GEMINI_API_KEY in environment or src/config/config.py."
            )

        prompt = self._build_prompt(question, list(retrieved_chunks))
        url = f"{self.api_base_url}/{quote(self.model_name)}:generateContent?key={quote(self.api_key)}"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
            },
        }
        request = Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )

        raw = self._request_with_retry(request)

        candidates = raw.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini returned no candidates: {raw}")

        parts = candidates[0].get("content", {}).get("parts", [])
        texts = [part.get("text", "") for part in parts if part.get("text")]
        return "\n".join(texts).strip()

    def _build_prompt(self, question: str, chunks: list[SearchResult]) -> str:
        context_parts: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            context_parts.append(
                "\n".join(
                    [
                        f"[{idx}] chunk_id: {chunk.chunk_id}",
                        f"title: {chunk.chunk_title}",
                        f"source_title: {chunk.source_title}",
                        f"source_url: {chunk.source_url}",
                        f"content: {chunk.chunk_body}",
                    ]
                )
            )

        context = "\n\n".join(context_parts)
        return (
            "Bạn là trợ lý RAG y khoa tiếng Việt.\n"
            "Chỉ trả lời dựa trên context được cung cấp.\n"
            "Nếu context không đủ, nói rõ là không đủ thông tin.\n"
            "Ưu tiên câu trả lời ngắn gọn, chính xác, có trích dẫn [số] theo chunk.\n\n"
            f"Câu hỏi: {question}\n\n"
            f"Context:\n{context}\n\n"
            "Hãy trả lời bằng tiếng Việt."
        )

    def _request_with_retry(self, request: Request) -> dict:
        retryable_status_codes = {429, 500, 503, 504}
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace").strip()
                if exc.code not in retryable_status_codes or attempt == self.max_attempts:
                    message = (
                        f"Gemini request failed with HTTP {exc.code}."
                        f"{f' Response: {error_body}' if error_body else ''}"
                    )
                    raise RuntimeError(message) from exc
                last_error = RuntimeError(
                    f"Temporary Gemini error HTTP {exc.code}"
                    f"{f' | {error_body}' if error_body else ''}"
                )
            except URLError as exc:
                if attempt == self.max_attempts:
                    raise RuntimeError(f"Gemini request failed: {exc}") from exc
                last_error = exc

            delay = self.retry_base_delay * (2 ** (attempt - 1))
            if delay > 0:
                time.sleep(delay)

        if last_error is not None:
            raise RuntimeError(f"Gemini request failed after retries: {last_error}") from last_error
        raise RuntimeError("Gemini request failed after retries.")
