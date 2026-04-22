from __future__ import annotations

import os
import sys
from pathlib import Path


def _is_colab_runtime() -> bool:
    return "google.colab" in sys.modules or bool(os.getenv("COLAB_RELEASE_TAG"))


def _looks_like_project_root(path: Path) -> bool:
    return (path / "src").exists() and (path / "data_prep").exists()


def _resolve_project_root() -> Path:
    env_project_root = os.getenv("RAG_PROJECT_ROOT")
    if env_project_root:
        return Path(env_project_root).expanduser().resolve()

    file_based_root = Path(__file__).resolve().parent.parent.parent
    if _looks_like_project_root(file_based_root):
        return file_based_root

    if _is_colab_runtime():
        colab_candidates = [
            Path.cwd(),
            Path("/content/Rag"),
            Path("/content/drive/MyDrive/Rag"),
            Path("/content/drive/MyDrive/RAG_disease_improvement"),
        ]
        for candidate in colab_candidates:
            candidate = candidate.expanduser().resolve()
            if _looks_like_project_root(candidate):
                return candidate

    return file_based_root


def _resolve_data_root(project_root: Path) -> Path:
    env_data_root = os.getenv("RAG_DATA_ROOT")
    if env_data_root:
        return Path(env_data_root).expanduser().resolve()

    default_data_root = project_root / "Data"
    if default_data_root.exists() or not _is_colab_runtime():
        return default_data_root

    colab_data_candidates = [
        project_root / "Data",
        Path("/content/Data"),
        Path("/content/drive/MyDrive/Data"),
        Path("/content/drive/MyDrive/Rag/Data"),
        Path("/content/drive/MyDrive/RAG_disease_improvement/Data"),
    ]
    for candidate in colab_data_candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.exists():
            return candidate

    return default_data_root


def _get_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _get_csv_env(name: str, default: str) -> list[str]:
    raw_value = os.getenv(name, default)
    return [item.strip() for item in raw_value.split(",") if item.strip()]


IS_COLAB = _is_colab_runtime()
PROJECT_ROOT = _resolve_project_root()
DATA_ROOT = _resolve_data_root(PROJECT_ROOT)

DATA_PROCESSED_DIR = DATA_ROOT / "processed"
VECTORSTORE_DIR = DATA_ROOT / "vectorstore"

HEALTH_AZ_URL = "https://tamanhhospital.vn/benh-hoc-a-z/"
HEALTH_INDEX_HTML = DATA_PROCESSED_DIR / "health_disease_index.html"
HEALTH_LINKS_CSV = DATA_PROCESSED_DIR / "health_disease_links.csv"
HEALTH_CONTENT_JSONL = DATA_PROCESSED_DIR / "health_disease_content.jsonl"
HEALTH_HTML_DIR = DATA_PROCESSED_DIR / "health_html"

HEALTH_CHUNKS_JSONL = DATA_PROCESSED_DIR / "health_disease_chunks.jsonl"
HEALTH_VECTORSTORE_EMBEDDINGS = VECTORSTORE_DIR / "health_disease_e5_small_embeddings.npy"
HEALTH_METADATA_JSONL = VECTORSTORE_DIR / "health_disease_e5_small_metadata.jsonl"

CRAWLER_RETRY_MAX = 3
CRAWLER_RETRY_BASE_DELAY = 3.0

SCRAPER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

HEALTH_STOP_MARKERS = {
    "BÀI VIẾT LIÊN QUAN",
    "CÓ THỂ BẠN QUAN TÂM",
    "BÀI VIẾT CÙNG CHỦ ĐỀ",
    "ĐẶT LỊCH HẸN",
    "ĐĂNG KÝ NHẬN TIN",
    "ĐỐI TÁC BẢO HIỂM",
    "XEM THÊM",
}

SCRAPER_SKIP_EXACT = {
    "Mục lục",
    "ĐẶT LỊCH HẸN",
    "XEM HỒ SƠ",
    "Bệnh viện Đa khoa Tâm Anh",
    "Bệnh viện Đa khoa Tâm Anh TP.HCM",
    "Bệnh viện Đa khoa Tâm Anh Hà Nội",
}

E5_MODEL_NAME = "intfloat/multilingual-e5-small"
E5_MAX_TOKENS = 512
E5_OVERLAP = 64
E5_BATCH_SIZE = 32
E5_PASSAGE_PREFIX = "passage: "
E5_QUERY_PREFIX = "query: "

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash").strip()
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_TEMPERATURE = 0.2
GEMINI_API_TIMEOUT = 60
GEMINI_RETRY_MAX_ATTEMPTS = 4
GEMINI_RETRY_BASE_DELAY = 2.0

BACKEND_HOST = os.getenv("RAG_BACKEND_HOST", "0.0.0.0").strip() or "0.0.0.0"
BACKEND_PORT = _get_int_env("RAG_BACKEND_PORT", 8000)
BACKEND_CORS_ORIGINS = _get_csv_env(
    "RAG_BACKEND_CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
)

HYBRID_VECTOR_TOP_K = 20
HYBRID_BM25_TOP_K = 20
HYBRID_RERANK_TOP_K = 8

CROSS_ENCODER_MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL_NAME", "BAAI/bge-reranker-v2-m3").strip()
CROSS_ENCODER_MAX_TOKENS = 512
CROSS_ENCODER_BATCH_SIZE = 8

TITLE_CHUNK_MAX_HEADING_WORDS = 24
TITLE_CHUNK_MAX_HEADING_CHARS = 160
