from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_ROOT = PROJECT_ROOT / "Data"
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

TITLE_CHUNK_MAX_HEADING_WORDS = 24
TITLE_CHUNK_MAX_HEADING_CHARS = 160
