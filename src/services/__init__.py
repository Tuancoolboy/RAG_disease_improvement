"""Shared RAG services for CLI and web backends."""

from .rag_service import RAGService, get_default_rag_service

__all__ = ["RAGService", "get_default_rag_service"]
