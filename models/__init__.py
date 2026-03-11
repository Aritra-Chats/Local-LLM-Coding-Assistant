"""models — Ollama client, embedding client, and model registry."""
from models.ollama_client import OllamaClient
from models.embedding_client import EmbeddingClient
from models.model_registry import ModelRegistry, ModelEntry

__all__ = ["OllamaClient", "EmbeddingClient", "ModelRegistry", "ModelEntry"]
