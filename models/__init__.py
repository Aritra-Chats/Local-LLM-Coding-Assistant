"""models — Ollama client, embedding client, and model registry."""
from models.ollama_client import OllamaClient
from models.embedding_client import EmbeddingClient
from models.model_registry import ModelRegistry, ModelEntry
from models.inference_engine import InferenceEngine, FAISSIndex
from models.ollama_cloud_client import OllamaCloudClient
from models.external_api_client import ExternalAPIClient

__all__ = [
    "OllamaClient",
    "EmbeddingClient",
    "ModelRegistry",
    "ModelEntry",
    "InferenceEngine",
    "FAISSIndex",
    "OllamaCloudClient",
    "ExternalAPIClient",
]
