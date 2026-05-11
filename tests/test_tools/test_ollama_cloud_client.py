"""Stub tests for models/ollama_cloud_client.py."""
import os, pytest

def test_requires_api_key():
    import unittest.mock as mock
    from models.ollama_cloud_client import OllamaCloudClient
    env = {k: v for k, v in os.environ.items() if k != "OLLAMA_API_KEY"}
    with mock.patch.dict(os.environ, env, clear=True):
        with pytest.raises(ValueError):
            OllamaCloudClient(api_key="")

def test_strip_cloud_suffix():
    from models.ollama_cloud_client import OllamaCloudClient
    assert OllamaCloudClient._strip_cloud_suffix("llama3:8b:cloud") == "llama3:8b"
    assert OllamaCloudClient._strip_cloud_suffix("llama3:8b") == "llama3:8b"

def test_auth_header():
    from models.ollama_cloud_client import OllamaCloudClient
    c = OllamaCloudClient(api_key="mykey")
    assert "mykey" in c._auth_header()["Authorization"]
