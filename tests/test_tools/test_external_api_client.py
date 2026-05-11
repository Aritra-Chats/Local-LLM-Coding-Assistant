"""Stub tests for models/external_api_client.py."""
import os, pytest

def test_known_providers():
    from models.external_api_client import ExternalAPIClient
    for p in ("anthropic", "openai", "google"):
        ExternalAPIClient(p)

def test_unknown_provider():
    from models.external_api_client import ExternalAPIClient
    with pytest.raises(AssertionError):
        ExternalAPIClient("bogus")

def test_unavailable_without_key():
    import unittest.mock as mock
    from models.external_api_client import ExternalAPIClient
    c = ExternalAPIClient("openai")
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    with mock.patch.dict(os.environ, env, clear=True):
        assert c.is_available() is False
