"""Stub tests for core/api_key_manager.py."""
import os

def test_imports():
    from core.api_key_manager import APIKeyManager, REQUIRED_KEYS
    assert "OLLAMA_API_KEY" in REQUIRED_KEYS

def test_save_key_roundtrip(tmp_path):
    from core.api_key_manager import APIKeyManager
    mgr = APIKeyManager(env_file=tmp_path / ".env")
    mgr.save_key("TEST_SENTINEL_KEY", "secret123")
    assert "TEST_SENTINEL_KEY" in (tmp_path / ".env").read_text()
    os.environ.pop("TEST_SENTINEL_KEY", None)
