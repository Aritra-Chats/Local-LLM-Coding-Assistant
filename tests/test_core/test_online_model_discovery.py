"""Stub tests for core/online_model_discovery.py."""
def test_degradation():
    from core.online_model_discovery import OnlineModelDiscoveryEngine
    import unittest.mock as mock
    router = mock.MagicMock()
    router.select_coding_model.return_value = "codellama:13b"
    engine = OnlineModelDiscoveryEngine(ollama_cloud_client=None, local_router=router)
    st = {"routing_domain": "debugging", "complexity": "medium"}
    engine.discover(st)
    assert "provider" in st["selected_model"]
