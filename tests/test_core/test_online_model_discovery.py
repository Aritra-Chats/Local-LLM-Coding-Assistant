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


def test_cloud_selection_uses_cloud_models():
    from core.online_model_discovery import OnlineModelDiscoveryEngine

    class CloudClient:
        def list_models(self):
            return [
                {"name": "small:7b-cloud", "details": {"parameter_size": "7b"}},
                {"name": "large:13b-cloud", "details": {"parameter_size": "13b"}},
            ]

    engine = OnlineModelDiscoveryEngine(ollama_cloud_client=CloudClient(), local_router=None)
    engine._web_search_scores = lambda domain, complexity: {}

    st = {"routing_domain": "debugging", "complexity": "medium"}
    engine.discover(st)

    assert st["selected_model"]["provider"] == "ollama_cloud"
    assert st["selected_model"]["model"] == "large:13b-cloud"
