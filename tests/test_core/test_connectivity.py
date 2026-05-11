"""Stub tests for core/connectivity.py."""
def test_connectivity_checker_imports():
    from core.connectivity import ConnectivityChecker
    assert hasattr(ConnectivityChecker, "check")
    assert hasattr(ConnectivityChecker, "check_ollama_cloud")

def test_check_returns_bool():
    from core.connectivity import ConnectivityChecker
    import socket, unittest.mock as mock
    with mock.patch("socket.socket") as ms:
        ms.return_value.connect.side_effect = OSError("refused")
        assert isinstance(ConnectivityChecker.check(), bool)
