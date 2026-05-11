"""Stub tests for core/task_segregator.py."""
def test_classify_keyword():
    from core.task_segregator import TaskSegregator
    import unittest.mock as mock
    seg = TaskSegregator(mock.MagicMock(), "model")
    st = {"raw_description": "Fix traceback error", "domain": "other"}
    seg.classify(st)
    assert st["routing_domain"] == "debugging"

def test_segregate_fallback():
    from core.task_segregator import TaskSegregator
    import unittest.mock as mock
    client = mock.MagicMock()
    client.generate.side_effect = RuntimeError("no model")
    seg = TaskSegregator(client, "model")
    result = seg.segregate("Do something cool")
    assert len(result) == 1
