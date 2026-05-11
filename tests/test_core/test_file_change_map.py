"""Stub tests for core/file_change_map.py."""
import time

def test_record_and_resolve():
    from core.file_change_map import FileChangeMap, FileChangeEvent
    fcm = FileChangeMap()
    ev = FileChangeEvent("src/foo.py", "/p/src/foo.py", "create", "s1", "coding", int(time.time()*1000))
    fcm.record(ev)
    assert fcm.resolve("src/foo.py") == "/p/src/foo.py"
    assert fcm.resolve("nonexistent") is None

def test_persist_roundtrip(tmp_path):
    from core.file_change_map import FileChangeMap, FileChangeEvent
    fcm = FileChangeMap()
    fcm.record(FileChangeEvent("a.py", "/p/a.py", "create", "s1", "ag", 123456))
    out = tmp_path / "changes.json"
    fcm.save(out)
    loaded = FileChangeMap.load(out)
    assert loaded.resolve("a.py") == "/p/a.py"
