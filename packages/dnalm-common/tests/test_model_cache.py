"""Tests LRUModelCache's eviction bookkeeping with cheap fake "models" (no torch/GPU needed)."""

import pytest
from dnalm_common.model_cache import LRUModelCache


class _FakeModel:
    """Stands in for a real model; only needs a `.to()` method for eviction to call."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.device = "cuda"

    def to(self, device: str) -> "_FakeModel":
        self.device = device
        return self


def _loader(name: str):
    return (f"tokenizer-{name}", _FakeModel(name))


def test_cache_returns_same_object_on_hit():
    cache = LRUModelCache(max_entries=2)
    loads = []

    def loader():
        loads.append(1)
        return _loader("a")

    first = cache.get_or_load("a", loader)
    second = cache.get_or_load("a", loader)
    assert first is second
    assert len(loads) == 1  # loader only called once


def test_evicts_least_recently_used_past_capacity():
    cache = LRUModelCache(max_entries=2)
    cache.get_or_load("a", lambda: _loader("a"))
    cache.get_or_load("b", lambda: _loader("b"))
    cache.get_or_load("c", lambda: _loader("c"))  # should evict "a"

    assert "a" not in cache
    assert "b" in cache
    assert "c" in cache
    assert len(cache) == 2


def test_accessing_an_entry_protects_it_from_eviction():
    cache = LRUModelCache(max_entries=2)
    cache.get_or_load("a", lambda: _loader("a"))
    cache.get_or_load("b", lambda: _loader("b"))
    cache.get_or_load("a", lambda: _loader("a"))  # touch "a" -> "b" becomes LRU
    cache.get_or_load("c", lambda: _loader("c"))  # should evict "b", not "a"

    assert "a" in cache
    assert "b" not in cache
    assert "c" in cache


def test_eviction_moves_model_off_device():
    cache = LRUModelCache(max_entries=1)
    _tok_a, model_a = cache.get_or_load("a", lambda: _loader("a"))
    cache.get_or_load("b", lambda: _loader("b"))  # evicts "a"

    assert model_a.device == "cpu"


def test_rejects_non_positive_capacity():
    with pytest.raises(ValueError):
        LRUModelCache(max_entries=0)


def test_idle_entries_are_dropped(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("dnalm_common.model_cache.time.monotonic", lambda: clock[0])
    cache = LRUModelCache(max_entries=2, idle_unload_seconds=60)
    cache._start_reaper = lambda: None  # driven by hand below
    a = cache.get_or_load("a", lambda: _loader("a"))
    clock[0] += 50
    cache.get_or_load("b", lambda: _loader("b"))
    clock[0] += 20  # a idle 70 s, b idle 20 s
    assert cache.evict_idle() == ["a"]
    assert "a" not in cache and "b" in cache
    assert a[1].device == "cuda"  # not moved: an in-flight request may still use it


def test_idle_unload_off_by_default(monkeypatch):
    monkeypatch.delenv("DNALM_IDLE_UNLOAD_SECONDS", raising=False)
    cache = LRUModelCache(max_entries=2)
    cache.get_or_load("a", lambda: _loader("a"))
    assert cache.evict_idle() == []
    assert cache._reaper is None


def test_idle_unload_from_env(monkeypatch):
    monkeypatch.setenv("DNALM_IDLE_UNLOAD_SECONDS", "600")
    assert LRUModelCache()._idle_unload_seconds == 600
