"""Bounded tokenizer/model cache with real eviction.

The original per-service caches (a plain `dict`) never evicted: every distinct
checkpoint ever requested stayed resident in GPU memory for the life of the
process. That was fine with one model family on one server; it isn't once
several families share one GPU. This cache keeps at most `max_entries` loaded
models, evicting the least-recently-used one (and actually releasing its VRAM)
when a new one is loaded past that limit.
"""

from __future__ import annotations

import gc
import threading
from collections import OrderedDict
from collections.abc import Callable

import torch


class LRUModelCache:
    """Thread-safe LRU cache for (tokenizer, model) pairs keyed by an opaque string.

    Args:
        max_entries: how many entries to keep resident before evicting the
            least-recently-used one. Configure per service via an env var
            (e.g. `NTV3_MAX_RESIDENT_MODELS`) based on that family's typical
            checkpoint size and the host's VRAM budget.
    """

    def __init__(self, max_entries: int = 2) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1.")
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._data: OrderedDict[str, tuple] = OrderedDict()

    def get_or_load(self, key: str, loader: Callable[[], tuple]) -> tuple:
        """Return the cached value for `key`, loading it via `loader()` on a miss."""
        with self._lock:
            cached = self._data.get(key)
            if cached is not None:
                self._data.move_to_end(key)
                return cached
            value = loader()
            self._data[key] = value
            self._evict_excess()
            return value

    def _evict_excess(self) -> None:
        """Caller must hold `_lock`. Drop least-recently-used entries past `max_entries`."""
        while len(self._data) > self._max_entries:
            _, evicted = self._data.popitem(last=False)
            _release(evicted)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data


def _release(value: tuple) -> None:
    """Best-effort: move a model off its device and free the VRAM it held."""
    try:
        _tokenizer, model = value
        if hasattr(model, "to"):
            model.to("cpu")
        del model
    except Exception:  # noqa: BLE001, S110 - eviction must never crash the caller's request
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
