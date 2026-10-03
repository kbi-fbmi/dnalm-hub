"""Bounded tokenizer/model cache with real eviction.

The original per-service caches (a plain `dict`) never evicted: every distinct
checkpoint ever requested stayed resident in GPU memory for the life of the
process. That was fine with one model family on one server; it isn't once
several families share one GPU. This cache keeps at most `max_entries` loaded
models, evicting the least-recently-used one (and actually releasing its VRAM)
when a new one is loaded past that limit.

With `DNALM_IDLE_UNLOAD_SECONDS` set, entries unused for that long are also
dropped, so a service that nobody calls gives its VRAM back to the others.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable

import torch

logger = logging.getLogger(__name__)


class LRUModelCache:
    """Thread-safe LRU cache for (tokenizer, model) pairs keyed by an opaque string.

    Args:
        max_entries: how many entries to keep resident before evicting the
            least-recently-used one. Configure per service via an env var
            (e.g. `NTV3_MAX_RESIDENT_MODELS`) based on that family's typical
            checkpoint size and the host's VRAM budget.
        idle_unload_seconds: drop entries not used for this long (0 = never).
            Defaults to the `DNALM_IDLE_UNLOAD_SECONDS` env var.
    """

    def __init__(self, max_entries: int = 2, idle_unload_seconds: float | None = None) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1.")
        if idle_unload_seconds is None:
            idle_unload_seconds = float(os.environ.get("DNALM_IDLE_UNLOAD_SECONDS", "0") or 0)
        self._max_entries = max_entries
        self._idle_unload_seconds = idle_unload_seconds
        self._lock = threading.Lock()
        self._data: OrderedDict[str, tuple] = OrderedDict()
        self._last_used: dict[str, float] = {}
        self._reaper: threading.Thread | None = None

    def get_or_load(self, key: str, loader: Callable[[], tuple]) -> tuple:
        """Return the cached value for `key`, loading it via `loader()` on a miss."""
        with self._lock:
            self._last_used[key] = time.monotonic()
            cached = self._data.get(key)
            if cached is not None:
                self._data.move_to_end(key)
                return cached
            value = loader()
            self._data[key] = value
            self._evict_excess()
            self._start_reaper()
            return value

    def _evict_excess(self) -> None:
        """Caller must hold `_lock`. Drop least-recently-used entries past `max_entries`."""
        while len(self._data) > self._max_entries:
            key, evicted = self._data.popitem(last=False)
            self._last_used.pop(key, None)
            _release(evicted)

    def evict_idle(self) -> list[str]:
        """Drop entries unused for `idle_unload_seconds`; return their keys.

        Unlike LRU eviction this does not move the model to the CPU: a long
        request may still be using it, and its VRAM is freed once that request
        drops its references.
        """
        if self._idle_unload_seconds <= 0:
            return []
        cutoff = time.monotonic() - self._idle_unload_seconds
        with self._lock:
            idle = [k for k in self._data if self._last_used.get(k, 0.0) < cutoff]
            for key in idle:
                del self._data[key]
                self._last_used.pop(key, None)
        if idle:
            logger.info("Unloaded idle model(s): %s", ", ".join(idle))
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return idle

    def _start_reaper(self) -> None:
        """Caller must hold `_lock`. Start the idle-unload thread once, if enabled."""
        if self._idle_unload_seconds <= 0 or self._reaper is not None:
            return
        interval = min(60.0, max(1.0, self._idle_unload_seconds / 4))

        def run() -> None:
            while True:
                time.sleep(interval)
                try:
                    self.evict_idle()
                except Exception:
                    logger.exception("Idle model unload failed")

        self._reaper = threading.Thread(target=run, name="idle-model-unload", daemon=True)
        self._reaper.start()

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
