"""Run a service's model calls one at a time.

`MCPServer` runs synchronous tools in worker threads, so concurrent requests reach
the model concurrently. The model code is not thread-safe: NTv2's remote rotary
embedding, for example, rebuilds a shared cos/sin cache on every forward pass, and
DNABERT-2's per-layer hook state is shared too; concurrent calls then fail or hang.
On one GPU, parallel forward passes of the same model also gain little, since
batching (a list of sequences in one call) is what uses the GPU efficiently.

The services also share the GPU with each other. PyTorch keeps memory freed after a
call in its cache, where other processes can't use it: one long-sequence batch could
leave several GB reserved. After each call, cached memory above
`DNALM_CUDA_CACHE_LIMIT_MB` (default 1024) is returned to the GPU, and running out of
GPU memory becomes a readable error instead of "Error executing tool".
"""

from __future__ import annotations

import functools
import os
import threading
from collections.abc import Callable
from typing import TypeVar

import torch

F = TypeVar("F", bound=Callable)

_MODEL_LOCK = threading.Lock()
_CACHE_LIMIT_BYTES = int(os.environ.get("DNALM_CUDA_CACHE_LIMIT_MB", "1024")) * 2**20


def one_call_at_a_time(fn: F) -> F:
    """Wrap a tool that runs the model so that only one such call runs per process."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _MODEL_LOCK:
            try:
                return fn(*args, **kwargs)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                raise ValueError(
                    "Out of GPU memory. The GPU is shared with the other model services; "
                    "send fewer or shorter sequences per call, or try again later."
                ) from exc
            finally:
                release_cached_gpu_memory()

    return wrapper  # type: ignore[return-value]


def release_cached_gpu_memory(limit_bytes: int | None = None) -> bool:
    """Return PyTorch's cached (reserved but unused) GPU memory above `limit_bytes` to the GPU."""
    if not torch.cuda.is_available():
        return False
    limit = _CACHE_LIMIT_BYTES if limit_bytes is None else limit_bytes
    if torch.cuda.memory_reserved() - torch.cuda.memory_allocated() <= limit:
        return False
    torch.cuda.empty_cache()
    return True
