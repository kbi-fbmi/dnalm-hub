"""one_call_at_a_time serialises calls across threads and keeps the wrapped signature."""

import inspect
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from dnalm_common import one_call_at_a_time


def test_calls_never_overlap():
    active, peak = [0], [0]
    guard = threading.Lock()

    @one_call_at_a_time
    def tool(i: int, checkpoint: str = "x") -> int:
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.01)
        with guard:
            active[0] -= 1
        return i

    with ThreadPoolExecutor(8) as pool:
        assert list(pool.map(tool, range(16))) == list(range(16))
    assert peak[0] == 1
    assert list(inspect.signature(tool).parameters) == [
        "i",
        "checkpoint",
    ]  # MCP builds the schema from it


def test_out_of_memory_becomes_a_readable_error():
    import pytest
    import torch

    @one_call_at_a_time
    def tool():
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(ValueError, match="Out of GPU memory"):
        tool()
