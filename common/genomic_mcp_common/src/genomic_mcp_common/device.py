"""Device/dtype selection from an env var.

Each service keeps its own family-prefixed env var names (e.g. ``NTV3_DEVICE``,
``NTV2_DTYPE``) so a single host can run several of these containers with
independent device/dtype overrides; this module just holds the shared logic.
"""

from __future__ import annotations

import os

import torch

_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def select_device(env_var: str) -> str:
    """Return the device override from `env_var`, or auto-detect cuda/mps/cpu."""
    override = os.environ.get(env_var)
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def select_dtype(env_var: str, default: torch.dtype = torch.float32) -> torch.dtype:
    """Return the torch dtype named by `env_var` (float32/bfloat16/float16, case-insensitive), or `default`."""
    override = os.environ.get(env_var, "").strip().lower()
    if override:
        if override not in _DTYPE_MAP:
            raise ValueError(f"Invalid {env_var}={override!r}; choose one of {sorted(_DTYPE_MAP)}.")
        return _DTYPE_MAP[override]
    return default
