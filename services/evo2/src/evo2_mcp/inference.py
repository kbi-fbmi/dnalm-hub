"""Model loading and inference helpers for Evo 2 (community transformers port).

Evo 2 (Arc Institute) is a causal StripedHyena-2 DNA language model: Hyena
convolution blocks (short/medium FIR, long implicit IIR) striped with a few
grouped-query attention blocks, one token per nucleotide (raw UTF-8 bytes,
``"ACGT" -> [65, 67, 71, 84]``, no special tokens). This service runs the
pure-PyTorch port https://github.com/Aquiles-ai/Evo2-transformers (no Vortex,
Transformer Engine, FP8 or custom kernels), loaded via ``trust_remote_code``.

Quirks of the port this module works around (all verified on the pinned revision):

- **Never pass an attention mask to a full-sequence forward.** With a mask the
  port's attention calls SDPA with ``is_causal=False``, i.e. bidirectional
  attention, so position t would see the future. Without a mask attention is
  causal, and right padding cannot change any real position (every mixer is
  causal), so no mask is needed. Decode steps (one query) do need the mask,
  see ``generate``. For the same reason ``model.generate`` is not used: it
  always builds a mask for the prefill.
- **Keep the fp32 tensors fp32.** The checkpoint stores the Hyena IIR poles and
  residues and the RoPE ``inv_freq`` in fp32 and everything else in bf16, but
  ``from_pretrained(dtype=bfloat16)`` casts all of them. ``load`` copies those
  tensors back from the safetensors file in fp32 (the port computes its long
  filters and FFTs in fp32 anyway).
- **Memory on long inputs.** The port builds each IIR long filter through an
  ``(channels, 16, L)`` fp32 temporary and runs every FFT convolution over all
  4096 channels at once; ``_patch_memory_hogs`` swaps in a term-by-term filter and
  a channel-chunked FFT (same math), which roughly halves peak VRAM at 16 kb and
  makes 64 kb fit. Lengths are still capped by ``EVO2_MAX_SEQ_LEN``, far below
  the 1M native context.
- **fp32 LM head.** The port's bf16 head rounds logits to 0.5-steps (~0.25
  nats/base); scoring and sampling use an fp32 head (``lm_logits``) instead.

Each sequence runs as its own forward pass: a batch returns exactly what the
same sequences give one at a time, and peak memory is bounded by the length cap
rather than by the batch size.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import types

import numpy as np
import torch
from dnalm_common import (
    LRUModelCache,
    score_snp_whole_sequence,
    select_device,
    select_dtype,
    validate_sequence,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from .registry import ResolvedModel

DEVICE = select_device("EVO2_DEVICE")
# bf16 by default: the 7B is ~14 GB in bf16 (28 GB in fp32 would not leave room for activations).
DTYPE = select_dtype("EVO2_DTYPE", default=torch.bfloat16)

_MAX_RESIDENT_MODELS = int(os.environ.get("EVO2_MAX_RESIDENT_MODELS", "1"))
# Service-wide length cap (bases). Measured on a 32 GB RTX PRO 4500 with the 7B in
# bf16: see README ("Limits") for peak VRAM per length.
MAX_SEQ_LEN = int(os.environ.get("EVO2_MAX_SEQ_LEN", "32768"))
MAX_NEW_TOKENS = int(os.environ.get("EVO2_MAX_NEW_TOKENS", "2048"))
# per_token embeddings are (bases x hidden) floats: 4096 per base for the 7B.
MAX_OUTPUT_VALUES = int(os.environ.get("EVO2_MAX_OUTPUT_VALUES", "16777216"))

# Channels per FFT convolution call (see chunked_fft_conv).
_FFT_CHANNEL_CHUNK = int(os.environ.get("EVO2_FFT_CHANNEL_CHUNK", "512"))

_cache = LRUModelCache(max_entries=_MAX_RESIDENT_MODELS)
# One forward pass at a time: a single long 7B pass can take most of the VRAM.
_gpu_lock = threading.Lock()

NUCLEOTIDE_TOKENS = {nt: ord(nt) for nt in "ACGT"}  # byte-level tokenizer: id = ASCII code
_BLOCK_LAYER_RE = re.compile(r"^blocks\.(\d+)\.mlp\.l3$")
# Relative depth of blocks.28 in the 32-block 7B, used for unregistered checkpoints.
_RECOMMENDED_DEPTH = 28 / 32


def _fp32_checkpoint_tensors(m: ResolvedModel) -> dict[str, torch.Tensor]:
    """Every tensor the checkpoint stores in fp32 (poles, residues, inv_freq), read in fp32."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    from safetensors import safe_open

    try:
        index = hf_hub_download(m.repo_id, "model.safetensors.index.json", revision=m.revision)
        with open(index) as fh:
            shards = sorted(set(json.load(fh)["weight_map"].values()))
    except EntryNotFoundError:
        shards = ["model.safetensors"]
    tensors = {}
    for shard in shards:
        path = hf_hub_download(m.repo_id, shard, revision=m.revision)
        with safe_open(path, framework="pt") as f:
            for key in f.keys():  # noqa: SIM118 - safe_open has no __iter__
                if f.get_slice(key).get_dtype() == "F32":
                    tensors[key] = f.get_tensor(key)
    return tensors


def _restore_fp32(model, tensors: dict[str, torch.Tensor]) -> int:
    """Replace the (down-cast) parameters/buffers named in `tensors` with their fp32 values."""
    state = dict(model.named_parameters()) | dict(model.named_buffers())
    restored = 0
    with torch.no_grad():
        for name, value in tensors.items():
            target = state.get(name)
            if target is None or target.dtype == torch.float32:
                continue
            target.data = value.to(device=target.device, dtype=torch.float32)
            restored += 1
    return restored


def _iir_filter_low_memory(self, L: int, device, dtype: torch.dtype) -> torch.Tensor:
    """Drop-in for the port's `Evo2HyenaMixer._iir_filter`, without the `(H, state, L)` temporary.

    The port evaluates `h[c, t] = sum_s residues[c, s] * exp(log_poles[c, s] * t)` by
    materializing all `state_size` (16) terms at once in fp32: 256 KiB per base for
    the 7B, 8 GiB at 32 kb, which made the filter the largest allocation of a long
    forward pass. Summing one term at a time gives the same filter (fp32
    accumulation, different summation order) in `(H, L)` memory.
    """
    t = torch.arange(L, device=device, dtype=torch.float32)
    log_poles = self.log_poles.to(torch.float32)[..., 0]  # (H, S)
    residues = self.residues.to(torch.float32)  # (H, S)
    h = torch.zeros(log_poles.shape[0], L, device=device, dtype=torch.float32)
    for s in range(log_poles.shape[1]):
        h.add_(residues[:, s, None] * torch.exp(log_poles[:, s, None] * t))
    return h[None].to(dtype)


def chunked_fft_conv(fft_conv, chunk: int):
    """Wrap the port's `_fft_conv(u, k, d_bias)` to run `chunk` channels at a time.

    Every channel is an independent convolution, so the result is the same; only
    the fp32/complex FFT temporaries (~160 KiB per base for 4096 channels) shrink
    to `chunk / channels` of that.
    """

    def wrapper(u: torch.Tensor, k: torch.Tensor, d_bias: torch.Tensor) -> torch.Tensor:
        channels = u.shape[1]
        if channels <= chunk:
            return fft_conv(u, k, d_bias)
        k = k.reshape(channels, k.shape[-1])
        out = torch.empty_like(u)
        for s in range(0, channels, chunk):
            out[:, s : s + chunk] = fft_conv(
                u[:, s : s + chunk], k[None, s : s + chunk], d_bias[s : s + chunk]
            )
        return out

    wrapper.evo2_mcp_chunked = True
    return wrapper


def _patch_memory_hogs(model) -> None:
    """Swap in the low-memory IIR filter and channel-chunked FFT convolution (same math)."""
    for module in model.modules():
        if hasattr(module, "_iir_filter") and hasattr(module, "log_poles"):
            module._iir_filter = types.MethodType(_iir_filter_low_memory, module)
            remote = sys.modules[type(module).__module__]
            fft_conv = getattr(remote, "_fft_conv", None)
            if fft_conv is not None and not getattr(fft_conv, "evo2_mcp_chunked", False):
                remote._fft_conv = chunked_fft_conv(fft_conv, _FFT_CHANNEL_CHUNK)


def load(m: ResolvedModel):
    """Load (and LRU-cache, `EVO2_MAX_RESIDENT_MODELS`, default 1) the tokenizer + model."""

    def _load() -> tuple:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                m.repo_id, revision=m.revision, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                m.repo_id, revision=m.revision, trust_remote_code=True, dtype=DTYPE
            )
            if DTYPE != torch.float32:
                _restore_fp32(model, _fp32_checkpoint_tensors(m))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Evo 2 model '{m.repo_id}' from HuggingFace: {exc}"
            ) from exc
        if getattr(model.config, "model_type", None) != "evo2":
            raise ValueError(
                f"'{m.repo_id}' is not an Evo 2 transformers-port checkpoint "
                f"(model_type={model.config.model_type!r})."
            )
        _patch_memory_hogs(model)
        model.to(DEVICE)
        model.eval()
        return tokenizer, model

    return _cache.get_or_load(f"{m.repo_id}@{m.revision or 'main'}", _load)


def max_length(m: ResolvedModel, model=None) -> int:
    """Longest accepted sequence (bases): the checkpoint's context, capped by EVO2_MAX_SEQ_LEN."""
    native = m.max_tokens
    if native is None and model is not None:
        native = int(getattr(model.config, "max_position_embeddings", MAX_SEQ_LEN))
    return min(native, MAX_SEQ_LEN) if native else MAX_SEQ_LEN


def check_length(n: int, limit: int, what: str = "Sequence") -> None:
    if n > limit:
        raise ValueError(
            f"{what} is {n} bp, above this service's {limit} bp limit (Evo 2's native context is "
            "longer, but this pure-PyTorch port is slow and memory heavy on long inputs; the "
            "operator can raise EVO2_MAX_SEQ_LEN). Sequences are never truncated -- split it into "
            "shorter windows."
        )


def _encode(sequence: str) -> torch.Tensor:
    """`(1, L)` token ids: Evo 2's tokenizer is raw UTF-8 bytes (validated A/C/G/T/N only)."""
    return torch.tensor([list(sequence.encode("ascii"))], dtype=torch.long, device=DEVICE)


def _prepare(m: ResolvedModel, model, sequences: list[str]) -> list[str]:
    if not sequences:
        raise ValueError("sequences must be a non-empty list of DNA sequences.")
    seqs = [validate_sequence(s) for s in sequences]
    limit = max_length(m, model)
    for i, s in enumerate(seqs):
        check_length(len(s), limit, f"Sequence {i}" if len(seqs) > 1 else "Sequence")
    return seqs


# --- layers ----------------------------------------------------------------------


def num_blocks(model) -> int:
    return len(model.model.layers)


def recommended_layers(m: ResolvedModel, model) -> list[str]:
    if m.recommended_layers:
        return list(m.recommended_layers)
    return [f"blocks.{round(_RECOMMENDED_DEPTH * num_blocks(model)) - 1}.mlp.l3"]


def parse_layer(m: ResolvedModel, model, layer: str | int | None) -> int | str:
    """Normalize `layer` to a hidden-state index (int) or an Evo 2 `blocks.N.mlp.l3` name.

    - "last" / None: the final hidden state (after the final RMSNorm), index num_blocks.
    - an integer (or integer string): hidden-state index as in `output_hidden_states`
      (0 = token embeddings, i = output of block i-1; negative counts from the end).
    - "blocks.N.mlp.l3": output of block N's MLP before the residual add -- the layer
      naming of Arc's evo2 package (`return_embeddings=True, layer_names=[...]`).
    - "recommended": the first layer of get_embedding_layers(which="recommended").
    """
    n = num_blocks(model)
    if isinstance(layer, str):
        key = layer.strip()
        if key.lower() == "recommended":
            key = recommended_layers(m, model)[0]
        match = _BLOCK_LAYER_RE.match(key)
        if match:
            block = int(match.group(1))
            if not 0 <= block < n:
                raise ValueError(f"Block index must be between 0 and {n - 1}; got {layer!r}.")
            return f"blocks.{block}.mlp.l3"
        layer = key
    if layer is None or (isinstance(layer, str) and layer.lower() == "last"):
        return n
    try:
        idx = int(layer)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"layer_name must be 'last', 'recommended', a hidden-state index (0-{n}) or "
            f"'blocks.N.mlp.l3' (N = 0-{n - 1}); got {layer!r}."
        ) from exc
    if idx < 0:
        idx = n + 1 + idx
    if not 0 <= idx <= n:
        raise ValueError(f"layer index must be between 0 and {n} (or 'last'); got {layer!r}.")
    return idx


def _layer_module(model, layer: int | str):
    inner = model.model
    if isinstance(layer, str):
        return inner.layers[int(_BLOCK_LAYER_RE.match(layer).group(1))].mlp
    if layer == 0:
        return inner.embed_tokens
    if layer == len(inner.layers):
        return inner.final_norm
    return inner.layers[layer - 1]


class _StopForward(Exception):
    """Raised by the capture hook: nothing after the requested layer needs computing."""


def _capture(model, ids: torch.Tensor, layer: int | str) -> torch.Tensor:
    """Hidden state `(L, hidden)` at `layer` for one sequence, skipping the later blocks."""
    captured = {}

    def hook(_module, _inputs, output):
        captured["h"] = output[0] if isinstance(output, tuple) else output
        raise _StopForward

    handle = _layer_module(model, layer).register_forward_hook(hook)
    try:
        model.model(input_ids=ids)  # no attention_mask: see module docstring
    except _StopForward:
        pass
    finally:
        handle.remove()
    return captured["h"][0].float()


# --- tools -----------------------------------------------------------------------


def model_info(m: ResolvedModel) -> dict:
    _, model = load(m)
    cfg = model.config
    safe_cfg = {
        k: v
        for k, v in cfg.to_dict().items()
        if isinstance(v, (int, float, str, bool, list)) or v is None
    }
    return {
        "repo_id": m.repo_id,
        "revision": m.revision,
        "device": DEVICE,
        "dtype": str(DTYPE).replace("torch.", ""),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "vocab_size": int(cfg.vocab_size),
        "hidden_size": int(cfg.hidden_size),
        "num_blocks": num_blocks(model),
        "num_hidden_state_layers": num_blocks(model) + 1,
        "attention_blocks": list(cfg.attn_layer_idxs),
        "native_context": int(cfg.max_position_embeddings),
        "max_sequence_length": max_length(m, model),
        "max_new_tokens": MAX_NEW_TOKENS,
        "tokenizer": "byte-level, one token per base, no special tokens",
        "causal": True,
        "config": safe_cfg,
    }


def list_embedding_layers(m: ResolvedModel, which: str) -> dict:
    _, model = load(m)
    n = num_blocks(model)
    if which == "recommended":
        layers: list[int | str] = recommended_layers(m, model)
        info = (
            "Evo 2's paper and Arc's examples use intermediate-layer embeddings, not the last "
            "layer: for evo2_7b Arc's README embeds with 'blocks.28.mlp.l3' and the paper's sparse "
            "autoencoders read layer 26. 'blocks.N.mlp.l3' is block N's MLP output (before the "
            "residual add), named as in Arc's evo2 package. For checkpoints without a published "
            "choice this is a heuristic at the same relative depth."
        )
    elif which == "all":
        layers = list(range(n + 1)) + [f"blocks.{b}.mlp.l3" for b in range(n)]
        info = (
            f"Hidden-state indices 0-{n} (0 = token embeddings, i = output of block i-1, {n} = "
            "final layer after the last RMSNorm = 'last'), plus the per-block MLP outputs "
            "'blocks.N.mlp.l3'. Every layer has one position per base."
        )
    else:
        raise ValueError("which must be 'recommended' or 'all'.")
    return {
        "layers": layers,
        "bases_per_position": dict.fromkeys(layers, 1),
        "num_hidden_state_layers": n + 1,
        "info": info,
    }


def compute_embeddings(
    m: ResolvedModel, sequences: list[str], layer: str | int | None, pooling: str
) -> tuple[list[np.ndarray], int | str, int]:
    """One vector per sequence ("mean") or one `(bases, hidden)` matrix per sequence ("per_token")."""
    if pooling not in ("mean", "per_token"):
        raise ValueError("pooling must be 'mean' or 'per_token'.")
    _, model = load(m)
    seqs = _prepare(m, model, sequences)
    layer_id = parse_layer(m, model, layer)
    if pooling == "per_token":
        total = sum(len(s) for s in seqs) * int(model.config.hidden_size)
        if total > MAX_OUTPUT_VALUES:
            raise ValueError(
                f"per_token output would be {total:,} values (bases x hidden size "
                f"{model.config.hidden_size}), above the {MAX_OUTPUT_VALUES:,} limit. Use "
                "pooling='mean', shorter sequences, or fewer sequences per call."
            )
    out = []
    with _gpu_lock, torch.inference_mode():
        for s in seqs:
            h = _capture(model, _encode(s), layer_id)
            out.append((h.mean(dim=0) if pooling == "mean" else h).cpu().numpy())
    return out, layer_id, num_blocks(model)


def compare_sequences(m: ResolvedModel, sequence_a: str, sequence_b: str, layer: str | int | None):
    embeddings, layer_id, _ = compute_embeddings(m, [sequence_a, sequence_b], layer, "mean")
    a, b = embeddings
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return (float(np.dot(a, b) / denom) if denom > 0 else 0.0), layer_id


def lm_logits(model, hidden: torch.Tensor) -> torch.Tensor:
    """Next-token logits from final hidden states, with the LM head computed in fp32.

    The port's bf16 head rounds logits of magnitude 64-128 to steps of 0.5, i.e. up
    to +-0.25 nats per base; that rounding, not the model, dominated the noise of
    SNP deltas. The head is only (512 x hidden), so fp32 costs nothing.
    """
    return torch.nn.functional.linear(hidden.float(), model.lm_head.weight.float())


def token_log_probs(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """`log P(x_t | x_<t)` for t = 1..L-1 from `(L, vocab)` logits and `(L,)` ids.

    The first base has no context and is not scored (as in Arc's `score_sequences`).
    """
    log_probs = torch.log_softmax(logits[:-1].float(), dim=-1)
    return log_probs.gather(-1, ids[1:, None]).squeeze(-1)


def log_likelihood(m: ResolvedModel, sequence: str) -> float:
    """Exact causal log-likelihood: sum of `log P(x_t | x_<t)` over bases 2..L."""
    _, model = load(m)
    (seq,) = _prepare(m, model, [sequence])
    if len(seq) < 2:
        raise ValueError(
            "Sequence must be at least 2 bp long to score (the first base has no context)."
        )
    ids = _encode(seq)
    with _gpu_lock, torch.inference_mode():
        hidden = model.model(input_ids=ids).last_hidden_state[0]  # no attention_mask
        return float(token_log_probs(lm_logits(model, hidden), ids[0]).sum().item())


def score_variant(
    m: ResolvedModel, sequence: str, alt_allele: str, position: int | None = None
) -> dict:
    return score_snp_whole_sequence(
        sequence, alt_allele, lambda s: log_likelihood(m, s), position=position
    )


def sample_next(
    logits: torch.Tensor, temperature: float, top_k: int, generator: torch.Generator | None
) -> int:
    """Pick the next base from `(vocab,)` logits, restricted to A/C/G/T.

    temperature 0 is greedy. top_k keeps the k most likely of the four bases
    (0 = no filtering).
    """
    ids = torch.tensor(list(NUCLEOTIDE_TOKENS.values()), device=logits.device)
    scores = logits.float()[ids]
    if temperature == 0:
        return int(ids[int(scores.argmax())])
    scores = scores / temperature
    if 0 < top_k < len(ids):
        kth = torch.topk(scores, top_k).values[-1]
        scores = scores.masked_fill(scores < kth, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    choice = torch.multinomial(probs.cpu(), 1, generator=generator)
    return int(ids[int(choice)])


def validate_generation_args(max_new_tokens: int, temperature: float, top_k: int) -> None:
    if not 1 <= int(max_new_tokens) <= MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be between 1 and {MAX_NEW_TOKENS}.")
    if not 0 <= float(temperature) <= 10:
        raise ValueError("temperature must be between 0 (greedy) and 10.")
    if not 0 <= int(top_k) <= len(NUCLEOTIDE_TOKENS):
        raise ValueError("top_k must be between 0 (no filtering) and 4 (sampling is over A/C/G/T).")


def generate(
    m: ResolvedModel,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int | None = None,
) -> dict:
    """Autoregressive continuation of `prompt` with the port's decoding cache.

    Prefill runs the full (causal, unmasked) forward once and seeds the per-layer
    caches; each step then feeds one token with explicit `position_ids` and an
    all-ones attention mask over the cached keys (without a mask the port's SDPA
    call would apply a top-left causal mask and the new token would only see the
    first key).
    """
    validate_generation_args(max_new_tokens, temperature, top_k)
    _, model = load(m)
    (seq,) = _prepare(m, model, [prompt])
    check_length(len(seq) + max_new_tokens, max_length(m, model), "Prompt + max_new_tokens")
    generator = torch.Generator().manual_seed(int(seed)) if seed is not None else None
    ids = _encode(seq)
    new: list[int] = []
    with _gpu_lock, torch.inference_mode():
        out = model.model(input_ids=ids, use_cache=True)
        cache = out.past_key_values
        logits = lm_logits(model, out.last_hidden_state[0, -1])
        next_id = sample_next(logits, temperature, top_k, generator)
        new.append(next_id)
        for _ in range(max_new_tokens - 1):
            pos = len(seq) + len(new) - 1
            step = torch.tensor([[next_id]], device=DEVICE)
            out = model.model(
                input_ids=step,
                past_key_values=cache,
                use_cache=True,
                position_ids=torch.tensor([[pos]], device=DEVICE),
                attention_mask=torch.ones(1, pos + 1, dtype=torch.long, device=DEVICE),
            )
            cache = out.past_key_values
            logits = lm_logits(model, out.last_hidden_state[0, -1])
            next_id = sample_next(logits, temperature, top_k, generator)
            new.append(next_id)
    generated = bytes(new).decode("ascii")
    return {
        "prompt": seq,
        "generated_sequence": generated,
        "full_sequence": seq + generated,
        # Evo 2 has one token per base, so the two counts are always equal.
        "num_new_tokens": len(new),
        "num_new_bases": len(generated),
    }
