"""Model loading and inference helpers for NTv3.

NTv3 checkpoints are single-base (character-level) DNA language models shipped as
custom ``trust_remote_code`` architectures on the HuggingFace Hub. See the model
cards under https://huggingface.co/InstaDeepAI, e.g.
https://huggingface.co/InstaDeepAI/NTv3_100M_pre

Two kinds of checkpoint:

- **pre** (``model_type="ntv3"``, ``AutoModelForMaskedLM``): masked LM over DNA only.
- **post** (``model_type="ntv3_posttrained"``, ``AutoModel``): the same backbone
  additionally trained on functional genomics, conditioned on a **species** token
  (``species_ids``, required by the model). Besides MLM logits and embeddings it
  outputs genome-annotation probabilities (``bed_tracks_logits``: 21 element
  classes such as exon/intron/promoter) and experimental signal tracks
  (``bigwig_tracks_logits``: RNA-seq/ChIP/ATAC/... per species), both only for the
  central ``keep_target_center_fraction`` (37.5%) of the padded input window.
  Mirrors InstaDeep's own ``ntv3_tracks_pipeline.py`` shipped in the post repos.

NTv3 is U-Net shaped: hidden states 0 and N are full length, the ones in between
are downsampled by powers of two (down to 1/128 inside the transformer).

All heavy lifting happens here so the MCP tool functions in ``server.py`` stay thin.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from dnalm_common import DEFAULT_VALID_NUCLEOTIDES, LRUModelCache, select_device, select_dtype
from dnalm_common import validate_sequence as validate_sequence  # re-exported for callers/tests
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

VALID_NUCLEOTIDES = DEFAULT_VALID_NUCLEOTIDES

DEVICE = select_device("NTV3_DEVICE")
DTYPE = select_dtype("NTV3_DTYPE")

_MAX_RESIDENT_MODELS = int(os.environ.get("NTV3_MAX_RESIDENT_MODELS", "2"))
_cache = LRUModelCache(max_entries=_MAX_RESIDENT_MODELS)

POST_MODEL_TYPE = "ntv3_posttrained"
DEFAULT_SPECIES = "human"
# Cap on numbers returned by annotate_sequence/predict_tracks, so a long sequence
# at bin_size=1 can't produce a response of hundreds of MB.
MAX_OUTPUT_VALUES = int(os.environ.get("NTV3_MAX_OUTPUT_VALUES", "500000"))
MAX_TRACKS_PER_CALL = 64


def load(repo_id: str):
    """Load (and cache) the tokenizer + model for a given HuggingFace repo id.

    Cached via a bounded LRU (`NTV3_MAX_RESIDENT_MODELS`, default 2): loading a
    checkpoint past that limit evicts the least-recently-used one and frees its
    VRAM, since this GPU may be shared with other dnalm-hub services.
    Post-trained checkpoints only register `AutoModel`, pre-trained ones
    `AutoModelForMaskedLM`; the config's `model_type` decides which.
    """

    def _load() -> tuple:
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
            config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
            model_cls = AutoModel if config.model_type == POST_MODEL_TYPE else AutoModelForMaskedLM
            model = model_cls.from_pretrained(repo_id, trust_remote_code=True, torch_dtype=DTYPE)
        except Exception as exc:  # noqa: BLE001 - surface a clear, actionable message
            raise RuntimeError(
                f"Failed to load NTv3 model '{repo_id}' from HuggingFace: {exc}"
            ) from exc
        tokenizer.padding_side = "right"
        model.to(DEVICE)
        model.eval()
        return tokenizer, model

    return _cache.get_or_load(repo_id, _load)


def is_post_trained(model) -> bool:
    return getattr(model.config, "model_type", None) == POST_MODEL_TYPE


def supported_species(config) -> list[str]:
    return sorted(k for k in config.species_to_token_id if not k.startswith("<"))


def resolve_species(model, species: str | None) -> str | None:
    """Validate `species` for this model: required-with-default for post, rejected for pre."""
    if not is_post_trained(model):
        if species:
            raise ValueError(
                "`species` only applies to post-trained NTv3 checkpoints (e.g. '100m-post'); "
                "pre-trained checkpoints are species-agnostic -- omit it."
            )
        return None
    name = (species or DEFAULT_SPECIES).strip().lower()
    options = supported_species(model.config)
    if name not in options:
        raise ValueError(f"Unknown species {species!r}. Supported species: {options}")
    return name


def _forward(model, batch: dict, species: str | None, **kwargs):
    """Run the model; post-trained models get `species_ids` (they don't take an attention mask)."""
    if is_post_trained(model):
        input_ids = batch["input_ids"]
        species_ids = model.encode_species([species] * input_ids.shape[0]).to(input_ids.device)
        return model(input_ids=input_ids, species_ids=species_ids, **kwargs)
    return model(**batch, **kwargs)


def _parse_layer(layer: str | int | None, n_layers: int) -> int:
    if layer in (None, "last", "LAST"):
        return n_layers
    try:
        idx = int(layer)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"layer must be an integer (as int or string) or 'last'; got {layer!r}") from exc
    if idx < 0:
        idx = n_layers + 1 + idx
    if not (0 <= idx <= n_layers):
        raise ValueError(f"layer must be between 0 and {n_layers} (or 'last'); got {layer!r}")
    return idx


def layer_lengths(real_lengths: torch.Tensor, padded_len: int, layer_len: int) -> torch.Tensor:
    """Number of non-padding positions per sequence at a (possibly downsampled) layer.

    Inputs are right-padded to a multiple of the total downsampling factor, so a
    layer of length `layer_len` covers `padded_len / layer_len` bases per position.
    """
    factor = padded_len // layer_len
    return torch.div(real_lengths + factor - 1, factor, rounding_mode="floor")


def _tokenize(tokenizer, sequences: list[str], pad_multiple: int):
    batch = tokenizer(
        sequences,
        add_special_tokens=False,
        padding=True,
        pad_to_multiple_of=pad_multiple,
        return_tensors="pt",
        return_attention_mask=True,
    )
    return {k: v.to(DEVICE) for k, v in batch.items()}


def _probe_hidden_states(repo_id: str, pad_multiple: int):
    """Run a tiny dummy forward pass to inspect how many hidden-state layers a model exposes."""
    tokenizer, model = load(repo_id)
    dummy = "A" * pad_multiple
    batch = _tokenize(tokenizer, [dummy], pad_multiple)
    with torch.no_grad():
        out = _forward(model, batch, resolve_species(model, None), output_hidden_states=True)
    return tokenizer, model, out


def model_info(repo_id: str, pad_multiple: int) -> dict:
    tokenizer, model, out = _probe_hidden_states(repo_id, pad_multiple)
    n_params = sum(p.numel() for p in model.parameters())
    raw_cfg = model.config.to_dict()
    safe_cfg = {k: v for k, v in raw_cfg.items() if isinstance(v, (int, float, str, bool)) or v is None}
    info = {
        "repo_id": repo_id,
        "device": DEVICE,
        "dtype": str(DTYPE).replace("torch.", ""),
        "num_parameters": n_params,
        "vocab_size": len(tokenizer),
        "hidden_size": int(out.hidden_states[-1].shape[-1]),
        "num_hidden_state_layers": len(out.hidden_states),
        "pad_multiple": pad_multiple,
        "mask_token": tokenizer.mask_token,
        "post_trained": is_post_trained(model),
        "config": safe_cfg,
    }
    if is_post_trained(model):
        info["supported_species"] = supported_species(model.config)
        info["default_species"] = DEFAULT_SPECIES
        info["annotation_elements"] = list(model.config.bed_elements_names)
        info["predicted_center_fraction"] = model.config.keep_target_center_fraction
    return info


def list_embedding_layers(repo_id: str, pad_multiple: int, which: str) -> dict:
    """Enumerate hidden-state layer indices usable as `layer_name` in embed_sequence.

    `which="recommended"` is a lightweight heuristic (the final layer, plus the
    full-resolution input layer) rather than an InstaDeep-published recommendation.
    `which="all"` returns every valid index; layers between the first and the last
    are downsampled (U-Net), so their per_token output has fewer rows than bases.
    """
    _, _, out = _probe_hidden_states(repo_id, pad_multiple)
    n_layers = len(out.hidden_states) - 1
    full_len = out.hidden_states[0].shape[1]
    resolution = [full_len // h.shape[1] for h in out.hidden_states]
    if which == "all":
        layers = list(range(0, n_layers + 1))
        info = (
            f"All {n_layers + 1} hidden-state layers (0=input embeddings, {n_layers}=final layer/'last'). "
            "`bases_per_position` gives each layer's downsampling (U-Net architecture)."
        )
    elif which == "recommended":
        layers = [n_layers]
        info = (
            "Heuristic only: the final full-resolution layer. NTv3 does not publish a recommended "
            "embedding layer; use which='all' to explore the downsampled middle layers."
        )
    else:
        raise ValueError("which must be 'recommended' or 'all'.")
    return {
        "layers": layers,
        "bases_per_position": {i: resolution[i] for i in layers},
        "num_hidden_state_layers": len(out.hidden_states),
        "info": info,
    }


def compute_embeddings(
    repo_id: str,
    sequences: list[str],
    layer: str | int | None,
    pooling: str,
    pad_multiple: int,
    species: str | None = None,
) -> tuple[list[np.ndarray], int, int]:
    """One 1D vector per sequence ("mean") or one (positions, hidden) matrix per sequence ("per_token").

    Padding never leaks into the result: mean pooling averages only real
    positions, and per_token rows are trimmed to each sequence's real length
    (at the chosen layer's resolution).
    """
    if not sequences:
        raise ValueError("sequences must be a non-empty list of DNA sequences.")
    if pooling not in ("mean", "per_token"):
        raise ValueError("pooling must be 'mean' or 'per_token'.")
    tokenizer, model = load(repo_id)
    species = resolve_species(model, species)
    seqs = [validate_sequence(s) for s in sequences]
    batch = _tokenize(tokenizer, seqs, pad_multiple)
    with torch.no_grad():
        out = _forward(model, batch, species, output_hidden_states=True)
    hidden_states = out.hidden_states
    n_layers = len(hidden_states) - 1
    idx = _parse_layer(layer, n_layers)
    h = hidden_states[idx].float()
    real = layer_lengths(batch["attention_mask"].sum(dim=1), batch["input_ids"].shape[1], h.shape[1])

    if pooling == "mean":
        mask = (torch.arange(h.shape[1], device=h.device)[None, :] < real[:, None]).unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return list(pooled.cpu().numpy()), idx, n_layers
    return [h[i, : int(n)].cpu().numpy() for i, n in enumerate(real.tolist())], idx, n_layers


def score_variant(
    repo_id: str,
    sequence: str,
    alt_allele: str,
    pad_multiple: int,
    position: int | None = None,
    ref_allele: str | None = None,
    species: str | None = None,
) -> dict:
    """Zero-shot single-nucleotide substitution scoring via one masked-LM forward pass.

    If `position` is omitted it defaults to the sequence center (matching evo2-mcp's
    `score_snp`, which always scores the center position). If `ref_allele` is
    omitted it is read directly from `sequence[position]`; if given, it must match.
    """
    tokenizer, model = load(repo_id)
    species = resolve_species(model, species)
    seq = validate_sequence(sequence)
    if position is None:
        position = len(seq) // 2
    if not (0 <= position < len(seq)):
        raise ValueError(f"position must be within [0, {len(seq) - 1}] for a sequence of length {len(seq)}.")

    ref = (ref_allele.strip().upper() if ref_allele else seq[position])
    alt = alt_allele.strip().upper()
    if len(ref) != 1 or ref not in VALID_NUCLEOTIDES:
        raise ValueError(f"ref_allele must be a single character in {sorted(VALID_NUCLEOTIDES)}; got {ref_allele!r}.")
    if len(alt) != 1 or alt not in VALID_NUCLEOTIDES:
        raise ValueError(f"alt_allele must be a single character in {sorted(VALID_NUCLEOTIDES)}; got {alt_allele!r}.")
    if seq[position] != ref:
        raise ValueError(
            f"Reference allele mismatch: sequence has '{seq[position]}' at position {position}, expected '{ref}'."
        )
    if tokenizer.mask_token_id is None:
        raise RuntimeError(f"Tokenizer for '{repo_id}' has no mask token; variant scoring is unavailable.")

    batch = _tokenize(tokenizer, [seq], pad_multiple)
    input_ids = batch["input_ids"].clone()
    input_ids[0, position] = tokenizer.mask_token_id
    batch["input_ids"] = input_ids

    with torch.no_grad():
        out = _forward(model, batch, species)
    logits = out.logits[0, position].float()
    log_probs = torch.log_softmax(logits, dim=-1)

    ref_id = tokenizer.convert_tokens_to_ids(ref)
    alt_id = tokenizer.convert_tokens_to_ids(alt)
    if ref_id is None or ref_id == tokenizer.unk_token_id:
        raise ValueError(f"ref_allele '{ref}' is not a recognized token for this tokenizer.")
    if alt_id is None or alt_id == tokenizer.unk_token_id:
        raise ValueError(f"alt_allele '{alt}' is not a recognized token for this tokenizer.")

    ref_log_prob = log_probs[ref_id].item()
    alt_log_prob = log_probs[alt_id].item()
    mutated_seq = seq[:position] + alt + seq[position + 1 :]
    return {
        "sequence": seq,
        "mutated_sequence": mutated_seq,
        "position": position,
        "ref_allele": ref,
        "alt_allele": alt,
        "ref_log_prob": ref_log_prob,
        "alt_log_prob": alt_log_prob,
        "log_likelihood_ratio": alt_log_prob - ref_log_prob,
        "species": species,
    }


def predict_masked(
    repo_id: str,
    sequence: str,
    positions: list[int],
    top_k: int,
    pad_multiple: int,
    species: str | None = None,
) -> list[dict]:
    tokenizer, model = load(repo_id)
    species = resolve_species(model, species)
    seq = validate_sequence(sequence)

    if not positions:
        positions = [i for i, c in enumerate(seq) if c == "N"]
        if not positions:
            raise ValueError(
                "No 'positions' given and the sequence contains no 'N' characters to auto-mask."
            )
    for p in positions:
        if not (0 <= p < len(seq)):
            raise ValueError(f"position {p} is out of range [0, {len(seq) - 1}] for this sequence.")
    if tokenizer.mask_token_id is None:
        raise RuntimeError(f"Tokenizer for '{repo_id}' has no mask token; masked prediction is unavailable.")
    top_k = max(1, min(top_k, len(VALID_NUCLEOTIDES)))

    single = _tokenize(tokenizer, [seq], pad_multiple)
    base_ids = single["input_ids"][0]
    base_attn = single["attention_mask"][0]

    batch_ids = base_ids.unsqueeze(0).repeat(len(positions), 1).clone()
    for row, p in enumerate(positions):
        batch_ids[row, p] = tokenizer.mask_token_id
    batch_attn = base_attn.unsqueeze(0).repeat(len(positions), 1)

    with torch.no_grad():
        out = _forward(model, {"input_ids": batch_ids, "attention_mask": batch_attn}, species)

    nt_ids = {nt: tokenizer.convert_tokens_to_ids(nt) for nt in sorted(VALID_NUCLEOTIDES)}
    nt_ids = {nt: i for nt, i in nt_ids.items() if i is not None and i != tokenizer.unk_token_id}

    results = []
    for row, p in enumerate(positions):
        probs = torch.softmax(out.logits[row, p].float(), dim=-1)
        ranked = sorted(((nt, probs[i].item()) for nt, i in nt_ids.items()), key=lambda kv: kv[1], reverse=True)
        results.append(
            {
                "position": p,
                "original": seq[p],
                "predictions": [{"nucleotide": nt, "probability": prob} for nt, prob in ranked[:top_k]],
            }
        )
    return results


def compare_sequences(
    repo_id: str,
    sequence_a: str,
    sequence_b: str,
    layer: str | int | None,
    pad_multiple: int,
    species: str | None = None,
) -> tuple[float, int]:
    embeddings, layer_idx, _ = compute_embeddings(
        repo_id, [sequence_a, sequence_b], layer, "mean", pad_multiple, species=species
    )
    a, b = embeddings[0], embeddings[1]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    similarity = float(np.dot(a, b) / denom) if denom > 0 else 0.0
    return similarity, layer_idx


# --- post-trained only: species, genome annotation, functional tracks -------------


def _post_config(repo_id: str):
    """Config of a post-trained checkpoint (cheap: no weights), or a clear error."""
    config = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
    if config.model_type != POST_MODEL_TYPE:
        raise ValueError(
            f"'{repo_id}' is a pre-trained NTv3 checkpoint; species, annotation and tracks need a "
            "post-trained one (e.g. '100m-post'). Call list_available_checkpoints (stage='post')."
        )
    return config


def species_overview(repo_id: str) -> dict:
    config = _post_config(repo_id)
    tracks = config.bigwigs_per_species
    return {
        "default_species": DEFAULT_SPECIES,
        "species": [{"name": s, "num_tracks": len(tracks.get(s, []))} for s in supported_species(config)],
        "annotation_elements": list(config.bed_elements_names),
        "predicted_center_fraction": config.keep_target_center_fraction,
    }


def list_track_ids(repo_id: str, species: str | None, contains: str | None, limit: int, offset: int) -> dict:
    config = _post_config(repo_id)
    name = (species or DEFAULT_SPECIES).strip().lower()
    if name not in supported_species(config):
        raise ValueError(f"Unknown species {species!r}. Supported species: {supported_species(config)}")
    ids = list(config.bigwigs_per_species.get(name, []))
    if contains:
        ids = [t for t in ids if contains.lower() in t.lower()]
    limit = max(1, min(int(limit), 1000))
    offset = max(0, int(offset))
    return {"species": name, "total": len(ids), "offset": offset, "track_ids": ids[offset : offset + limit]}


def center_region(padded_len: int, seq_len: int, keep_fraction: float) -> tuple[int, int, int]:
    """Where the post heads' predictions fall, in sequence coordinates.

    The model crops its output to the central `keep_fraction` of the padded
    window. Returns `(crop_start, region_start, region_end)`: `crop_start` is the
    first padded position the output covers, and the region is that crop clipped
    to the real (unpadded) sequence.
    """
    crop_len = int(padded_len * keep_fraction)
    crop_start = (padded_len - crop_len) // 2
    region_start = crop_start
    region_end = min(crop_start + crop_len, seq_len)
    if region_end <= region_start:
        raise ValueError(
            f"Sequence too short: predictions cover only positions {crop_start}-{crop_start + crop_len} of the "
            f"{padded_len}-long padded window, which is past the end of this {seq_len} bp sequence. Pass a longer "
            "window (ideally a multiple of 128 bp; the central 37.5% gets predictions)."
        )
    return crop_start, region_start, region_end


def bin_mean(values: np.ndarray, bin_size: int) -> np.ndarray:
    """Average `(positions, columns)` over consecutive bins of `bin_size` positions (last bin may be partial)."""
    if bin_size < 1:
        raise ValueError("bin_size must be >= 1.")
    if bin_size == 1:
        return values
    n = values.shape[0]
    starts = np.arange(0, n, bin_size)
    return np.add.reduceat(values, starts, axis=0) / np.diff(np.append(starts, n))[:, None]


def check_output_size(num_positions: int, num_columns: int, bin_size: int) -> None:
    n_bins = -(-num_positions // bin_size)
    total = n_bins * num_columns
    if total > MAX_OUTPUT_VALUES:
        needed = -(-num_positions * num_columns // MAX_OUTPUT_VALUES)
        raise ValueError(
            f"Response would contain {total:,} values (limit {MAX_OUTPUT_VALUES:,}). Increase bin_size "
            f"(>= {needed}) or request fewer elements/tracks."
        )


def _post_forward_single(repo_id: str, sequence: str, species: str | None, pad_multiple: int, output_track: bool):
    tokenizer, model = load(repo_id)
    if not is_post_trained(model):
        _post_config(repo_id)  # raises the "needs a post-trained checkpoint" error
    species = resolve_species(model, species)
    seq = validate_sequence(sequence)
    batch = _tokenize(tokenizer, [seq], pad_multiple)
    padded_len = batch["input_ids"].shape[1]
    crop_start, region_start, region_end = center_region(
        padded_len, len(seq), model.config.keep_target_center_fraction
    )
    with torch.no_grad():
        out = _forward(model, batch, species, output_track=output_track)
    lo, hi = region_start - crop_start, region_end - crop_start
    meta = {"species": species, "sequence_length": len(seq), "region_start": region_start, "region_end": region_end}
    return model, out, (lo, hi), meta


def annotate(
    repo_id: str,
    sequence: str,
    pad_multiple: int,
    species: str | None = None,
    elements: list[str] | None = None,
    bin_size: int = 1,
) -> dict:
    """Per-position probability of each genomic element (softmax over absent/present, as InstaDeep's pipeline)."""
    names =list(_post_config(repo_id).bed_elements_names)
    wanted = names if not elements else elements
    unknown = [e for e in wanted if e not in names]
    if unknown:
        raise ValueError(f"Unknown element(s) {unknown}. Available: {names}")
    model, out, (lo, hi), meta = _post_forward_single(repo_id, sequence, species, pad_multiple, output_track=False)
    check_output_size(hi - lo, len(wanted), bin_size)
    probs = torch.softmax(out.bed_tracks_logits[0, lo:hi].float(), dim=-1)[..., 1]  # (positions, elements)
    cols = [names.index(e) for e in wanted]
    binned = bin_mean(probs[:, cols].cpu().numpy(), bin_size)
    return {
        **meta,
        "bin_size": bin_size,
        "elements": wanted,
        "max_probability": {e: round(float(binned[:, i].max()), 4) for i, e in enumerate(wanted)},
        "probabilities": {e: np.round(binned[:, i], 4).tolist() for i, e in enumerate(wanted)},
    }


def predict_tracks(
    repo_id: str,
    sequence: str,
    track_ids: list[str],
    pad_multiple: int,
    species: str | None = None,
    bin_size: int = 1,
) -> dict:
    """Predicted signal for selected experimental tracks (raw model output scale)."""
    if not track_ids:
        raise ValueError("track_ids must list at least one track id; see list_tracks.")
    if len(track_ids) > MAX_TRACKS_PER_CALL:
        raise ValueError(f"At most {MAX_TRACKS_PER_CALL} tracks per call; got {len(track_ids)}.")
    config = _post_config(repo_id)
    name = (species or DEFAULT_SPECIES).strip().lower()
    available = list(config.bigwigs_per_species.get(name, []))
    if name in config.species_to_token_id and not available:
        with_tracks = sorted(s for s, t in config.bigwigs_per_species.items() if t)
        raise ValueError(f"Species '{name}' has no predicted tracks in this model. Species with tracks: {with_tracks}")
    unknown = [t for t in track_ids if t not in available]
    if unknown and available:
        raise ValueError(f"Unknown track id(s) for {name}: {unknown[:10]}. Use list_tracks(species='{name}') to search.")
    model, out, (lo, hi), meta = _post_forward_single(repo_id, sequence, species, pad_multiple, output_track=True)
    check_output_size(hi - lo, len(track_ids), bin_size)
    cols = [available.index(t) for t in track_ids]
    signal = out.bigwig_tracks_logits[0, lo:hi][:, cols].float().cpu().numpy()
    binned = bin_mean(signal, bin_size)
    return {
        **meta,
        "bin_size": bin_size,
        "track_ids": track_ids,
        "tracks": {t: np.round(binned[:, i], 4).tolist() for i, t in enumerate(track_ids)},
    }
