"""Known GENERator checkpoints published by GenerTeam on the HuggingFace Hub.

Tools accept short aliases (``"v2-eukaryote-1.2b"``) instead of full repo ids. Any
other HuggingFace repo id is passed straight through, as long as it is a
Llama-architecture checkpoint with GENERator's ``vocab.txt`` 6-mer vocabulary.

All registered repos are plain Llama weights (``model_type: llama``) plus a remote
``GENERatorForCausalLM`` subclass and ``DNAKmerTokenizer``. This server loads the
weights with the stock ``LlamaForCausalLM`` and re-implements the tokenizer
(``kmer.py``), so no remote code runs. Revisions are still pinned, so a Hub-side
change to the weights or vocabulary can't silently change what this server runs.

Source: https://huggingface.co/GenerTeam , https://github.com/GenerTeam/GENERator
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    revision: str | None
    params: str
    generation: str  # "v1" | "v2"
    domain: str  # "eukaryote" | "prokaryote"
    max_tokens: int  # context limit in tokens, including the <s> markers
    notes: str = ""

    @property
    def separator(self) -> bool:
        """v2 model cards append a trailing ``<s>`` separator when embedding; v1 cards don't."""
        return self.generation == "v2"


# max_position_embeddings = 16384 tokens of 6 bases: the ~98 kb context from the paper.
_MAX_TOKENS = 16384

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "v2-eukaryote-1.2b": ModelSpec(
        "GenerTeam/GENERator-v2-eukaryote-1.2b-base",
        "c41b0018da9ee13b9e96ee54647de8da381ccd72",
        "1.2B",
        "v2",
        "eukaryote",
        _MAX_TOKENS,
        "Recommended default: GENERator-v2, eukaryotic DNA, 1.2B parameters, ~98 kb context.",
    ),
    "v2-eukaryote-3b": ModelSpec(
        "GenerTeam/GENERator-v2-eukaryote-3b-base",
        "7dc01bccce5b65e15141170538afdc2ff09d8dde",
        "3B",
        "v2",
        "eukaryote",
        _MAX_TOKENS,
    ),
    "v2-prokaryote-1.2b": ModelSpec(
        "GenerTeam/GENERator-v2-prokaryote-1.2b-base",
        "8b2f768b0d293953518ff91d34600f9322ef1f94",
        "1.2B",
        "v2",
        "prokaryote",
        _MAX_TOKENS,
    ),
    "v2-prokaryote-3b": ModelSpec(
        "GenerTeam/GENERator-v2-prokaryote-3b-base",
        "b18ac86df77359d894d7bc050cea78e2d0713021",
        "3B",
        "v2",
        "prokaryote",
        _MAX_TOKENS,
    ),
    "eukaryote-1.2b": ModelSpec(
        "GenerTeam/GENERator-eukaryote-1.2b-base",
        "5e872c94264891f9adf59d8ea64e426bb68badb5",
        "1.2B",
        "v1",
        "eukaryote",
        _MAX_TOKENS,
        "Original GENERator (v1, arXiv 2502.07272), eukaryotic DNA, 1.2B parameters.",
    ),
    "eukaryote-3b": ModelSpec(
        "GenerTeam/GENERator-eukaryote-3b-base",
        "7515b17659f092997335226c3fb6aafd06c0add9",
        "3B",
        "v1",
        "eukaryote",
        _MAX_TOKENS,
        "Original GENERator (v1), eukaryotic DNA, 3B parameters.",
    ),
}

DEFAULT_MODEL_ALIAS = "v2-eukaryote-1.2b"


@dataclass(frozen=True)
class ResolvedModel:
    repo_id: str
    revision: str | None
    max_tokens: int | None  # None: unregistered repo, read config.max_position_embeddings
    separator: bool  # append a trailing <s> when embedding (v2 recipe)


def resolve_model(name: str | None) -> ResolvedModel:
    """Resolve a registry alias or a raw HuggingFace repo id."""
    if not name:
        name = DEFAULT_MODEL_ALIAS
    key = name.strip().lower()
    if key in MODEL_REGISTRY:
        spec = MODEL_REGISTRY[key]
        return ResolvedModel(spec.repo_id, spec.revision, spec.max_tokens, spec.separator)
    if "/" in name:
        repo = name.strip()
        # Unregistered repo: follow the v2 embedding recipe only for repos named like v2.
        return ResolvedModel(repo, None, None, "-v2-" in repo.lower())
    raise ValueError(
        f"Unknown model '{name}'. Call list_available_checkpoints for available aliases, or pass a "
        "full HuggingFace repo id (e.g. 'GenerTeam/GENERator-v2-eukaryote-1.2b-base')."
    )
