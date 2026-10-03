"""GENERator's 6-mer tokenizer, re-implemented without ``trust_remote_code``.

The Hub repos ship a remote ``DNAKmerTokenizer`` (``tokenizer.py``). Its behavior
is simple and fully determined by ``vocab.txt`` plus ``k`` from
``tokenizer_config.json``: 32 special tokens, then every ``A/T/C/G`` k-mer.
DNA is cut into non-overlapping k-mers from the left; a k-mer that is not in the
vocabulary (it contains an ``N``) and a trailing chunk shorter than k both become
``<oov>``. This module reads the same two files, so no remote code runs, and
exposes what the server needs on top:

- the remainder policy for sequences whose length is not a multiple of k
  (``apply_remainder_policy``),
- wildcard matching (``matching_kmer_ids``) so an exact log-likelihood can
  marginalize over ``N`` bases and over a partial last k-mer,
- base-level lookup tables for the authors' per-base generation mode.

Pure Python (no torch), so it is unit-tested without model weights.
"""

from __future__ import annotations

from dataclasses import dataclass

BASES = "ACGT"
REMAINDER_POLICIES = ("reject", "trim_left", "pad_left")


@dataclass(frozen=True)
class KmerVocab:
    k: int
    token_to_id: dict[str, int]
    kmers: tuple[str, ...]  # every pure A/C/G/T k-mer, in token-id order
    kmer_ids: tuple[int, ...]  # their token ids, aligned with `kmers`

    @classmethod
    def from_vocab_text(cls, text: str, k: int) -> KmerVocab:
        """Parse GENERator's ``vocab.txt`` (one ``<token> <id>`` per line)."""
        token_to_id: dict[str, int] = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            token, idx = line.rsplit(maxsplit=1)
            token_to_id[token] = int(idx)
        for special in ("<s>", "<pad>", "<oov>"):
            if special not in token_to_id:
                raise ValueError(f"vocab.txt has no {special} token; not a GENERator vocabulary.")
        pairs = sorted(
            (i, t) for t, i in token_to_id.items() if len(t) == k and set(t) <= set(BASES)
        )
        if len(pairs) != 4**k:
            raise ValueError(f"vocab.txt has {len(pairs)} DNA {k}-mers, expected {4**k}.")
        return cls(k, token_to_id, tuple(t for _, t in pairs), tuple(i for i, _ in pairs))

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<s>"]

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def oov_id(self) -> int:
        return self.token_to_id["<oov>"]

    def encode(self, seq: str) -> list[int]:
        """Token ids for `seq` (no special tokens). Length must be a multiple of k.

        Same ids as the remote tokenizer: a k-mer containing ``N`` maps to ``<oov>``.
        """
        if len(seq) % self.k:
            raise ValueError(f"internal: encode() needs a multiple of {self.k} bases.")
        oov = self.oov_id
        return [self.token_to_id.get(seq[i : i + self.k], oov) for i in range(0, len(seq), self.k)]

    def decode(self, ids: list[int]) -> str:
        """Concatenate the k-mers of `ids`. Only DNA k-mer ids are accepted."""
        kmer_of = dict(zip(self.kmer_ids, self.kmers, strict=True))
        try:
            return "".join(kmer_of[i] for i in ids)
        except KeyError as exc:
            raise ValueError(f"internal: token id {exc.args[0]} is not a DNA k-mer.") from exc

    def matching_kmer_ids(self, pattern: str) -> list[int]:
        """Ids of the k-mers that start with `pattern`, treating ``N`` as any base.

        `pattern` may be shorter than k (a partial last k-mer): then every
        completion matches. Summing the model's probabilities over these ids gives
        the exact probability of the observed bases at that token position.
        """
        if not pattern or len(pattern) > self.k:
            raise ValueError(f"internal: pattern must have 1..{self.k} bases.")
        fixed = [(p, b) for p, b in enumerate(pattern) if b != "N"]
        return [
            i
            for kmer, i in zip(self.kmers, self.kmer_ids, strict=True)
            if all(kmer[p] == b for p, b in fixed)
        ]

    def base_tables(self) -> tuple[list[list[int]], list[int]]:
        """Lookup tables for per-base (marginal) generation.

        Returns `(kmer_base, flat_to_id)`: ``kmer_base[p][j]`` is the index in
        ``BASES`` of base p of ``kmers[j]``, and ``flat_to_id[f]`` is the token id of
        the k-mer whose base indices b_0..b_{k-1} give ``f = sum(b_p * 4**(k-1-p))``.
        """
        kmer_base = [[BASES.index(kmer[p]) for kmer in self.kmers] for p in range(self.k)]
        flat_to_id = [0] * (4**self.k)
        for kmer, i in zip(self.kmers, self.kmer_ids, strict=True):
            flat = 0
            for b in kmer:
                flat = flat * 4 + BASES.index(b)
            flat_to_id[flat] = i
        return kmer_base, flat_to_id


def apply_remainder_policy(seq: str, policy: str, k: int) -> tuple[str, int]:
    """Make `seq` a whole number of k-mers, or reject it. Returns `(seq, adjusted_bases)`.

    - ``"reject"`` (default everywhere): raise a `ValueError` naming the remainder.
    - ``"trim_left"``: drop the first ``len % k`` bases -- the authors' recipe
      (``seq[len(seq) % 6:]`` in the model cards). The count is reported back.
    - ``"pad_left"``: prepend ``k - len % k`` ``A`` bases -- the authors' other
      recommended option for generation prompts. The count is reported back.

    Without one of these, the stock tokenizer would turn the leftover bases into a
    single uninformative ``<oov>`` token, which the authors warn against.
    """
    if policy not in REMAINDER_POLICIES:
        raise ValueError(f"remainder must be one of {list(REMAINDER_POLICIES)}; got {policy!r}.")
    r = len(seq) % k
    if r == 0:
        return seq, 0
    if policy == "trim_left":
        if len(seq) < k:
            raise ValueError(
                f"Sequence of {len(seq)} bp is shorter than one {k}-mer; remainder='trim_left' "
                "would leave nothing. Use remainder='pad_left' or a longer sequence."
            )
        return seq[r:], r
    if policy == "pad_left":
        return "A" * (k - r) + seq, k - r
    raise ValueError(
        f"GENERator reads DNA as non-overlapping {k}-mers, and this sequence is {len(seq)} bp "
        f"({len(seq)} % {k} = {r}). Pass a length that is a multiple of {k}, or set "
        f"remainder='trim_left' (drop the first {r} bases, as the GENERator authors do) or "
        f"remainder='pad_left' (prepend {k - r} 'A' bases). Bases are never dropped silently."
    )
