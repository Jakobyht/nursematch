"""Utilities for generating random DNA sequences."""

from __future__ import annotations

import random

from .sequence import BASES, DNASequence


def random_sequence(
    length: int,
    rng: random.Random,
    gc_content: float | None = None,
) -> DNASequence:
    """Generate a random DNA sequence of the given length.

    When ``gc_content`` is provided (0..1), G/C bases are drawn with that
    probability and A/T bases share the remainder; otherwise all four bases are
    equally likely.
    """
    if length < 0:
        raise ValueError("length must be non-negative")
    if gc_content is None:
        bases = [rng.choice(BASES) for _ in range(length)]
    else:
        if not 0.0 <= gc_content <= 1.0:
            raise ValueError("gc_content must be between 0 and 1")
        gc = ("G", "C")
        at = ("A", "T")
        bases = [
            rng.choice(gc) if rng.random() < gc_content else rng.choice(at)
            for _ in range(length)
        ]
    return DNASequence("".join(bases))
