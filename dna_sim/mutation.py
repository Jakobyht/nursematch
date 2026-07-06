"""Mutation operators acting on DNA sequences.

All operators take a random-number generator so that simulations are
reproducible when seeded.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .sequence import BASES, DNASequence


@dataclass(frozen=True)
class MutationEvent:
    """A single mutation applied to a sequence."""

    kind: str  # "substitution", "insertion" or "deletion"
    position: int
    before: str
    after: str


def substitute(
    seq: DNASequence, position: int, rng: random.Random
) -> tuple[DNASequence, MutationEvent]:
    """Replace the base at ``position`` with a different random base."""
    original = seq.bases[position]
    alternatives = [b for b in BASES if b != original]
    new_base = rng.choice(alternatives)
    mutated = seq.bases[:position] + new_base + seq.bases[position + 1 :]
    event = MutationEvent("substitution", position, original, new_base)
    return DNASequence(mutated), event


def insert(
    seq: DNASequence, position: int, rng: random.Random
) -> tuple[DNASequence, MutationEvent]:
    """Insert a random base before ``position``."""
    new_base = rng.choice(BASES)
    mutated = seq.bases[:position] + new_base + seq.bases[position:]
    event = MutationEvent("insertion", position, "", new_base)
    return DNASequence(mutated), event


def delete(
    seq: DNASequence, position: int, rng: random.Random
) -> tuple[DNASequence, MutationEvent]:
    """Delete the base at ``position``."""
    original = seq.bases[position]
    mutated = seq.bases[:position] + seq.bases[position + 1 :]
    event = MutationEvent("deletion", position, original, "")
    return DNASequence(mutated), event


@dataclass(frozen=True)
class MutationModel:
    """Per-base probabilities for each class of mutation during replication.

    The probabilities are applied independently at each base position, so the
    expected number of mutations scales with sequence length.
    """

    substitution_rate: float = 1e-3
    insertion_rate: float = 1e-4
    deletion_rate: float = 1e-4

    def __post_init__(self) -> None:
        total = self.substitution_rate + self.insertion_rate + self.deletion_rate
        if not 0.0 <= total <= 1.0:
            raise ValueError(
                "combined mutation rate must be between 0 and 1, got "
                f"{total}"
            )
        for name, rate in (
            ("substitution_rate", self.substitution_rate),
            ("insertion_rate", self.insertion_rate),
            ("deletion_rate", self.deletion_rate),
        ):
            if rate < 0:
                raise ValueError(f"{name} must be non-negative, got {rate}")

    def apply(
        self, seq: DNASequence, rng: random.Random
    ) -> tuple[DNASequence, list[MutationEvent]]:
        """Apply the model across a sequence, returning the mutated copy.

        Positions are processed left to right on the original coordinate frame.
        To keep indices stable under insertions and deletions, mutations are
        collected first and then applied to a fresh list of bases.
        """
        bases = list(seq.bases)
        events: list[MutationEvent] = []
        # Build the result base by base so insert/delete don't shift positions
        # we have not yet visited.
        result: list[str] = []
        for i, base in enumerate(bases):
            roll = rng.random()
            if roll < self.substitution_rate:
                alternatives = [b for b in BASES if b != base]
                new_base = rng.choice(alternatives)
                result.append(new_base)
                events.append(MutationEvent("substitution", i, base, new_base))
            elif roll < self.substitution_rate + self.deletion_rate:
                events.append(MutationEvent("deletion", i, base, ""))
                # base is dropped
            elif (
                roll
                < self.substitution_rate
                + self.deletion_rate
                + self.insertion_rate
            ):
                new_base = rng.choice(BASES)
                result.append(new_base)
                result.append(base)
                events.append(MutationEvent("insertion", i, "", new_base))
            else:
                result.append(base)
        return DNASequence("".join(result)), events
