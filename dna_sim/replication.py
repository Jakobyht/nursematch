"""Semiconservative DNA replication with proofreading errors."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .mutation import MutationEvent, MutationModel
from .sequence import DNASequence


@dataclass
class ReplicationResult:
    """The outcome of copying a template strand."""

    template: DNASequence
    copy: DNASequence
    events: list[MutationEvent] = field(default_factory=list)

    @property
    def mutated(self) -> bool:
        return bool(self.events)


def replicate(
    template: DNASequence,
    model: MutationModel,
    rng: random.Random,
) -> ReplicationResult:
    """Produce a daughter strand from ``template`` under a mutation model.

    Replication is modelled directly on the given strand (5'->3'); the copy is
    identical to the template except where polymerase errors, captured by the
    mutation model, introduce changes.
    """
    copy, events = model.apply(template, rng)
    return ReplicationResult(template=template, copy=copy, events=events)
