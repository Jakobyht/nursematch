"""protocell — building a cell: compartments from self-assembling membranes.

A living cell needs a boundary that forms and holds itself. This package models
amphiphiles (coarse-grained lipids) whose hydrophobic tails cohere while their
hydrophilic heads face outward, so membranes self-assemble from the physics —
the missing pillar alongside replication (:mod:`nucleic`) on the road to a
minimal protocell containing replicating, evolving DNA.
"""

from .membrane import (
    HEAD,
    TAIL,
    Membrane,
    build_membrane,
    membrane_forces_factory,
)

__all__ = [
    "HEAD",
    "TAIL",
    "Membrane",
    "build_membrane",
    "membrane_forces_factory",
]
