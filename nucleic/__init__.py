"""nucleic — the bridge from atoms to DNA.

Turns DNA sequences (the informational layer, :mod:`dna_sim`) into physical
structures of coarse-grained base beads (the physical layer, :mod:`atomsim`),
where Watson-Crick base pairing emerges from a specific hydrogen-bond-like
interaction between complementary bases. This is the rung that connects
"DNA as a string" to "DNA as matter in motion".
"""

from .pairing import (
    COMPLEMENT,
    PAIR_STRENGTH,
    BasePair,
    are_complementary,
    pair_strength,
    pairing_forces_factory,
    pairing_pair,
)
from .replication import (
    ReplicationResult,
    replicate_template,
    select_complement,
)
from .strand import BASE_BEAD, Duplex, build_duplex

__all__ = [
    "COMPLEMENT",
    "PAIR_STRENGTH",
    "BasePair",
    "are_complementary",
    "pair_strength",
    "pairing_pair",
    "pairing_forces_factory",
    "BASE_BEAD",
    "Duplex",
    "build_duplex",
    "ReplicationResult",
    "replicate_template",
    "select_complement",
]
