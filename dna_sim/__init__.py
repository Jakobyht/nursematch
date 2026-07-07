"""dna_sim — a small, dependency-free DNA simulation toolkit.

It models DNA sequences and the core molecular processes acting on them:
complementation, transcription, translation, mutation, replication, and
population-level evolution under selection.
"""

from .evolution import (
    GenerationStats,
    Population,
    gc_target_fitness,
)
from .generate import random_sequence
from .mutation import MutationEvent, MutationModel, delete, insert, substitute
from .replication import ReplicationResult, replicate
from .sequence import (
    BASES,
    CODON_TABLE,
    DNASequence,
    InvalidSequenceError,
)

__version__ = "0.1.0"

__all__ = [
    "BASES",
    "CODON_TABLE",
    "DNASequence",
    "InvalidSequenceError",
    "MutationEvent",
    "MutationModel",
    "substitute",
    "insert",
    "delete",
    "ReplicationResult",
    "replicate",
    "GenerationStats",
    "Population",
    "gc_target_fitness",
    "random_sequence",
    "__version__",
]
