"""Build physical DNA strands and duplexes from sequences.

This is the bridge between the informational layer (:mod:`dna_sim`, where DNA is
a string of bases) and the physical layer (:mod:`atomsim`, where matter is atoms
moving under forces). A sequence becomes a chain of coarse-grained base beads
joined by a backbone; two complementary strands become a duplex whose base
pairs are held by the specific pairing interaction, so Watson-Crick pairing is
something the dynamics *does*, not something we assert.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from atomsim.atom import Atom
from atomsim.elements import Element
from atomsim.forcefield import Bond
from atomsim.system import MolecularSystem
from atomsim.vector import Vec3
from dna_sim.sequence import DNASequence

from .pairing import (
    BasePair,
    pair_strength,
    pairing_forces_factory,
)

# One coarse-grained bead per nucleotide. The mass is an effective
# coarse-grained value (not the literal ~330 amu of a nucleotide) chosen so the
# reduced dynamics relax on a convenient timescale; a modest excluded-volume
# size keeps beads apart without overwhelming the base-pairing well that binds
# partners at ~3 Å.
BASE_BEAD = Element(
    symbol="Nb",
    name="NucleotideBead",
    atomic_number=0,
    mass=100.0,
    lj_sigma=5.0,
    lj_epsilon=1.0,
    covalent_radius=3.0,
)

# Geometry of the coarse-grained ladder (Å).
BACKBONE_SPACING = 6.0  # distance between consecutive beads along a strand
BACKBONE_STIFFNESS = 200.0  # kJ/(mol·Å²)


@dataclass
class Duplex:
    """A two-strand DNA system laid out as a physical ladder of base pairs."""

    system: MolecularSystem
    bases: list[str]  # base identity per atom index
    pairs: list[BasePair]  # candidate Watson-Crick pairs across the strands
    strand_a: list[int]  # atom indices of strand A, 5'->3'
    strand_b: list[int]  # atom indices of strand B, positioned across from A

    def mean_pair_distance(self) -> float:
        atoms = self.system.atoms
        if not self.pairs:
            return 0.0
        total = sum(
            atoms[p.i].position.distance_to(atoms[p.j].position)
            for p in self.pairs
        )
        return total / len(self.pairs)

    def paired_fraction(self, tolerance: float = 1.0) -> float:
        """Fraction of base pairs currently within ``tolerance`` Å of their
        equilibrium pairing distance."""
        atoms = self.system.atoms
        if not self.pairs:
            return 0.0
        bound = 0
        for p in self.pairs:
            r = atoms[p.i].position.distance_to(atoms[p.j].position)
            if abs(r - p.r0) <= tolerance:
                bound += 1
        return bound / len(self.pairs)

    def pairing_energy(self) -> float:
        return self.system.energy_breakdown().get("base_pairing", 0.0)


def _make_beads(
    bases: str, y: float, index_offset: int
) -> tuple[list[Atom], list[Bond], list[int]]:
    atoms: list[Atom] = []
    bonds: list[Bond] = []
    indices: list[int] = []
    for k, base in enumerate(bases):
        atoms.append(
            Atom(element=BASE_BEAD, position=Vec3(k * BACKBONE_SPACING, y, 0.0))
        )
        indices.append(index_offset + k)
        if k > 0:
            bonds.append(
                Bond(
                    index_offset + k - 1,
                    index_offset + k,
                    length=BACKBONE_SPACING,
                    stiffness=BACKBONE_STIFFNESS,
                )
            )
    return atoms, bonds, indices


def build_duplex(
    sequence: str | DNASequence,
    partner: str | DNASequence | None = None,
    separation: float = 6.0,
    damping: float = 1.5,
) -> Duplex:
    """Assemble a physical duplex from a sequence and a partner strand.

    Strand A is ``sequence`` (5'->3'). Strand B is ``partner`` (also given
    5'->3') if provided, otherwise the Watson-Crick reverse complement of A.
    Because DNA strands are *antiparallel*, strand B is laid down in reverse
    across the ladder, so A's 5' end pairs with B's 3' end. Strand B beads sit
    ``separation`` Å from their partners, so the duplex starts apart and the
    pairing interaction must pull it together. ``damping`` applies a viscous
    drag so the system relaxes rather than oscillating forever.

    Pairing strength for each rung comes from the *actual* base identities, so a
    non-complementary ``partner`` yields weak or absent pairing — specificity is
    not enforced by hand, it falls out of :func:`~nucleic.pairing.pair_strength`.
    """
    seq_a = DNASequence(str(sequence))
    bases_a = seq_a.bases
    if partner is None:
        partner_seq = str(seq_a.reverse_complement())
    else:
        partner_seq = DNASequence(str(partner)).bases
    if len(partner_seq) != len(bases_a):
        raise ValueError("partner strand must be the same length as the sequence")

    # Antiparallel: reverse the partner so ladder rung k pairs A[k] with the
    # partner base counted from its opposite (3') end.
    bases_b = partner_seq[::-1]

    atoms_a, bonds_a, idx_a = _make_beads(bases_a, y=0.0, index_offset=0)
    atoms_b, bonds_b, idx_b = _make_beads(
        bases_b, y=separation, index_offset=len(atoms_a)
    )
    atoms = atoms_a + atoms_b
    bonds = bonds_a + bonds_b
    bases = list(bases_a) + list(bases_b)

    pairs: list[BasePair] = []
    exclusions: set[frozenset[int]] = set()
    for k in range(len(bases_a)):
        i, j = idx_a[k], idx_b[k]
        strength = pair_strength(bases_a[k], bases_b[k])
        pairs.append(BasePair(i=i, j=j, strength=strength))
        # The pairing well governs this pair's short range; keep LJ out of it.
        exclusions.add(frozenset((i, j)))

    system = MolecularSystem(
        atoms=atoms,
        bonds=bonds,
        extra_forces=pairing_forces_factory(pairs),
        exclusions=exclusions,
        damping=damping,
    )
    return Duplex(
        system=system,
        bases=bases,
        pairs=pairs,
        strand_a=idx_a,
        strand_b=idx_b,
    )
