"""Template-directed replication, emerging from the physical substrate.

This is the rung the whole project has been climbing toward: a DNA strand
copying itself through physics. A template strand is held in place; free base
monomers are recruited one at a time to the growing end, and two things happen
*by physics*, not by fiat:

* **Fidelity is emergent.** Which base is added is decided by pairing affinity
  (:func:`~nucleic.pairing.pair_strength`) — the monomer that hydrogen-bonds to
  the exposed template base wins. This reproduces Watson-Crick complementarity
  from the interaction, not a hard-coded complement table.
* **Assembly is emergent.** Each recruited monomer physically docks against its
  template base under the pairing force and is stitched to the previous monomer
  by a backbone bond — processive polymerisation, the way a polymerase extends a
  strand one nucleotide at a time.

The assembled daughter is then checked against :mod:`dna_sim`: the physical copy
reproduces the informational complement operation, closing the loop between
matter and information.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from atomsim.atom import Atom
from atomsim.forcefield import Bond
from atomsim.system import MolecularSystem
from atomsim.vector import Vec3
from dna_sim.sequence import DNASequence

from .pairing import BasePair, pair_strength, pairing_forces_factory
from .strand import BACKBONE_SPACING, BACKBONE_STIFFNESS, BASE_BEAD

BASES = ("A", "C", "G", "T")


def select_complement(template_base: str) -> str:
    """Choose the incoming base for a template base by pairing affinity.

    Returns whichever of A/C/G/T binds the template base most strongly — the
    Watson-Crick complement — so fidelity comes out of the physics of pairing
    rather than a hand-written lookup.
    """
    return max(BASES, key=lambda b: pair_strength(template_base, b))


def select_base_thermal(
    template_base: str, temperature: float, rng: random.Random
) -> str:
    """Choose the incoming base by a thermodynamic competition at the active site.

    Each candidate base is weighted by a Boltzmann factor in its pairing energy,
    ``exp(pair_strength / temperature)``, so the well-matched (complementary)
    base is favoured but thermal energy can drive a mismatch. ``temperature`` is
    the thermal energy scale (kJ/mol), comparable to the pairing strengths
    (~16–24): near zero it is deterministic and perfect; as it approaches and
    exceeds the pairing strengths, misincorporation — i.e. mutation — emerges.

    This is the physical origin of replication error: fidelity is not perfect,
    it is a temperature-dependent balance between pairing energy and thermal
    noise.
    """
    if temperature <= 0.0:
        return select_complement(template_base)
    strengths = [pair_strength(template_base, b) for b in BASES]
    top = max(strengths)  # subtract the max for numerical stability
    weights = [math.exp((s - top) / temperature) for s in strengths]
    return rng.choices(BASES, weights=weights, k=1)[0]


def replicate_sequence(
    template: str | DNASequence,
    temperature: float = 0.0,
    rng: random.Random | None = None,
) -> str:
    """Replicate a template at the sequence level with thermal fidelity.

    A fast path that runs the same base-selection physics as
    :func:`replicate_template` without the molecular-dynamics docking, for when
    only the (possibly mutated) daughter sequence is needed — e.g. measuring how
    mutation rate depends on temperature, or feeding physical replication errors
    into the evolution layer. Returns the daughter in template-aligned
    (complement) order.
    """
    tseq = DNASequence(str(template))
    r = rng if rng is not None else random.Random()
    return "".join(select_base_thermal(b, temperature, r) for b in tseq.bases)


def replication_error_rate(
    template: str | DNASequence,
    temperature: float,
    trials: int,
    rng: random.Random,
) -> float:
    """Empirical per-base misincorporation rate at a given temperature.

    Replicates ``template`` ``trials`` times and reports the fraction of
    positions that differ from the perfect Watson-Crick complement.
    """
    tseq = DNASequence(str(template))
    perfect = str(tseq.complement())
    n = len(perfect)
    errors = 0
    for _ in range(trials):
        daughter = replicate_sequence(tseq, temperature, rng)
        errors += sum(1 for a, b in zip(daughter, perfect) if a != b)
    return errors / (trials * n)


@dataclass
class ReplicationResult:
    """The outcome of replicating a template strand."""

    template: str
    daughter: str  # newly synthesised strand, in template-aligned order
    system: MolecularSystem
    template_idx: list[int]
    daughter_idx: list[int]
    pairs: list[BasePair]

    def mean_pair_distance(self) -> float:
        atoms = self.system.atoms
        return sum(
            atoms[p.i].position.distance_to(atoms[p.j].position) for p in self.pairs
        ) / len(self.pairs)

    def paired_fraction(self, tolerance: float = 1.0) -> float:
        atoms = self.system.atoms
        bound = sum(
            1
            for p in self.pairs
            if abs(atoms[p.i].position.distance_to(atoms[p.j].position) - p.r0)
            <= tolerance
        )
        return bound / len(self.pairs)

    def as_daughter_strand(self) -> DNASequence:
        """The daughter read in its own 5'->3' direction (antiparallel)."""
        return DNASequence(self.daughter[::-1])


def replicate_template(
    template: str | DNASequence,
    separation: float = 6.0,
    damping: float = 1.5,
    steps_per_base: int = 1500,
    final_relax_steps: int | None = None,
    dt: float = 0.01,
    temperature: float = 0.0,
    rng: random.Random | None = None,
) -> ReplicationResult:
    """Physically replicate a template strand, one nucleotide at a time.

    The template is built as a pinned chain of base beads. For each template
    position the complementary monomer is selected by pairing affinity, placed
    near the growing end, bonded to the previous monomer, and allowed to dock
    under the force field. The result reports the assembled daughter strand and
    the physical system in which it is paired to the template.
    """
    tseq = DNASequence(str(template))
    tbases = tseq.bases
    n = len(tbases)
    if n == 0:
        raise ValueError("cannot replicate an empty template")

    atoms: list[Atom] = []
    bonds: list[Bond] = []
    pairs: list[BasePair] = []
    exclusions: set[frozenset[int]] = set()

    # Template strand: a pinned ladder along x.
    for k in range(n):
        atoms.append(
            Atom(element=BASE_BEAD, position=Vec3(k * BACKBONE_SPACING, 0.0, 0.0))
        )
        if k > 0:
            bonds.append(
                Bond(k - 1, k, length=BACKBONE_SPACING, stiffness=BACKBONE_STIFFNESS)
            )
    template_idx = list(range(n))
    fixed = set(template_idx)

    # The pairing force closes over the (growing) pairs list, so appending a new
    # pair as each monomer arrives extends the interaction automatically.
    system = MolecularSystem(
        atoms=atoms,
        bonds=bonds,
        extra_forces=pairing_forces_factory(pairs),
        exclusions=exclusions,
        fixed=fixed,
        damping=damping,
    )

    r = rng if rng is not None else random.Random()
    daughter_idx: list[int] = []
    daughter_chars: list[str] = []
    for k in range(n):
        # Fidelity from physics: pairing affinity vs thermal noise. At
        # temperature 0 this is the deterministic Watson-Crick complement.
        base = select_base_thermal(tbases[k], temperature, r)
        daughter_chars.append(base)
        new_index = len(atoms)
        # Deliver the monomer near its template site (as a polymerase would
        # position the incoming nucleotide at the active site); pairing docks it.
        atoms.append(
            Atom(
                element=BASE_BEAD,
                position=Vec3(k * BACKBONE_SPACING, separation, 0.0),
            )
        )
        daughter_idx.append(new_index)
        if k > 0:  # stitch to the previous monomer: processive extension
            bonds.append(
                Bond(
                    daughter_idx[k - 1],
                    new_index,
                    length=BACKBONE_SPACING,
                    stiffness=BACKBONE_STIFFNESS,
                )
            )
        pairs.append(
            BasePair(i=k, j=new_index, strength=pair_strength(tbases[k], base))
        )
        exclusions.add(frozenset((k, new_index)))
        system.run(steps=steps_per_base, dt=dt)

    # Final relaxation: let the whole assembled duplex settle. Later monomers
    # get less docking time during assembly, so scale this with strand length
    # by default. Pass 0 to skip when only the daughter sequence is needed.
    relax = max(4000, 1200 * n) if final_relax_steps is None else final_relax_steps
    if relax > 0:
        system.run(steps=relax, dt=dt)

    return ReplicationResult(
        template=tbases,
        daughter="".join(daughter_chars),
        system=system,
        template_idx=template_idx,
        daughter_idx=daughter_idx,
        pairs=pairs,
    )
