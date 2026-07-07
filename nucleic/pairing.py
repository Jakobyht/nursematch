"""Base-pairing as a physical interaction.

Watson-Crick pairing is, physically, hydrogen bonding between complementary
bases: adenine-thymine (two H-bonds) and guanine-cytosine (three, hence
stronger). We model each base as a coarse-grained bead and represent pairing as
a specific attractive well between complementary beads — the level of
abstraction real coarse-grained DNA models (e.g. oxDNA) use so that pairing and
helix formation *emerge* from the dynamics at a tractable scale.

The well is a Morse potential: an energy minimum of depth ``-strength`` at
separation ``r0``, a repulsive wall when too close, and a long-range attractive
tail so two strands starting apart can actually *find* each other and zip
together. Non-complementary beads feel no pairing attraction (only the
excluded-volume repulsion from the underlying force field), which is what makes
pairing *specific*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from atomsim.atom import Atom
from atomsim.vector import ZERO, Vec3

# Watson-Crick complement.
COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}

# Relative H-bond strengths: G-C (3 bonds) is stronger than A-T (2 bonds).
# Values in kJ/mol, the depth of the pairing well.
PAIR_STRENGTH = {
    frozenset(("A", "T")): 16.0,
    frozenset(("G", "C")): 24.0,
}

# Preferred pairing separation between base beads (Å), coarse-grained.
PAIR_DISTANCE = 3.0
# Morse steepness (1/Å): smaller = longer-range attraction. Tuned so strands a
# few Å apart still feel each other and can zip together.
PAIR_ALPHA = 0.8


def are_complementary(a: str, b: str) -> bool:
    return COMPLEMENT.get(a) == b


def pair_strength(a: str, b: str) -> float:
    """Depth of the pairing well for two bases (0 if not complementary)."""
    return PAIR_STRENGTH.get(frozenset((a, b)), 0.0)


def pairing_pair(
    atom_a: Atom,
    atom_b: Atom,
    strength: float,
    r0: float = PAIR_DISTANCE,
    alpha: float = PAIR_ALPHA,
) -> tuple[float, Vec3]:
    """Energy and the force on ``atom_a`` from a Morse pairing well with ``atom_b``.

    The Morse potential is ``U(r) = strength·((1 - e^{-alpha(r-r0)})² - 1)``,
    with minimum ``-strength`` at ``r0``, a repulsive wall for ``r < r0`` and a
    long-range attractive tail for ``r > r0``.

    Returns ``(0, 0)`` when ``strength`` is zero (non-complementary bases).
    The force on ``atom_b`` is the negative of the returned vector.
    """
    if strength == 0.0:
        return 0.0, ZERO
    delta = atom_a.position - atom_b.position
    r = delta.norm()
    if r == 0.0:
        return 0.0, ZERO
    x = math.exp(-alpha * (r - r0))
    energy = strength * ((1.0 - x) ** 2 - 1.0)
    # dU/dr = 2·strength·alpha·x·(1 - x);  F_a = -dU/dr · (delta/r)
    dU_dr = 2.0 * strength * alpha * x * (1.0 - x)
    scale = -dU_dr / r
    return energy, delta * scale


@dataclass
class BasePair:
    """A candidate pairing interaction between two base beads (by index)."""

    i: int
    j: int
    strength: float
    r0: float = PAIR_DISTANCE
    alpha: float = PAIR_ALPHA


def pairing_forces_factory(pairs: list[BasePair]):
    """Build an ``extra_forces`` callable computing all pairing interactions.

    The returned function matches :data:`atomsim.system.ExtraForces` and can be
    handed straight to a :class:`~atomsim.system.MolecularSystem`.
    """

    def extra_forces(atoms: list[Atom]) -> tuple[list[Vec3], dict[str, float]]:
        forces = [ZERO for _ in atoms]
        total = 0.0
        for p in pairs:
            energy, force = pairing_pair(
                atoms[p.i], atoms[p.j], p.strength, p.r0, p.alpha
            )
            total += energy
            forces[p.i] = forces[p.i] + force
            forces[p.j] = forces[p.j] - force
        return forces, {"base_pairing": total}

    return extra_forces
