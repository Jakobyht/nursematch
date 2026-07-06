"""The force field: how atoms push and pull on each other.

Three interactions, together enough to hold molecules together and let them
move realistically:

* Lennard-Jones  — van der Waals: strong repulsion when atoms overlap, weak
  attraction at medium range. Governs non-bonded contacts.
* Coulomb        — electrostatics between partial charges. Long-ranged; the
  reason opposite charges attract and like charges repel.
* Harmonic bond  — a spring between covalently bonded atoms, holding them near
  an equilibrium length.

Each function returns a potential energy and fills a per-atom force list. Units
are consistent with :mod:`atomsim.elements` (Å, amu, kJ/mol, charge in e).
"""

from __future__ import annotations

from dataclasses import dataclass

from .atom import Atom
from .elements import lorentz_berthelot
from .vector import ZERO, Vec3

# Coulomb constant in the MD unit system (kJ·Å / (mol·e²)).
# = 1 / (4·π·ε0) converted to these units.
COULOMB_CONSTANT = 1389.35458


@dataclass(frozen=True)
class Bond:
    """A harmonic covalent bond between two atoms (by index)."""

    i: int
    j: int
    length: float  # equilibrium length, Å
    stiffness: float = 1000.0  # kJ/(mol·Å²)


def lennard_jones_pair(a: Atom, b: Atom) -> tuple[float, Vec3]:
    """Energy and the force on ``a`` from the LJ interaction with ``b``.

    The force on ``b`` is the negative of the returned vector.
    """
    sigma, epsilon = lorentz_berthelot(a.element, b.element)
    delta = a.position - b.position
    r2 = delta.norm_sq()
    if r2 == 0.0:
        return 0.0, ZERO
    inv_r2 = (sigma * sigma) / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * epsilon * (inv_r12 - inv_r6)
    # magnitude of dU/dr folded into the vector along ``delta``:
    # F = 24·ε·(2·(σ/r)^12 - (σ/r)^6) / r² · delta
    scale = 24.0 * epsilon * (2.0 * inv_r12 - inv_r6) / r2
    return energy, delta * scale


def coulomb_pair(a: Atom, b: Atom) -> tuple[float, Vec3]:
    """Energy and the force on ``a`` from electrostatic interaction with ``b``."""
    if a.charge == 0.0 or b.charge == 0.0:
        return 0.0, ZERO
    delta = a.position - b.position
    r2 = delta.norm_sq()
    if r2 == 0.0:
        return 0.0, ZERO
    r = r2 ** 0.5
    energy = COULOMB_CONSTANT * a.charge * b.charge / r
    # F = k·qa·qb / r² along the unit vector = energy/r² · delta
    scale = energy / r2
    return energy, delta * scale


def harmonic_bond(a: Atom, b: Atom, bond: Bond) -> tuple[float, Vec3]:
    """Energy and the force on ``a`` from a harmonic bond to ``b``."""
    delta = a.position - b.position
    r = delta.norm()
    if r == 0.0:
        return 0.0, ZERO
    stretch = r - bond.length
    energy = 0.5 * bond.stiffness * stretch * stretch
    # F = -k·(r - r0)·(delta/r) on atom a
    scale = -bond.stiffness * stretch / r
    return energy, delta * scale


def compute_forces(
    atoms: list[Atom],
    bonds: list[Bond],
    nonbonded_cutoff: float | None = None,
    exclusions: set[frozenset[int]] | None = None,
) -> tuple[list[Vec3], dict[str, float]]:
    """Compute the total force on every atom and a breakdown of the energy.

    Non-bonded (LJ + Coulomb) interactions are summed over every unique pair;
    a distance cutoff can skip far-apart pairs. Bonded pairs are excluded from
    the non-bonded sum so the spring is not fighting van der Waals repulsion.
    Additional ``exclusions`` (as ``frozenset({i, j})`` index pairs) are also
    skipped — used when a specialised interaction, such as DNA base-pairing,
    governs a pair's short-range behaviour instead of Lennard-Jones.

    Returns ``(forces, energy)`` where ``energy`` has keys ``lennard_jones``,
    ``coulomb``, ``bond`` and ``potential`` (their sum).
    """
    n = len(atoms)
    forces = [ZERO for _ in range(n)]
    energy = {"lennard_jones": 0.0, "coulomb": 0.0, "bond": 0.0}
    bonded_pairs = {frozenset((b.i, b.j)) for b in bonds}
    if exclusions:
        bonded_pairs |= exclusions
    cutoff_sq = None if nonbonded_cutoff is None else nonbonded_cutoff ** 2

    for i in range(n):
        for j in range(i + 1, n):
            if frozenset((i, j)) in bonded_pairs:
                continue
            if cutoff_sq is not None:
                if (atoms[i].position - atoms[j].position).norm_sq() > cutoff_sq:
                    continue
            lj_e, lj_f = lennard_jones_pair(atoms[i], atoms[j])
            c_e, c_f = coulomb_pair(atoms[i], atoms[j])
            energy["lennard_jones"] += lj_e
            energy["coulomb"] += c_e
            pair_force = lj_f + c_f
            forces[i] = forces[i] + pair_force
            forces[j] = forces[j] - pair_force

    for bond in bonds:
        b_e, b_f = harmonic_bond(atoms[bond.i], atoms[bond.j], bond)
        energy["bond"] += b_e
        forces[bond.i] = forces[bond.i] + b_f
        forces[bond.j] = forces[bond.j] - b_f

    energy["potential"] = (
        energy["lennard_jones"] + energy["coulomb"] + energy["bond"]
    )
    return forces, energy
