"""Builders for a few small molecules, assembled from atoms and bonds.

These are the first rung of the ladder from atoms to life: concrete arrangements
of atoms held together by the force field. Geometries use approximate
equilibrium bond lengths and angles; the simulation relaxes them from there.
"""

from __future__ import annotations

import math

from .atom import Atom
from .forcefield import Bond
from .system import MolecularSystem


def water() -> MolecularSystem:
    """A single water molecule (H2O) with realistic geometry and charges.

    O-H bond length ~0.96 Å, H-O-H angle ~104.5°, TIP3P-like partial charges.
    """
    angle = math.radians(104.5)
    r = 0.96
    o = Atom.of("O", (0.0, 0.0, 0.0), charge=-0.834)
    h1 = Atom.of("H", (r, 0.0, 0.0), charge=0.417)
    h2 = Atom.of(
        "H",
        (r * math.cos(angle), r * math.sin(angle), 0.0),
        charge=0.417,
    )
    bonds = [
        Bond(0, 1, length=r, stiffness=1882.8),
        Bond(0, 2, length=r, stiffness=1882.8),
    ]
    return MolecularSystem(atoms=[o, h1, h2], bonds=bonds)


def hydrogen() -> MolecularSystem:
    """A hydrogen molecule (H2) with a ~0.74 Å bond."""
    r = 0.74
    atoms = [Atom.of("H", (0.0, 0.0, 0.0)), Atom.of("H", (r, 0.0, 0.0))]
    bonds = [Bond(0, 1, length=r, stiffness=3000.0)]
    return MolecularSystem(atoms=atoms, bonds=bonds)


def methane() -> MolecularSystem:
    """A methane molecule (CH4) with tetrahedral geometry, ~1.09 Å C-H bonds."""
    r = 1.09
    # Tetrahedral directions from the carbon at the origin.
    dirs = [
        (1, 1, 1),
        (1, -1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
    ]
    norm = math.sqrt(3.0)
    atoms = [Atom.of("C", (0.0, 0.0, 0.0), charge=-0.48)]
    bonds = []
    for k, (dx, dy, dz) in enumerate(dirs, start=1):
        pos = (r * dx / norm, r * dy / norm, r * dz / norm)
        atoms.append(Atom.of("H", pos, charge=0.12))
        bonds.append(Bond(0, k, length=r, stiffness=1422.0))
    return MolecularSystem(atoms=atoms, bonds=bonds)
