"""atomsim — the physical substrate: simulating atoms and matter.

Classical molecular dynamics from the ground up. Atoms are point masses that
interact through a force field (Lennard-Jones, Coulomb, harmonic bonds) and
move under Newton's equations integrated with velocity Verlet.

This is the physical layer of the project. The goal is to build upward from
here — atoms -> molecules -> macromolecules -> DNA -> life — connecting to the
informational layer in :mod:`dna_sim`.
"""

from .atom import Atom
from .elements import ELEMENTS, Element, element, lorentz_berthelot
from .forcefield import (
    Bond,
    compute_forces,
    coulomb_pair,
    harmonic_bond,
    lennard_jones_pair,
)
from .molecules import hydrogen, methane, water
from .system import BOLTZMANN, MolecularSystem
from .vector import Vec3

__all__ = [
    "Atom",
    "Element",
    "ELEMENTS",
    "element",
    "lorentz_berthelot",
    "Bond",
    "compute_forces",
    "lennard_jones_pair",
    "coulomb_pair",
    "harmonic_bond",
    "MolecularSystem",
    "BOLTZMANN",
    "Vec3",
    "water",
    "hydrogen",
    "methane",
]
