"""Self-assembling amphiphile membranes — compartments from physics.

A cell needs a boundary. Real lipids build one by themselves: their water-loving
heads face the solvent while their water-fearing tails hide from it, and this
*hydrophobic effect* drives dispersed molecules to assemble into micelles,
bilayers and vesicles with no template.

We model it the way coarse-grained membrane simulations do (Cooke-Deserno style,
implicit solvent):

* every bead has excluded volume — a purely repulsive WCA interaction;
* tail (hydrophobic) beads additionally *attract* each other, standing in for
  the free-energy cost of exposing them to water.

Heads carry no cohesion, so once tails clump the heads are pushed to the
outside. Compartmentalisation emerges from the interaction, not from a rule.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from atomsim.atom import Atom
from atomsim.elements import Element
from atomsim.forcefield import Bond
from atomsim.system import MolecularSystem
from atomsim.vector import ZERO, Vec3

# Coarse-grained beads. lj_epsilon is 0 so the built-in Lennard-Jones is inert;
# all non-bonded interaction comes from the membrane force below.
HEAD = Element("Hd", "AmphiphileHead", 0, mass=30.0, lj_sigma=0.0,
               lj_epsilon=0.0, covalent_radius=0.0)
TAIL = Element("Tl", "AmphiphileTail", 0, mass=30.0, lj_sigma=0.0,
               lj_epsilon=0.0, covalent_radius=0.0)

BEAD_SIGMA = 4.0            # bead diameter (Å)
BOND_LENGTH = 1.12 * BEAD_SIGMA  # ~ WCA minimum, so bonded beads don't clash
BOND_STIFFNESS = 200.0
_RMIN2 = (2.0 ** (1.0 / 3.0)) * BEAD_SIGMA * BEAD_SIGMA  # (2^(1/6)·sigma)^2


def membrane_forces_factory(
    hydrophobic: set[int],
    sigma: float = BEAD_SIGMA,
    eps_attract: float = 20.0,
    eps_repel: float = 5.0,
    attract_range: float = 0.5,
    cutoff: float | None = None,
    max_force: float = 200.0,
):
    """Build the membrane interaction as an ``extra_forces`` callable.

    Tail-tail pairs (both indices in ``hydrophobic``) attract through a Morse
    well of depth ``eps_attract`` centred at contact (``sigma``), with a
    long-range tail set by ``attract_range`` (smaller = longer reach) and an
    inherent repulsive wall giving the tails their own excluded volume. This
    long-ranged cohesion is what lets dispersed, overdamped amphiphiles find one
    another and coalesce. Every other pair feels only WCA repulsion.

    Pairwise force magnitude is capped at ``max_force`` so incidental overlaps in
    the random starting configuration relax smoothly instead of exploding.
    """
    cutoff_r = cutoff if cutoff is not None else 6.0 * sigma
    cutoff2 = cutoff_r * cutoff_r
    sigma2 = sigma * sigma

    def forces(atoms: list[Atom]) -> tuple[list[Vec3], dict[str, float]]:
        n = len(atoms)
        f = [ZERO for _ in range(n)]
        energy = 0.0
        for i in range(n):
            hi = i in hydrophobic
            for j in range(i + 1, n):
                delta = atoms[i].position - atoms[j].position
                r2 = delta.norm_sq()
                if r2 == 0.0:
                    continue
                both_tail = hi and (j in hydrophobic)
                if both_tail:
                    if r2 > cutoff2:
                        continue
                    r = math.sqrt(r2)
                    x = math.exp(-attract_range * (r - sigma))
                    energy += eps_attract * ((1.0 - x) ** 2 - 1.0)
                    dU_dr = 2.0 * eps_attract * attract_range * x * (1.0 - x)
                    scale = -dU_dr / r
                else:
                    if r2 >= _RMIN2:
                        continue  # WCA is zero beyond its minimum
                    inv = sigma2 / r2
                    inv6 = inv ** 3
                    inv12 = inv6 ** 2
                    energy += 4.0 * eps_repel * (inv12 - inv6) + eps_repel
                    scale = 24.0 * eps_repel * (2.0 * inv12 - inv6) / r2
                pair = delta * scale
                mag = pair.norm()
                if mag > max_force:
                    pair = pair * (max_force / mag)
                f[i] = f[i] + pair
                f[j] = f[j] - pair
        return f, {"membrane": energy}

    return forces


@dataclass
class Membrane:
    """A collection of amphiphiles evolving under the membrane force."""

    system: MolecularSystem
    head_idx: list[int]
    tail_idx: list[int]
    n_amphiphiles: int

    def centroid(self) -> Vec3:
        pts = self.system.atoms
        total = Vec3()
        for a in pts:
            total = total + a.position
        return total / len(pts)

    def _mean_radius(self, indices: list[int]) -> float:
        c = self.centroid()
        atoms = self.system.atoms
        return sum(atoms[i].position.distance_to(c) for i in indices) / len(indices)

    def head_radius(self) -> float:
        return self._mean_radius(self.head_idx)

    def tail_radius(self) -> float:
        return self._mean_radius(self.tail_idx)

    def _radius_of_gyration(self, indices: list[int]) -> float:
        """Spread of a set of beads about *their own* centroid (shape-robust)."""
        atoms = self.system.atoms
        c = Vec3()
        for i in indices:
            c = c + atoms[i].position
        c = c / len(indices)
        msd = sum(atoms[i].position.distance_to(c) ** 2 for i in indices) / len(indices)
        return msd ** 0.5

    def head_gyration(self) -> float:
        return self._radius_of_gyration(self.head_idx)

    def tail_gyration(self) -> float:
        return self._radius_of_gyration(self.tail_idx)

    def has_hydrophobic_core(self) -> bool:
        """True when the tails are packed more tightly than the heads — a
        hydrophobic core wrapped in a hydrophilic surface, the signature of a
        micelle/membrane and shape-robust (uses each group's own spread)."""
        return self.tail_gyration() < self.head_gyration()

    def is_micelle(self) -> bool:
        """Heads farther from the aggregate centre than tails."""
        return self.head_radius() > self.tail_radius()


def build_membrane(
    n_amphiphiles: int = 10,
    tails_per_amphiphile: int = 2,
    spread: float = 6.0,
    damping: float = 1.0,
    rng: random.Random | None = None,
) -> Membrane:
    """Scatter amphiphiles in a plane, ready to self-assemble.

    Each amphiphile is a chain HEAD-TAIL(-TAIL...) placed at a random position
    and orientation within a disk of radius ``spread``. Motion stays in the
    z = 0 plane. Run the returned system to watch a micelle form.
    """
    r = rng if rng is not None else random.Random()
    atoms: list[Atom] = []
    bonds: list[Bond] = []
    hydrophobic: set[int] = set()
    head_idx: list[int] = []
    tail_idx: list[int] = []

    for _ in range(n_amphiphiles):
        # Random centre in a disk, random in-plane orientation.
        cx = r.uniform(-spread, spread)
        cy = r.uniform(-spread, spread)
        angle = r.uniform(0, 2.0 * math.pi)
        dx, dy = -BOND_LENGTH * math.cos(angle), -BOND_LENGTH * math.sin(angle)

        base = len(atoms)
        atoms.append(Atom(element=HEAD, position=Vec3(cx, cy, 0.0)))
        head_idx.append(base)
        prev = base
        for t in range(tails_per_amphiphile):
            pos = Vec3(cx + dx * (t + 1), cy + dy * (t + 1), 0.0)
            idx = len(atoms)
            atoms.append(Atom(element=TAIL, position=pos))
            hydrophobic.add(idx)
            tail_idx.append(idx)
            bonds.append(Bond(prev, idx, length=BOND_LENGTH, stiffness=BOND_STIFFNESS))
            prev = idx

    system = MolecularSystem(
        atoms=atoms,
        bonds=bonds,
        extra_forces=membrane_forces_factory(hydrophobic),
        damping=damping,
    )
    return Membrane(
        system=system,
        head_idx=head_idx,
        tail_idx=tail_idx,
        n_amphiphiles=n_amphiphiles,
    )
