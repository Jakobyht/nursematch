"""A molecular system and its time evolution via velocity Verlet.

This is the heart of the physical simulation: given atoms, bonds, and a force
field, it steps Newton's equations forward in time. Everything above (DNA,
life) is meant to eventually rest on this substrate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .atom import Atom
from .forcefield import Bond, compute_forces
from .vector import Vec3

# Boltzmann constant in the MD unit system: kJ/(mol·K).
BOLTZMANN = 0.0083144621


@dataclass
class MolecularSystem:
    """A collection of atoms and bonds evolving under a force field."""

    atoms: list[Atom]
    bonds: list[Bond] = field(default_factory=list)
    nonbonded_cutoff: float | None = None
    time: float = 0.0

    def potential_energy(self) -> float:
        _, energy = compute_forces(self.atoms, self.bonds, self.nonbonded_cutoff)
        return energy["potential"]

    def kinetic_energy(self) -> float:
        return sum(a.kinetic_energy() for a in self.atoms)

    def total_energy(self) -> float:
        return self.potential_energy() + self.kinetic_energy()

    def energy_breakdown(self) -> dict[str, float]:
        """Full energy report including kinetic and total."""
        _, energy = compute_forces(self.atoms, self.bonds, self.nonbonded_cutoff)
        ke = self.kinetic_energy()
        energy["kinetic"] = ke
        energy["total"] = energy["potential"] + ke
        return energy

    def temperature(self) -> float:
        """Instantaneous temperature from the equipartition theorem.

        T = 2·KE / (dof · k_B), with dof = 3N (translational only; this simple
        model does not remove centre-of-mass motion).
        """
        n = len(self.atoms)
        if n == 0:
            return 0.0
        dof = 3 * n
        return 2.0 * self.kinetic_energy() / (dof * BOLTZMANN)

    def center_of_mass(self) -> Vec3:
        total_mass = sum(a.mass for a in self.atoms)
        if total_mass == 0.0:
            raise ValueError("system has zero total mass")
        weighted = Vec3()
        for a in self.atoms:
            weighted = weighted + a.position * a.mass
        return weighted / total_mass

    def step(self, dt: float) -> None:
        """Advance the system by one velocity-Verlet step of size ``dt``.

        Velocity Verlet:
            v(t+½dt) = v(t) + ½·a(t)·dt
            x(t+dt)  = x(t) + v(t+½dt)·dt
            recompute a(t+dt)
            v(t+dt)  = v(t+½dt) + ½·a(t+dt)·dt
        This is symplectic, so total energy stays bounded over long runs.
        """
        forces, _ = compute_forces(self.atoms, self.bonds, self.nonbonded_cutoff)
        half_dt = 0.5 * dt

        # First half-kick + drift.
        half_velocities: list[Vec3] = []
        for atom, force in zip(self.atoms, forces):
            accel = force / atom.mass
            v_half = atom.velocity + accel * half_dt
            half_velocities.append(v_half)
            atom.position = atom.position + v_half * dt

        # Recompute forces at the new positions and do the second half-kick.
        new_forces, _ = compute_forces(
            self.atoms, self.bonds, self.nonbonded_cutoff
        )
        for atom, force, v_half in zip(self.atoms, new_forces, half_velocities):
            accel = force / atom.mass
            atom.velocity = v_half + accel * half_dt

        self.time += dt

    def run(self, steps: int, dt: float, sample_every: int = 0) -> list[dict]:
        """Integrate for ``steps`` steps, optionally sampling energies.

        When ``sample_every`` > 0, an energy breakdown (with ``time``) is
        recorded every ``sample_every`` steps, plus one at the start.
        """
        samples: list[dict] = []
        if sample_every > 0:
            samples.append(self._sample())
        for k in range(1, steps + 1):
            self.step(dt)
            if sample_every > 0 and k % sample_every == 0:
                samples.append(self._sample())
        return samples

    def _sample(self) -> dict:
        report = self.energy_breakdown()
        report["time"] = self.time
        report["temperature"] = self.temperature()
        return report
