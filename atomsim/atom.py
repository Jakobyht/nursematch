"""The Atom: a point mass carrying charge, position, and velocity."""

from __future__ import annotations

from dataclasses import dataclass, field

from .elements import Element, element
from .vector import ZERO, Vec3


@dataclass
class Atom:
    """A single atom in the simulation.

    Position and velocity are mutable because the integrator advances them each
    step. Charge is the partial charge in elementary-charge units (e), which
    drives the electrostatic (Coulomb) interaction.
    """

    element: Element
    position: Vec3 = field(default_factory=lambda: ZERO)
    velocity: Vec3 = field(default_factory=lambda: ZERO)
    charge: float = 0.0

    @classmethod
    def of(
        cls,
        symbol: str,
        position: Vec3 | tuple[float, float, float] = ZERO,
        velocity: Vec3 | tuple[float, float, float] = ZERO,
        charge: float = 0.0,
    ) -> "Atom":
        """Convenience constructor from an element symbol and coordinates."""
        return cls(
            element=element(symbol),
            position=_as_vec(position),
            velocity=_as_vec(velocity),
            charge=charge,
        )

    @property
    def mass(self) -> float:
        return self.element.mass

    @property
    def symbol(self) -> str:
        return self.element.symbol

    def kinetic_energy(self) -> float:
        """Classical kinetic energy, ½·m·v²."""
        return 0.5 * self.mass * self.velocity.norm_sq()

    def __str__(self) -> str:
        p = self.position
        return f"{self.symbol}({p.x:.2f}, {p.y:.2f}, {p.z:.2f})"


def _as_vec(v: Vec3 | tuple[float, float, float]) -> Vec3:
    if isinstance(v, Vec3):
        return v
    return Vec3(*v)
