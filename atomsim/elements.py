"""Element data for the atoms that make up biological matter.

Parameters are given in a consistent molecular-dynamics unit system:

* mass            : atomic mass units (amu / Dalton)
* Lennard-Jones σ : Ångström  (distance where the LJ potential is zero)
* Lennard-Jones ε : kJ/mol    (depth of the LJ potential well)
* covalent radius : Ångström  (used to guess bonds from geometry)

Values are approximate, drawn from common force-field parameterisations
(OPLS/AMBER-like). They are good enough to make matter behave qualitatively
correctly — bonds hold, atoms repel at short range and attract at long range —
which is what a from-atoms-up life simulation needs as a foundation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Element:
    symbol: str
    name: str
    atomic_number: int
    mass: float  # amu
    lj_sigma: float  # Å
    lj_epsilon: float  # kJ/mol
    covalent_radius: float  # Å

    def __str__(self) -> str:
        return self.symbol


# The core elements of life (CHNOPS) plus a couple of common partners.
_ELEMENTS = [
    Element("H", "Hydrogen", 1, 1.008, 2.50, 0.126, 0.31),
    Element("C", "Carbon", 6, 12.011, 3.40, 0.360, 0.76),
    Element("N", "Nitrogen", 7, 14.007, 3.25, 0.711, 0.71),
    Element("O", "Oxygen", 8, 15.999, 3.12, 0.669, 0.66),
    Element("P", "Phosphorus", 15, 30.974, 3.74, 0.837, 1.07),
    Element("S", "Sulfur", 16, 32.06, 3.55, 1.046, 1.05),
    Element("Na", "Sodium", 11, 22.990, 2.58, 0.062, 1.66),
    Element("Cl", "Chlorine", 17, 35.45, 3.47, 1.109, 1.02),
]

ELEMENTS: dict[str, Element] = {e.symbol: e for e in _ELEMENTS}


def element(symbol: str) -> Element:
    """Look up an element by chemical symbol (case-sensitive, e.g. 'C')."""
    try:
        return ELEMENTS[symbol]
    except KeyError as exc:
        raise KeyError(
            f"unknown element {symbol!r}; known: {sorted(ELEMENTS)}"
        ) from exc


def lorentz_berthelot(a: Element, b: Element) -> tuple[float, float]:
    """Combine two elements' LJ parameters for a mixed pair.

    Uses the standard Lorentz-Berthelot mixing rules: arithmetic mean for σ,
    geometric mean for ε.
    """
    sigma = 0.5 * (a.lj_sigma + b.lj_sigma)
    epsilon = (a.lj_epsilon * b.lj_epsilon) ** 0.5
    return sigma, epsilon
