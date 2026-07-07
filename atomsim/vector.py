"""A minimal, dependency-free 3D vector for molecular dynamics.

Kept deliberately small: molecular dynamics only needs add/sub/scale, dot,
norm, and distance. Vectors are immutable so positions and velocities are
never mutated in place by accident.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> "Vec3":
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def norm_sq(self) -> float:
        """Squared length — cheaper than ``norm`` when only comparing."""
        return self.dot(self)

    def norm(self) -> float:
        return math.sqrt(self.norm_sq())

    def unit(self) -> "Vec3":
        n = self.norm()
        if n == 0.0:
            raise ValueError("cannot normalise a zero-length vector")
        return self / n

    def distance_to(self, other: "Vec3") -> float:
        return (self - other).norm()


ZERO = Vec3(0.0, 0.0, 0.0)
