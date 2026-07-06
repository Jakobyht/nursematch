import math

from atomsim import (
    Atom,
    Bond,
    MolecularSystem,
    Vec3,
    compute_forces,
    element,
    lennard_jones_pair,
)
from atomsim.elements import lorentz_berthelot


def test_vec3_arithmetic():
    a = Vec3(1, 2, 3)
    b = Vec3(4, 5, 6)
    assert a + b == Vec3(5, 7, 9)
    assert b - a == Vec3(3, 3, 3)
    assert a * 2 == Vec3(2, 4, 6)
    assert a.dot(b) == 32
    assert math.isclose(Vec3(3, 4, 0).norm(), 5.0)


def test_element_lookup():
    c = element("C")
    assert c.atomic_number == 6
    assert c.mass > 12


def test_lorentz_berthelot_mixing():
    h = element("H")
    o = element("O")
    sigma, eps = lorentz_berthelot(h, o)
    assert math.isclose(sigma, 0.5 * (h.lj_sigma + o.lj_sigma))
    assert math.isclose(eps, (h.lj_epsilon * o.lj_epsilon) ** 0.5)


def test_lj_force_is_zero_at_sigma_scaled_minimum():
    # At the LJ minimum r = 2^(1/6)·σ the force should vanish.
    a = Atom.of("C", (0, 0, 0))
    b = Atom.of("C", (0, 0, 0))
    sigma, _ = lorentz_berthelot(a.element, b.element)
    rmin = 2 ** (1 / 6) * sigma
    b.position = Vec3(rmin, 0, 0)
    _, force = lennard_jones_pair(a, b)
    assert abs(force.x) < 1e-6


def test_lj_repulsive_when_close():
    # Closer than sigma -> strong repulsion pushing atom a in -x (away from b).
    a = Atom.of("C", (0, 0, 0))
    b = Atom.of("C", (1.0, 0, 0))  # well inside sigma (~3.4)
    _, force = lennard_jones_pair(a, b)
    assert force.x < 0  # a is pushed toward -x, away from b at +x


def test_forces_are_newtonian_pairwise():
    # Total force on an isolated pair must sum to zero (Newton's third law).
    atoms = [Atom.of("O", (0, 0, 0), charge=-0.5), Atom.of("H", (1.0, 0, 0), charge=0.5)]
    forces, _ = compute_forces(atoms, [])
    total = forces[0] + forces[1]
    assert abs(total.x) < 1e-9
    assert abs(total.y) < 1e-9
    assert abs(total.z) < 1e-9


def test_harmonic_bond_restores_length():
    # Two atoms bonded at length 1.0 but placed at 1.5 should be pulled together.
    atoms = [Atom.of("C", (0, 0, 0)), Atom.of("C", (1.5, 0, 0))]
    bonds = [Bond(0, 1, length=1.0, stiffness=500.0)]
    forces, energy = compute_forces(atoms, bonds)
    assert energy["bond"] > 0
    assert forces[0].x > 0  # atom 0 pulled toward +x (toward atom 1)
    assert forces[1].x < 0


def test_energy_conservation_diatomic():
    # A stretched bond oscillates; velocity Verlet should conserve total energy.
    atoms = [Atom.of("C", (0, 0, 0)), Atom.of("C", (1.3, 0, 0))]
    bonds = [Bond(0, 1, length=1.2, stiffness=800.0)]
    system = MolecularSystem(atoms=atoms, bonds=bonds)
    e0 = system.total_energy()
    energies = [e0]
    for _ in range(2000):
        system.step(dt=0.001)
        energies.append(system.total_energy())
    drift = (max(energies) - min(energies)) / abs(e0)
    assert drift < 0.01  # < 1% energy drift over 2000 steps


def test_temperature_nonnegative_and_scales_with_ke():
    fast = MolecularSystem(atoms=[Atom.of("H", velocity=(10, 0, 0))])
    slow = MolecularSystem(atoms=[Atom.of("H", velocity=(1, 0, 0))])
    assert fast.temperature() > slow.temperature() >= 0


def test_center_of_mass():
    atoms = [Atom.of("H", (0, 0, 0)), Atom.of("H", (2, 0, 0))]
    system = MolecularSystem(atoms=atoms)
    com = system.center_of_mass()
    assert math.isclose(com.x, 1.0)


def test_run_collects_samples():
    atoms = [Atom.of("C", (0, 0, 0)), Atom.of("C", (1.3, 0, 0))]
    bonds = [Bond(0, 1, length=1.2)]
    system = MolecularSystem(atoms=atoms, bonds=bonds)
    samples = system.run(steps=100, dt=0.001, sample_every=25)
    assert len(samples) == 5  # initial + 4
    assert all("total" in s and "temperature" in s for s in samples)
