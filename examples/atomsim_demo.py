"""Demo of the atomic (molecular-dynamics) layer.

Run with:  python examples/atomsim_demo.py

Shows the physical substrate the project builds life on: atoms interacting
through a real force field, molecules holding together, and total energy being
conserved by the velocity-Verlet integrator.
"""

from atomsim import Atom, MolecularSystem, lennard_jones_pair, water
from atomsim.elements import lorentz_berthelot
from atomsim.vector import Vec3


def show_lj_curve() -> None:
    print("== Lennard-Jones between two carbons (energy vs distance) ==")
    a = Atom.of("C", (0, 0, 0))
    b = Atom.of("C", (0, 0, 0))
    sigma, _ = lorentz_berthelot(a.element, b.element)
    for r in (0.8 * sigma, sigma, 2 ** (1 / 6) * sigma, 1.5 * sigma, 2.5 * sigma):
        b.position = Vec3(r, 0, 0)
        energy, force = lennard_jones_pair(a, b)
        print(f"  r={r:5.2f} Å   E={energy:8.3f} kJ/mol   Fx={force.x:9.3f}")
    print()


def relax_water() -> None:
    print("== A water molecule settling into equilibrium ==")
    system = water()
    print(f"  atoms: {[str(a) for a in system.atoms]}")
    samples = system.run(steps=2000, dt=0.0005, sample_every=500)
    print(f"  {'time':>6} {'potential':>10} {'kinetic':>9} {'total':>10} {'T(K)':>8}")
    for s in samples:
        print(
            f"  {s['time']:>6.2f} {s['potential']:>10.3f} "
            f"{s['kinetic']:>9.3f} {s['total']:>10.3f} {s['temperature']:>8.1f}"
        )
    e = [s["total"] for s in samples]
    drift = (max(e) - min(e)) / abs(e[0]) if e[0] else 0.0
    print(f"  total-energy drift: {drift * 100:.3f}%")
    print()


def diatomic_vibration() -> None:
    print("== A carbon dimer vibrating (bond stretched then released) ==")
    from atomsim import Bond

    atoms = [Atom.of("C", (0, 0, 0)), Atom.of("C", (1.6, 0, 0))]
    bonds = [Bond(0, 1, length=1.3, stiffness=1000.0)]
    system = MolecularSystem(atoms=atoms, bonds=bonds)
    print(f"  start length: {atoms[0].position.distance_to(atoms[1].position):.3f} Å")
    for _ in range(5):
        system.run(steps=200, dt=0.0005)
        length = atoms[0].position.distance_to(atoms[1].position)
        print(f"  t={system.time:.2f}  length={length:.3f} Å  E={system.total_energy():.3f}")
    print()


def main() -> None:
    show_lj_curve()
    relax_water()
    diatomic_vibration()


if __name__ == "__main__":
    main()
