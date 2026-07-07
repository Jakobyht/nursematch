import random

from atomsim import Atom, MolecularSystem
from atomsim.vector import Vec3
from protocell import build_membrane
from protocell.membrane import HEAD, TAIL, membrane_forces_factory


def _rg(atoms):
    cx = sum(a.position.x for a in atoms) / len(atoms)
    cy = sum(a.position.y for a in atoms) / len(atoms)
    return (sum((a.position.x - cx) ** 2 + (a.position.y - cy) ** 2 for a in atoms)
            / len(atoms)) ** 0.5


def test_tail_beads_attract_each_other():
    atoms = [Atom(element=TAIL, position=Vec3(0, 0, 0)),
             Atom(element=TAIL, position=Vec3(8, 0, 0))]
    forces, energy = membrane_forces_factory({0, 1})(atoms)
    assert forces[0].x > 0  # tail 0 pulled toward tail 1 (+x): hydrophobic cohesion
    assert energy["membrane"] < 0  # attractive well is negative


def test_head_beads_only_have_excluded_volume():
    # Close heads repel...
    near = [Atom(element=HEAD, position=Vec3(0, 0, 0)),
            Atom(element=HEAD, position=Vec3(3, 0, 0))]
    forces, _ = membrane_forces_factory(set())(near)
    assert forces[0].x < 0  # pushed apart (no cohesion between heads)
    # ...but feel nothing once beyond the repulsive range.
    far = [Atom(element=HEAD, position=Vec3(0, 0, 0)),
           Atom(element=HEAD, position=Vec3(9, 0, 0))]
    forces_far, energy_far = membrane_forces_factory(set())(far)
    assert abs(forces_far[0].x) < 1e-12
    assert energy_far["membrane"] == 0.0


def test_head_and_tail_do_not_attract():
    # A head and a tail beyond contact feel no attraction (only excluded volume).
    atoms = [Atom(element=HEAD, position=Vec3(0, 0, 0)),
             Atom(element=TAIL, position=Vec3(8, 0, 0))]
    forces, energy = membrane_forces_factory({1})(atoms)
    assert abs(forces[0].x) < 1e-12
    assert energy["membrane"] == 0.0


def test_membrane_forces_obey_newtons_third_law():
    atoms = [Atom(element=TAIL, position=Vec3(0, 0, 0)),
             Atom(element=TAIL, position=Vec3(5, 1, 0))]
    forces, _ = membrane_forces_factory({0, 1})(atoms)
    total = forces[0] + forces[1]
    assert abs(total.x) < 1e-12 and abs(total.y) < 1e-12


def test_hydrophobic_tails_coalesce():
    # Dispersed tail beads should cluster under hydrophobic cohesion.
    atoms = [Atom(element=TAIL, position=Vec3(x, 0, 0)) for x in (-9, -3, 3, 9)]
    system = MolecularSystem(
        atoms=atoms, extra_forces=membrane_forces_factory({0, 1, 2, 3}), damping=1.0
    )
    rg_start = _rg(atoms)
    system.run(steps=3000, dt=0.005)
    rg_end = _rg(atoms)
    assert rg_end < 0.8 * rg_start  # tails drew together


def test_amphiphiles_self_assemble_into_a_hydrophobic_core():
    # A batch of amphiphiles, scattered and randomly oriented, self-assembles so
    # the tails pack tighter than the heads — a compartment boundary from physics.
    membrane = build_membrane(n_amphiphiles=6, spread=6.0, rng=random.Random(1))
    membrane.system.run(steps=8000, dt=0.005)
    assert membrane.has_hydrophobic_core()


def test_membrane_stays_finite():
    # No blow-ups: after a run every coordinate is finite.
    import math

    membrane = build_membrane(n_amphiphiles=6, spread=6.0, rng=random.Random(1))
    membrane.system.run(steps=2000, dt=0.005)
    for a in membrane.system.atoms:
        assert math.isfinite(a.position.x) and math.isfinite(a.position.y)
