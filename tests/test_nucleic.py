import math

from atomsim import Atom, MolecularSystem
from atomsim.vector import Vec3
from nucleic import (
    BasePair,
    are_complementary,
    build_duplex,
    pair_strength,
    pairing_pair,
    pairing_forces_factory,
)
from nucleic.strand import BASE_BEAD


def test_complement_relationships():
    assert are_complementary("A", "T")
    assert are_complementary("G", "C")
    assert not are_complementary("A", "G")
    assert not are_complementary("A", "C")


def test_gc_binds_tighter_than_at():
    assert pair_strength("G", "C") > pair_strength("A", "T") > 0
    assert pair_strength("A", "G") == 0.0  # non-complementary: no pairing


def test_pairing_well_minimum_at_r0():
    a = Atom(element=BASE_BEAD, position=Vec3(0, 0, 0))
    b = Atom(element=BASE_BEAD, position=Vec3(3.0, 0, 0))  # r0 default = 3.0
    energy, force = pairing_pair(a, b, strength=16.0)
    assert math.isclose(energy, -16.0, abs_tol=1e-9)  # deepest at r0
    assert abs(force.x) < 1e-9  # force vanishes at the minimum


def test_pairing_is_attractive_when_too_far():
    a = Atom(element=BASE_BEAD, position=Vec3(0, 0, 0))
    b = Atom(element=BASE_BEAD, position=Vec3(4.0, 0, 0))  # beyond r0
    _, force = pairing_pair(a, b, strength=16.0)
    assert force.x > 0  # a is pulled toward +x (toward b)


def test_no_pairing_force_for_noncomplementary():
    a = Atom(element=BASE_BEAD, position=Vec3(0, 0, 0))
    b = Atom(element=BASE_BEAD, position=Vec3(4.0, 0, 0))
    energy, force = pairing_pair(a, b, strength=0.0)
    assert energy == 0.0
    assert force == Vec3(0, 0, 0)


def test_pairing_forces_factory_newton_third_law():
    atoms = [
        Atom(element=BASE_BEAD, position=Vec3(0, 0, 0)),
        Atom(element=BASE_BEAD, position=Vec3(0, 5, 0)),
    ]
    extra = pairing_forces_factory([BasePair(0, 1, strength=16.0)])
    forces, energy = extra(atoms)
    total = forces[0] + forces[1]
    assert abs(total.x) < 1e-12 and abs(total.y) < 1e-12
    assert energy["base_pairing"] < 0  # bound => negative energy


def test_complementary_strands_zip_together():
    # Start the two strands apart; pairing should pull them to ~r0.
    duplex = build_duplex("GCGCGC", separation=6.0)
    start_dist = duplex.mean_pair_distance()
    duplex.system.run(steps=6000, dt=0.01)
    end_dist = duplex.mean_pair_distance()
    assert end_dist < start_dist
    # Settled near the pairing distance (r0 = 3.0).
    assert abs(end_dist - 3.0) < 1.0
    assert duplex.paired_fraction(tolerance=1.0) == 1.0


def test_reverse_complement_partner_zips_antiparallel():
    # The biologically correct partner is the reverse complement, laid down
    # antiparallel. It must pair fully, just like the auto-complement path.
    from dna_sim import DNASequence

    seq = DNASequence("ATGCGA")
    duplex = build_duplex(str(seq), partner=str(seq.reverse_complement()))
    duplex.system.run(steps=8000, dt=0.01)
    assert duplex.paired_fraction(tolerance=1.0) == 1.0
    assert abs(duplex.mean_pair_distance() - 3.0) < 1.0


def test_noncomplementary_strands_do_not_zip():
    # Partner identical to sequence -> every rung is A-opposite-A etc: no pairing.
    duplex = build_duplex("AAAA", partner="AAAA", separation=6.0)
    duplex.system.run(steps=6000, dt=0.01)
    # No pairing attraction, so they do not collapse to the pairing distance.
    assert duplex.paired_fraction(tolerance=1.0) == 0.0
    assert math.isclose(duplex.pairing_energy(), 0.0, abs_tol=1e-9)


def test_pairing_lowers_energy_of_complementary_duplex():
    duplex = build_duplex("ATGC", separation=6.0)
    e_start = duplex.pairing_energy()
    duplex.system.run(steps=6000, dt=0.01)
    e_end = duplex.pairing_energy()
    assert e_end < e_start  # pairing energy becomes more negative as it binds
    assert e_end < 0
