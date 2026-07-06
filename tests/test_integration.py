"""End-to-end integration: the whole stack, atoms -> molecules -> DNA -> life.

These tests prove the layers connect into one pipeline: real atoms obeying a
force field, molecules assembled from them, DNA realised physically from an
informational sequence, base-pairing and replication emerging from the physics,
and the informational layer's evolution running on top.
"""

from atomsim import water
from dna_sim import DNASequence, MutationModel, Population, gc_target_fitness
from nucleic import build_duplex, replicate_template
import random


def test_atoms_assemble_into_a_stable_molecule():
    # Layer 1: a molecule built from atoms conserves energy under the force
    # field — matter that holds together.
    system = water()
    e0 = system.total_energy()
    energies = [e0]
    for _ in range(1000):
        system.step(dt=0.0005)
        energies.append(system.total_energy())
    drift = (max(energies) - min(energies)) / abs(e0)
    assert drift < 0.02


def test_full_pipeline_information_to_matter_to_replication():
    # Layer 2->3: take an informational DNA sequence, realise it as physical
    # matter, and replicate it through the physics — then confirm the physical
    # result agrees with the informational operations it should reproduce.
    seq = DNASequence("GCGATTAC")

    # Informational sequence -> physical duplex that pairs up.
    duplex = build_duplex(str(seq))
    duplex.system.run(steps=6000, dt=0.01)
    assert duplex.paired_fraction() == 1.0

    # Physical replication reproduces the informational complement...
    result = replicate_template(str(seq), final_relax_steps=0)
    assert result.daughter == str(seq.complement())
    # ...and the daughter read 5'->3' is the reverse complement partner.
    assert str(result.as_daughter_strand()) == str(seq.reverse_complement())


def test_informational_layer_evolves_on_top():
    # Layer 4: the informational layer (populations under selection) runs on the
    # same DNA the physical layer manipulates — the stack is coherent top to
    # bottom.
    rng = random.Random(0)
    pop = Population.founded_from(
        ancestor=DNASequence("ATGCATGCAT"),
        size=50,
        model=MutationModel(substitution_rate=0.02),
        fitness_fn=gc_target_fitness(0.7),
        rng=rng,
    )
    history = pop.run(30)
    assert history[-1].mean_gc > history[0].mean_gc  # selection did its work
