"""The whole pipeline in one run: atoms -> molecules -> DNA -> replication -> life.

Run with:  python examples/origin_pipeline.py

This is the capstone: a single narrative that walks the full stack the project
builds, from a handful of atoms obeying a force field up to DNA replicating
itself and a population evolving under selection.
"""

import random

from atomsim import water
from dna_sim import DNASequence, MutationModel, Population, gc_target_fitness
from nucleic import build_duplex, replicate_template


def stage_atoms() -> None:
    print("1. ATOMS -> MATTER")
    system = water()
    e0 = system.total_energy()
    system.run(steps=2000, dt=0.0005)
    e1 = system.total_energy()
    print(f"   A water molecule assembled from O + 2H holds together;")
    print(f"   total energy conserved: {e0:.2f} -> {e1:.2f} kJ/mol")
    print()


def stage_pairing() -> None:
    print("2. MATTER -> BASE PAIRING")
    duplex = build_duplex("GCGATTACGC")
    start = duplex.mean_pair_distance()
    duplex.system.run(steps=6000, dt=0.01)
    print(f"   Two complementary strands, started {start:.0f} Å apart, zip to "
          f"{duplex.mean_pair_distance():.1f} Å")
    print(f"   base pairing emerges from physics: {duplex.paired_fraction():.0%} paired")
    print()


def stage_replication() -> DNASequence:
    print("3. BASE PAIRING -> REPLICATION")
    template = DNASequence("GCGATTACGC")
    result = replicate_template(str(template))
    print(f"   Template   5'-{result.template}-3'")
    print(f"   copied to   3'-{result.daughter}-5'  (= dna_sim complement: "
          f"{result.daughter == str(template.complement())})")
    print(f"   duplex paired at {result.paired_fraction():.0%}; fidelity from pairing affinity")
    print()
    return result.as_daughter_strand()


def stage_life(seed_strand: DNASequence) -> None:
    print("4. REPLICATION -> LIFE (variation + selection)")
    rng = random.Random(2026)
    pop = Population.founded_from(
        ancestor=seed_strand,
        size=120,
        model=MutationModel(substitution_rate=0.03),
        fitness_fn=gc_target_fitness(0.75),
        rng=rng,
    )
    history = pop.run(40)
    print(f"   A population of the replicated strand evolves under selection:")
    print(f"   mean GC {history[0].mean_gc:.2f} -> {history[-1].mean_gc:.2f} "
          f"(target 0.75), fittest: {pop.fittest()}")
    print()


def main() -> None:
    print("=" * 60)
    print(" FROM ATOMS TO LIFE — the full simulated pipeline")
    print("=" * 60)
    print()
    stage_atoms()
    stage_pairing()
    daughter = stage_replication()
    stage_life(daughter)
    print("Atoms held together, paired, replicated, and evolved —")
    print("the substrate on which we can attempt to re-create life through DNA.")


if __name__ == "__main__":
    main()
