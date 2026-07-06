"""Demo: mutation emerging from heat, closing the Darwinian loop physically.

Run with:  python examples/mutation_from_heat.py

Replication fidelity is a balance between pairing energy and thermal noise. As
temperature rises, misincorporation — mutation — emerges. Those physically
generated errors are all the variation a population needs: with no informational
mutation model at all, selection alone turns physics-made diversity into
adaptation.
"""

import random

from dna_sim import DNASequence, MutationModel, Population, gc_target_fitness
from nucleic import replicate_sequence, replication_error_rate


def show_error_curve() -> None:
    print("== Replication error rate vs temperature ==")
    template = "GCGATTACGCTAGCTA"
    print(f"  template: {template}")
    for temp in (0.0, 2.0, 5.0, 10.0, 20.0, 40.0):
        rng = random.Random(42)
        rate = replication_error_rate(template, temp, trials=300, rng=rng)
        bar = "#" * round(rate * 40)
        print(f"  T={temp:5.1f} kJ/mol   error {rate:6.3f}  {bar}")
    print()


def evolve_on_physical_variation() -> None:
    print("== Darwinian loop: physics-made variation + selection ==")
    rng = random.Random(2026)
    template = "ATATATGCATATATAT"  # AT-rich -> low starting GC, room to adapt
    # Every population member is a *physically* replicated copy whose
    # differences come only from thermal replication errors.
    members = [
        DNASequence(replicate_sequence(template, temperature=12.0, rng=rng))
        for _ in range(150)
    ]
    # No informational mutation model — variation is entirely physics-derived.
    pop = Population(
        members=members,
        model=MutationModel(0.0, 0.0, 0.0),
        fitness_fn=gc_target_fitness(0.9),
        rng=rng,
    )
    history = pop.run(12)
    print(f"  variation source : thermal replication errors only")
    print(f"  selection target : GC content 0.90")
    print(f"  gen  0: mean GC {history[0].mean_gc:.3f}, "
          f"mean fitness {history[0].mean_fitness:.3f}")
    print(f"  gen {len(history) - 1:>2}: mean GC {history[-1].mean_gc:.3f}, "
          f"mean fitness {history[-1].mean_fitness:.3f}")
    print(f"  fittest survivor : {pop.fittest()}")
    print()
    print("Heat made the mutations; selection made them matter.")


def main() -> None:
    show_error_curve()
    evolve_on_physical_variation()


if __name__ == "__main__":
    main()
