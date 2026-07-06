"""End-to-end demo of the dna_sim toolkit.

Run with:  python examples/demo.py
"""

import random

from dna_sim import (
    DNASequence,
    MutationModel,
    Population,
    gc_target_fitness,
    random_sequence,
    replicate,
)


def main() -> None:
    rng = random.Random(2024)

    print("== Sequence basics ==")
    gene = DNASequence("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG")
    print(f"gene              : {gene}")
    print(f"reverse complement: {gene.reverse_complement()}")
    print(f"GC content        : {gene.gc_content():.2f}")
    print(f"protein           : {gene.translate()}")
    print()

    print("== Replication with errors ==")
    model = MutationModel(substitution_rate=0.02, insertion_rate=0.005, deletion_rate=0.005)
    strand = random_sequence(40, rng)
    print(f"template: {strand}")
    for gen in range(1, 4):
        result = replicate(strand, model, rng)
        strand = result.copy
        print(f"copy {gen} : {strand}  ({len(result.events)} mutations)")
    print()

    print("== Evolving toward 70% GC ==")
    ancestor = random_sequence(60, rng, gc_content=0.3)
    population = Population.founded_from(
        ancestor=ancestor,
        size=150,
        model=MutationModel(substitution_rate=0.03),
        fitness_fn=gc_target_fitness(0.7),
        rng=rng,
    )
    history = population.run(40)
    for stats in history[:: 8]:
        print(
            f"gen {stats.generation:>2}: "
            f"mean fitness {stats.mean_fitness:.3f}, "
            f"mean GC {stats.mean_gc:.3f}"
        )
    print(f"final mean GC: {history[-1].mean_gc:.3f} (target 0.70)")


if __name__ == "__main__":
    main()
