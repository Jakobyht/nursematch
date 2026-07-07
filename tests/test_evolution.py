import random

from dna_sim.evolution import Population, gc_target_fitness
from dna_sim.generate import random_sequence
from dna_sim.mutation import MutationModel
from dna_sim.sequence import DNASequence


def test_gc_target_fitness_peaks_at_target():
    fit = gc_target_fitness(0.5)
    assert fit(DNASequence("ATGC")) == 1.0  # GC = 0.5
    assert fit(DNASequence("GCGC")) == 0.5  # GC = 1.0, off by 0.5


def test_population_size_constant_across_generations():
    rng = random.Random(0)
    pop = Population.founded_from(
        ancestor=DNASequence("ATGCATGCAT"),
        size=20,
        model=MutationModel(0.02, 0.0, 0.0),
        fitness_fn=gc_target_fitness(0.6),
        rng=rng,
    )
    pop.run(10)
    assert len(pop.members) == 20
    assert pop.generation == 10


def test_selection_drives_gc_toward_target():
    rng = random.Random(7)
    ancestor = random_sequence(80, rng, gc_content=0.2)
    pop = Population.founded_from(
        ancestor=ancestor,
        size=200,
        model=MutationModel(substitution_rate=0.03, insertion_rate=0.0, deletion_rate=0.0),
        fitness_fn=gc_target_fitness(0.7),
        rng=rng,
    )
    history = pop.run(60)
    # Mean GC should move meaningfully from ~0.2 toward the 0.7 target.
    assert history[-1].mean_gc > history[0].mean_gc + 0.1


def test_run_records_initial_generation():
    rng = random.Random(0)
    pop = Population.founded_from(
        ancestor=DNASequence("ATGC"),
        size=5,
        model=MutationModel(0.0, 0.0, 0.0),
        fitness_fn=gc_target_fitness(0.5),
        rng=rng,
    )
    history = pop.run(3)
    assert [s.generation for s in history] == [0, 1, 2, 3]


def test_fittest_returns_best_member():
    rng = random.Random(3)
    pop = Population.founded_from(
        ancestor=DNASequence("ATATATAT"),
        size=10,
        model=MutationModel(0.1, 0.0, 0.0),
        fitness_fn=gc_target_fitness(1.0),
        rng=rng,
    )
    pop.run(5)
    best = pop.fittest()
    assert best.gc_content() >= min(m.gc_content() for m in pop.members)
