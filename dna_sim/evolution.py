"""Population-level evolution: replicate, mutate, and select over generations."""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from .mutation import MutationModel
from .replication import replicate
from .sequence import DNASequence

# A fitness function scores a sequence; higher is fitter.
FitnessFn = Callable[[DNASequence], float]


def gc_target_fitness(target: float) -> FitnessFn:
    """Fitness that peaks when GC content matches ``target`` (0..1).

    Returns a function scoring 1.0 at the target and falling off linearly to
    0.0 as GC content moves away from it.
    """

    def fitness(seq: DNASequence) -> float:
        return 1.0 - abs(seq.gc_content() - target)

    return fitness


@dataclass
class GenerationStats:
    """Summary statistics for one generation of a population."""

    generation: int
    size: int
    mean_fitness: float
    max_fitness: float
    mean_gc: float
    total_mutations: int


@dataclass
class Population:
    """A collection of DNA sequences evolving under mutation and selection."""

    members: list[DNASequence]
    model: MutationModel
    fitness_fn: FitnessFn
    rng: random.Random = field(default_factory=random.Random)
    generation: int = 0

    @classmethod
    def founded_from(
        cls,
        ancestor: DNASequence,
        size: int,
        model: MutationModel,
        fitness_fn: FitnessFn,
        rng: random.Random | None = None,
    ) -> "Population":
        """Create a population of ``size`` clones of a single ancestor."""
        if size <= 0:
            raise ValueError("population size must be positive")
        return cls(
            members=[ancestor for _ in range(size)],
            model=model,
            fitness_fn=fitness_fn,
            rng=rng or random.Random(),
        )

    def _select_parents(self, count: int) -> list[DNASequence]:
        """Fitness-proportionate selection (roulette wheel) with replacement.

        Falls back to uniform sampling when every member has zero fitness.
        """
        weights = [max(self.fitness_fn(m), 0.0) for m in self.members]
        if sum(weights) <= 0:
            return [self.rng.choice(self.members) for _ in range(count)]
        return self.rng.choices(self.members, weights=weights, k=count)

    def step(self) -> GenerationStats:
        """Advance the population by one generation and return its statistics.

        Each surviving parent, chosen by fitness, replicates once; the daughter
        strands form the next generation (constant population size).
        """
        parents = self._select_parents(len(self.members))
        next_members: list[DNASequence] = []
        total_mutations = 0
        for parent in parents:
            result = replicate(parent, self.model, self.rng)
            next_members.append(result.copy)
            total_mutations += len(result.events)

        self.members = next_members
        self.generation += 1
        return self._stats(total_mutations)

    def run(self, generations: int) -> list[GenerationStats]:
        """Run the simulation for a number of generations, collecting stats."""
        history = [self._stats(0)] if self.generation == 0 else []
        for _ in range(generations):
            history.append(self.step())
        return history

    def fittest(self) -> DNASequence:
        """Return the current member with the highest fitness."""
        return max(self.members, key=self.fitness_fn)

    def _stats(self, total_mutations: int) -> GenerationStats:
        fitnesses = [self.fitness_fn(m) for m in self.members]
        gcs = [m.gc_content() for m in self.members]
        size = len(self.members)
        return GenerationStats(
            generation=self.generation,
            size=size,
            mean_fitness=sum(fitnesses) / size,
            max_fitness=max(fitnesses),
            mean_gc=sum(gcs) / size,
            total_mutations=total_mutations,
        )
