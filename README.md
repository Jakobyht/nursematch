# dna-sim

A small, dependency-free Python toolkit for **simulating DNA** — its sequences
and the core molecular processes that act on them: complementation,
transcription, translation, mutation, replication, and population-level
evolution under selection.

Everything runs on the Python standard library. Seeded random number
generators make every simulation reproducible.

## Install

```bash
pip install -e .          # library + `dna-sim` command
pip install -e ".[dev]"   # also installs pytest
```

Requires Python 3.10+.

## Library usage

```python
import random
from dna_sim import DNASequence, MutationModel, Population, gc_target_fitness

# Sequences know their own biology.
seq = DNASequence("ATGGCCATTGTAATGGGCCGCTGA")
seq.reverse_complement()   # -> DNASequence("TCAGCGGCCCATTACAATGGCCAT")
seq.gc_content()           # -> 0.58...
seq.transcribe()           # -> "AUGGCCAUUGUAAUGGGCCGCUGA"  (mRNA)
seq.translate()            # -> "MAIVMGR"  (protein, stops at first stop codon)
seq.find_orfs()            # -> [(0, "MAIVMGR")]

# Replicate a strand with polymerase errors.
model = MutationModel(substitution_rate=0.02, insertion_rate=0.005, deletion_rate=0.005)
rng = random.Random(42)
daughter, events = model.apply(seq, rng)

# Evolve a whole population toward a target GC content.
pop = Population.founded_from(
    ancestor=seq,
    size=200,
    model=MutationModel(substitution_rate=0.03),
    fitness_fn=gc_target_fitness(0.7),
    rng=rng,
)
history = pop.run(generations=50)
print(history[-1].mean_gc)     # trends toward 0.70
print(pop.fittest())           # best sequence in the final generation
```

## Command line

```bash
# Analyze a sequence (or a random one with -n LENGTH)
dna-sim analyze -s ATGGCCATTGTAATGGGCCGCTGA
dna-sim --seed 1 analyze -n 90 --min-orf 5

# Replicate a strand with errors across generations
dna-sim --seed 1 replicate -s ATGCATGCATGC -g 8 --substitution-rate 0.05

# Evolve a population toward a GC target
dna-sim --seed 1 evolve -n 60 -p 150 -g 50 --gc-target 0.65
```

## What it models

| Module | Responsibility |
| --- | --- |
| `dna_sim.sequence` | `DNASequence`: complement, reverse complement, GC content, transcription, translation, ORF finding |
| `dna_sim.mutation` | Substitutions, insertions, deletions, and a per-base `MutationModel` |
| `dna_sim.replication` | Copying a template strand under a mutation model |
| `dna_sim.evolution` | `Population` evolving via fitness-proportionate selection |
| `dna_sim.generate` | Random sequence generation with optional GC bias |
| `dna_sim.cli` | `dna-sim` command-line entry point |

## Development

```bash
pip install -e ".[dev]"
pytest
python examples/demo.py
```

## License

MIT
