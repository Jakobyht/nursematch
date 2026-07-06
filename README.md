# life-from-atoms

A dependency-free Python project working toward one goal: **simulate matter
from the atom up, and re-create life through DNA on top of it.**

The philosophy borrows from how physics-based *digital twins* (e.g. the
simulation pipelines behind self-driving programs like Tesla's) are built —
model the real substrate accurately, calibrate against known physical
constants, and let complex behaviour *emerge* from validated low-level physics
rather than being scripted. Here the substrate is the atom.

Two layers, built to meet in the middle:

- **`atomsim`** — the *physical* layer. Classical molecular dynamics: atoms as
  point masses interacting through a real force field (Lennard-Jones van der
  Waals, Coulomb electrostatics, harmonic covalent bonds), integrated with the
  symplectic velocity-Verlet algorithm. This is how you "program atoms."
- **`dna_sim`** — the *informational* layer. DNA sequences and the processes
  acting on them: complementation, transcription, translation, mutation,
  replication, and population-level evolution under selection.

- **`nucleic`** — the *bridge*. Turns a `dna_sim` sequence into a physical
  strand of coarse-grained base beads, where Watson-Crick base pairing (A–T,
  G–C) *emerges* from a specific hydrogen-bond-like interaction between
  complementary bases. Two complementary strands, started apart, physically zip
  together; mismatched strands do not.

The mission is to grow this upward (atoms → molecules → macromolecules → DNA)
until DNA base-pairing and replication *emerge from the physics*, meeting
`dna_sim` from below — the point at which we can attempt to re-create life.

Everything runs on the Python standard library. Seeded random number
generators make every simulation reproducible.

### Progress toward life-from-atoms

1. ✅ **Atoms + force field + integrator** (`atomsim`) — energy-conserving MD
2. ✅ **Base-pairing emerges from physics** (`nucleic`) — complementary strands
   zip together; specificity (A–T vs G–C, matched vs mismatched) falls out of
   the interaction, not hand-coded rules
3. ⏳ Double-strand geometry and physically-driven replication
4. ⏳ Replication as a physical process, connected to `dna_sim` — the point at
   which we attempt to re-create life

## The bridge: `nucleic` (DNA base-pairing from physics)

```python
from nucleic import build_duplex

# Two complementary strands, started 6 Å apart, zip shut under the pairing force.
duplex = build_duplex("GCGATTACGC")          # partner defaults to reverse complement
duplex.system.run(steps=6000, dt=0.01)
duplex.paired_fraction()                      # -> 1.0 (fully paired)
duplex.mean_pair_distance()                   # -> ~3.0 Å

# Specificity is emergent: a non-complementary partner refuses to pair.
mismatch = build_duplex("AAAA", partner="AAAA")
mismatch.system.run(steps=6000, dt=0.01)
mismatch.paired_fraction()                    # -> 0.0
```

Run `python examples/pairing_demo.py` to watch strands zip (and a mismatch stay
open).

## The physical layer: `atomsim`

```python
from atomsim import Atom, MolecularSystem, Bond, water

# Build a water molecule and let it settle under the force field.
system = water()
samples = system.run(steps=2000, dt=0.0005, sample_every=500)
print(samples[-1]["total"], samples[-1]["temperature"])  # energy is conserved

# Or assemble atoms by hand:
atoms = [Atom.of("C", (0, 0, 0)), Atom.of("C", (1.6, 0, 0))]
bonds = [Bond(0, 1, length=1.3, stiffness=1000.0)]
dimer = MolecularSystem(atoms=atoms, bonds=bonds)
dimer.step(dt=0.0005)          # one velocity-Verlet step
dimer.total_energy()           # kinetic + potential
```

Run `python examples/atomsim_demo.py` to see the Lennard-Jones curve, a water
molecule conserving energy to ~0.02% drift, and a bond vibrating.

## The informational layer: `dna_sim`

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
