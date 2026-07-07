"""Command-line interface for the DNA simulation toolkit."""

from __future__ import annotations

import argparse
import random

from .evolution import Population, gc_target_fitness
from .generate import random_sequence
from .mutation import MutationModel
from .sequence import DNASequence, InvalidSequenceError


def _resolve_sequence(seq_arg: str | None, length: int, rng: random.Random) -> DNASequence:
    if seq_arg:
        return DNASequence(seq_arg)
    return random_sequence(length, rng)


def cmd_analyze(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    seq = _resolve_sequence(args.sequence, args.length, rng)
    print(f"sequence          : {seq}")
    print(f"length            : {len(seq)}")
    print(f"GC content        : {seq.gc_content():.3f}")
    print(f"base counts       : {seq.base_counts()}")
    print(f"reverse complement: {seq.reverse_complement()}")
    print(f"mRNA              : {seq.transcribe()}")
    print(f"protein           : {seq.translate() or '(none)'}")
    orfs = seq.find_orfs(min_codons=args.min_orf)
    print(f"ORFs (>= {args.min_orf} aa)   : {len(orfs)}")
    for start, protein in orfs:
        print(f"  @{start}: {protein}")
    return 0


def cmd_replicate(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    seq = _resolve_sequence(args.sequence, args.length, rng)
    model = MutationModel(
        substitution_rate=args.substitution_rate,
        insertion_rate=args.indel_rate,
        deletion_rate=args.indel_rate,
    )
    current = seq
    print(f"gen 0: {current}")
    for gen in range(1, args.generations + 1):
        mutated, events = model.apply(current, rng)
        current = mutated
        marker = f"  ({len(events)} mutations)" if events else ""
        print(f"gen {gen}: {current}{marker}")
    return 0


def cmd_evolve(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    ancestor = _resolve_sequence(args.sequence, args.length, rng)
    model = MutationModel(
        substitution_rate=args.substitution_rate,
        insertion_rate=args.indel_rate,
        deletion_rate=args.indel_rate,
    )
    population = Population.founded_from(
        ancestor=ancestor,
        size=args.population,
        model=model,
        fitness_fn=gc_target_fitness(args.gc_target),
        rng=rng,
    )
    history = population.run(args.generations)
    print(f"ancestor: {ancestor}  (GC={ancestor.gc_content():.3f})")
    print(f"target GC: {args.gc_target:.3f}, population: {args.population}")
    print()
    print(f"{'gen':>4} {'mean_fit':>9} {'max_fit':>8} {'mean_gc':>8} {'muts':>6}")
    for stats in history:
        print(
            f"{stats.generation:>4} "
            f"{stats.mean_fitness:>9.4f} "
            f"{stats.max_fitness:>8.4f} "
            f"{stats.mean_gc:>8.4f} "
            f"{stats.total_mutations:>6}"
        )
    print()
    print(f"fittest: {population.fittest()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dna-sim",
        description="Simulate DNA sequences, replication, and evolution.",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_sequence_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("-s", "--sequence", help="explicit DNA sequence (A/T/G/C)")
        p.add_argument("-n", "--length", type=int, default=60, help="length when generating a random sequence")

    p_analyze = sub.add_parser("analyze", help="report on a sequence")
    add_sequence_args(p_analyze)
    p_analyze.add_argument("--min-orf", type=int, default=1, help="minimum ORF length in amino acids")
    p_analyze.set_defaults(func=cmd_analyze)

    p_replicate = sub.add_parser("replicate", help="replicate a strand with errors over generations")
    add_sequence_args(p_replicate)
    p_replicate.add_argument("-g", "--generations", type=int, default=10)
    p_replicate.add_argument("--substitution-rate", type=float, default=0.01)
    p_replicate.add_argument("--indel-rate", type=float, default=0.002)
    p_replicate.set_defaults(func=cmd_replicate)

    p_evolve = sub.add_parser("evolve", help="evolve a population toward a GC target")
    add_sequence_args(p_evolve)
    p_evolve.add_argument("-g", "--generations", type=int, default=50)
    p_evolve.add_argument("-p", "--population", type=int, default=100)
    p_evolve.add_argument("--gc-target", type=float, default=0.6)
    p_evolve.add_argument("--substitution-rate", type=float, default=0.02)
    p_evolve.add_argument("--indel-rate", type=float, default=0.0)
    p_evolve.set_defaults(func=cmd_evolve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (InvalidSequenceError, ValueError) as exc:
        parser.error(str(exc))
        return 2  # unreachable; parser.error exits


if __name__ == "__main__":
    raise SystemExit(main())
