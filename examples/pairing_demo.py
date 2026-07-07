"""Demo: DNA base-pairing emerging from physics.

Run with:  python examples/pairing_demo.py

Two complementary strands start apart and *zip together* under nothing but the
pairing interaction and the integrator — Watson-Crick pairing as an emergent
behaviour of the physical substrate, not a scripted rule. A mismatched partner
refuses to pair, showing the specificity is real.
"""

from dna_sim import DNASequence
from nucleic import build_duplex


def zip_up(label: str, sequence: str, partner: str | None = None) -> None:
    duplex = build_duplex(sequence, partner=partner, separation=6.0)
    seq_a = "".join(duplex.bases[i] for i in duplex.strand_a)
    seq_b = "".join(duplex.bases[i] for i in duplex.strand_b)
    print(f"== {label} ==")
    print(f"  strand A : 5'-{seq_a}-3'")
    print(f"  strand B :    {seq_b}")
    print(f"  start: mean pair distance {duplex.mean_pair_distance():.2f} Å, "
          f"pairing energy {duplex.pairing_energy():.1f} kJ/mol")
    samples = duplex.system.run(steps=6000, dt=0.01, sample_every=1500)
    for s in samples:
        print(f"    t={s['time']:5.1f}  pairing E={s.get('base_pairing', 0.0):8.2f}")
    print(f"  end:   mean pair distance {duplex.mean_pair_distance():.2f} Å, "
          f"paired fraction {duplex.paired_fraction():.0%}")
    print()


def main() -> None:
    # A perfectly complementary partner: zips shut.
    zip_up("Complementary duplex (auto-complement)", "GCGATTACGC")

    # Take a DNA sequence from the informational layer and physically realise it.
    seq = DNASequence("ATGCGA")
    zip_up("From a dna_sim sequence", str(seq), str(seq.reverse_complement()))

    # A mismatched partner: pairing is specific, so it stays open.
    zip_up("Mismatched partner (no complementarity)", "AAAA", "AAAA")


if __name__ == "__main__":
    main()
