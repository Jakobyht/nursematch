"""Demo: DNA replication emerging from physics.

Run with:  python examples/replication_demo.py

A template strand is copied one nucleotide at a time. Each incoming base is
*selected by pairing affinity* (Watson-Crick fidelity from physics) and
physically docks against the template. The assembled daughter is then checked
against dna_sim — the physical copy reproduces the informational complement,
and replicating the daughter regenerates the original template.
"""

from dna_sim import DNASequence
from nucleic import replicate_template, select_complement


def main() -> None:
    print("== Fidelity is emergent: base chosen by pairing affinity ==")
    for base in "ACGT":
        print(f"  template {base}  ->  recruit {select_complement(base)}")
    print()

    template = "GCGATTACGC"
    print(f"== Replicating template 5'-{template}-3' ==")
    result = replicate_template(template)
    informational = str(DNASequence(template).complement())
    print(f"  template            : {result.template}")
    print(f"  daughter (assembled): {result.daughter}")
    print(f"  dna_sim complement  : {informational}")
    print(f"  physical == informational: {result.daughter == informational}")
    print(f"  daughter 5'->3'     : {result.as_daughter_strand()}  "
          f"(= reverse complement)")
    print(f"  duplex paired       : {result.paired_fraction():.0%} "
          f"at {result.mean_pair_distance():.2f} Å")
    print()

    print("== Semiconservative round trip ==")
    daughter = result.as_daughter_strand()
    grand = replicate_template(daughter, final_relax_steps=0).as_daughter_strand()
    print(f"  template     : {template}")
    print(f"  daughter     : {daughter}")
    print(f"  grand-daughter: {grand}")
    print(f"  regenerated original: {str(grand) == template}")


if __name__ == "__main__":
    main()
