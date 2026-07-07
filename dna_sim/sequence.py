"""Core DNA sequence representation and molecular-biology operations."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

# Watson-Crick base pairing for DNA.
DNA_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}
BASES = ("A", "T", "G", "C")

# Standard genetic code: RNA codon -> amino acid single-letter code.
# "*" marks a stop codon.
CODON_TABLE = {
    "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L",
    "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
    "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M",
    "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
    "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S",
    "CCU": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCU": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "UAU": "Y", "UAC": "Y", "UAA": "*", "UAG": "*",
    "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAU": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "UGU": "C", "UGC": "C", "UGA": "*", "UGG": "W",
    "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGU": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

START_CODON = "AUG"


class InvalidSequenceError(ValueError):
    """Raised when a sequence contains characters outside the DNA alphabet."""


@dataclass(frozen=True)
class DNASequence:
    """An immutable strand of DNA read 5' -> 3'.

    Bases are stored uppercase and validated against the {A, T, G, C} alphabet.
    """

    bases: str

    def __post_init__(self) -> None:
        upper = self.bases.upper()
        invalid = set(upper) - set(BASES)
        if invalid:
            raise InvalidSequenceError(
                f"sequence contains non-DNA characters: {sorted(invalid)}"
            )
        # Normalise to uppercase even though the dataclass is frozen.
        object.__setattr__(self, "bases", upper)

    def __len__(self) -> int:
        return len(self.bases)

    def __str__(self) -> str:
        return self.bases

    def __getitem__(self, index):  # type: ignore[no-untyped-def]
        return self.bases[index]

    def complement(self) -> "DNASequence":
        """Return the complementary strand (same 5'->3' direction reversed)."""
        return DNASequence("".join(DNA_COMPLEMENT[b] for b in self.bases))

    def reverse_complement(self) -> "DNASequence":
        """Return the reverse complement — the antiparallel partner strand."""
        return DNASequence(
            "".join(DNA_COMPLEMENT[b] for b in reversed(self.bases))
        )

    def gc_content(self) -> float:
        """Fraction of bases that are G or C (0.0 for an empty sequence)."""
        if not self.bases:
            return 0.0
        gc = sum(1 for b in self.bases if b in ("G", "C"))
        return gc / len(self.bases)

    def base_counts(self) -> dict[str, int]:
        """Count of each base, always including all four keys."""
        counts = Counter(self.bases)
        return {base: counts.get(base, 0) for base in BASES}

    def transcribe(self) -> str:
        """Transcribe the coding strand into messenger RNA (T -> U)."""
        return self.bases.replace("T", "U")

    def translate(self, to_stop: bool = True) -> str:
        """Translate the sequence into a protein (single-letter amino acids).

        Reads codons from the first base. When ``to_stop`` is True, translation
        halts at the first stop codon and the stop marker is omitted.
        """
        rna = self.transcribe()
        protein: list[str] = []
        for i in range(0, len(rna) - 2, 3):
            codon = rna[i : i + 3]
            amino = CODON_TABLE[codon]
            if amino == "*":
                if to_stop:
                    break
                protein.append(amino)
            else:
                protein.append(amino)
        return "".join(protein)

    def find_orfs(self, min_codons: int = 1) -> list[tuple[int, str]]:
        """Find open reading frames on this strand.

        Returns a list of ``(start_index, protein)`` for each ORF that begins
        with a start codon and ends at an in-frame stop codon, keeping only
        proteins of at least ``min_codons`` amino acids.
        """
        rna = self.transcribe()
        orfs: list[tuple[int, str]] = []
        for frame in range(3):
            i = frame
            while i < len(rna) - 2:
                if rna[i : i + 3] == START_CODON:
                    protein: list[str] = []
                    j = i
                    while j < len(rna) - 2:
                        amino = CODON_TABLE[rna[j : j + 3]]
                        if amino == "*":
                            if len(protein) >= min_codons:
                                orfs.append((i, "".join(protein)))
                            break
                        protein.append(amino)
                        j += 3
                    i = j + 3
                else:
                    i += 3
        return orfs
