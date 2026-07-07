import pytest

from dna_sim import DNASequence
from nucleic import replicate_template, select_complement


def test_select_complement_is_watson_crick():
    # Base selection is driven by pairing affinity, and must reproduce the
    # Watson-Crick complement for every base.
    assert select_complement("A") == "T"
    assert select_complement("T") == "A"
    assert select_complement("G") == "C"
    assert select_complement("C") == "G"


# Base selection (and thus the daughter sequence) is deterministic and does not
# depend on how far the monomers dock, so the sequence-level tests use a short
# simulation to stay fast; the docking geometry is checked separately below.
FAST = {"steps_per_base": 80, "final_relax_steps": 0}


@pytest.mark.parametrize("template", ["ATGC", "GCGATTAC", "TTAAGGCC"])
def test_replication_reproduces_informational_complement(template):
    # The physically assembled daughter must equal dna_sim's complement — the
    # physical layer reproduces the informational operation.
    result = replicate_template(template, **FAST)
    assert result.daughter == str(DNASequence(template).complement())


def test_replicated_duplex_is_fully_paired():
    # With full relaxation the assembled daughter is physically paired to the
    # template all along its length.
    result = replicate_template("ATGC")
    assert result.paired_fraction(tolerance=1.0) == 1.0
    assert abs(result.mean_pair_distance() - 3.0) < 1.0


def test_daughter_read_5to3_is_reverse_complement():
    template = "GCGATTAC"
    result = replicate_template(template, **FAST)
    # The daughter, read in its own 5'->3' direction, is the antiparallel
    # partner: the reverse complement of the template.
    assert str(result.as_daughter_strand()) == str(
        DNASequence(template).reverse_complement()
    )


def test_replication_round_trip_regenerates_template():
    # Replicating a template yields its reverse-complement partner (5'->3').
    # Replicating that partner regenerates the original template — the
    # semiconservative logic of DNA replication.
    template = "ATGCGA"
    daughter = replicate_template(template, **FAST).as_daughter_strand()
    grand_daughter = replicate_template(daughter, **FAST).as_daughter_strand()
    assert str(grand_daughter) == template


def test_fidelity_selected_base_binds_alternatives_do_not():
    # The selected base has real pairing affinity; the other three have none,
    # which is why only the complement docks.
    from nucleic.pairing import pair_strength

    for tbase in "ACGT":
        chosen = select_complement(tbase)
        assert pair_strength(tbase, chosen) > 0
        others = [b for b in "ACGT" if b != chosen]
        assert all(pair_strength(tbase, b) == 0 for b in others)


def test_empty_template_rejected():
    with pytest.raises(ValueError):
        replicate_template("")


# --- Thermal fidelity: mutations emerging from temperature ---

import random

from nucleic import (
    replicate_sequence,
    replication_error_rate,
    select_base_thermal,
)


def test_zero_temperature_replication_is_perfect():
    template = "GCGATTACGC"
    assert replicate_sequence(template, temperature=0.0) == str(
        DNASequence(template).complement()
    )


def test_high_temperature_introduces_mutations():
    rate = replication_error_rate(
        "GCGATTACGCTAGCTA", temperature=15.0, trials=100, rng=random.Random(1)
    )
    assert rate > 0.0


def test_mutation_rate_increases_with_temperature():
    template = "GCGATTACGCTAGCTA"
    cold = replication_error_rate(template, 2.0, 200, random.Random(7))
    warm = replication_error_rate(template, 8.0, 200, random.Random(7))
    hot = replication_error_rate(template, 20.0, 200, random.Random(7))
    assert cold < warm < hot


def test_thermal_selection_low_temperature_prefers_complement():
    # At low temperature the complement is chosen overwhelmingly.
    rng = random.Random(0)
    picks = [select_base_thermal("A", temperature=1.0, rng=rng) for _ in range(200)]
    assert picks.count("T") > 190  # ~always the complement


def test_thermal_errors_are_point_substitutions_of_the_complement():
    # A thermally-mutated daughter has the same length as the perfect
    # complement and differs only by substitutions — i.e. point mutations.
    template = "GCGATTACGC"
    perfect = str(DNASequence(template).complement())
    daughter = replicate_sequence(template, temperature=25.0, rng=random.Random(3))
    assert len(daughter) == len(perfect)
    assert daughter != perfect  # at high T, some positions mutated


def test_physical_replication_accepts_temperature():
    # The full molecular-dynamics replication path also takes a temperature and
    # rng and still returns a same-length daughter.
    result = replicate_template(
        "ATGC", temperature=10.0, rng=random.Random(5), final_relax_steps=0,
        steps_per_base=80,
    )
    assert len(result.daughter) == 4
