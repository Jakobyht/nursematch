import pytest

from dna_sim.sequence import DNASequence, InvalidSequenceError


def test_normalises_to_uppercase():
    assert DNASequence("atgc").bases == "ATGC"


def test_rejects_invalid_characters():
    with pytest.raises(InvalidSequenceError):
        DNASequence("ATGX")


def test_complement():
    assert str(DNASequence("ATGC").complement()) == "TACG"


def test_reverse_complement():
    assert str(DNASequence("ATGC").reverse_complement()) == "GCAT"


def test_reverse_complement_involution():
    seq = DNASequence("ATGCCGTAAT")
    assert seq.reverse_complement().reverse_complement() == seq


def test_gc_content():
    assert DNASequence("GCGC").gc_content() == 1.0
    assert DNASequence("ATAT").gc_content() == 0.0
    assert DNASequence("ATGC").gc_content() == 0.5


def test_gc_content_empty():
    assert DNASequence("").gc_content() == 0.0


def test_base_counts_includes_all_bases():
    assert DNASequence("AAAG").base_counts() == {"A": 3, "T": 0, "G": 1, "C": 0}


def test_transcribe():
    assert DNASequence("ATGC").transcribe() == "AUGC"


def test_translate_start_to_stop():
    # AUG GCA UAA -> M A stop
    assert DNASequence("ATGGCATAA").translate() == "MA"


def test_translate_without_stop_trim():
    assert DNASequence("ATGGCATAA").translate(to_stop=False) == "MA*"


def test_find_orfs():
    # ORF: ATG AAA TAA on frame 0
    seq = DNASequence("ATGAAATAA")
    orfs = seq.find_orfs()
    assert (0, "MK") in orfs


def test_find_orfs_min_length_filter():
    seq = DNASequence("ATGTAA")  # M then stop -> 1 aa
    assert seq.find_orfs(min_codons=2) == []
    assert seq.find_orfs(min_codons=1) == [(0, "M")]
