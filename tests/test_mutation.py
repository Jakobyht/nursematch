import random

from dna_sim.mutation import MutationModel, delete, insert, substitute
from dna_sim.sequence import DNASequence


def test_substitute_changes_base():
    rng = random.Random(0)
    seq = DNASequence("AAAA")
    mutated, event = substitute(seq, 1, rng)
    assert mutated.bases[1] != "A"
    assert event.kind == "substitution"
    assert event.position == 1
    assert event.before == "A"
    assert len(mutated) == len(seq)


def test_insert_grows_sequence():
    rng = random.Random(0)
    seq = DNASequence("AAAA")
    mutated, event = insert(seq, 2, rng)
    assert len(mutated) == len(seq) + 1
    assert event.kind == "insertion"


def test_delete_shrinks_sequence():
    rng = random.Random(0)
    seq = DNASequence("ATGC")
    mutated, event = delete(seq, 0, rng)
    assert str(mutated) == "TGC"
    assert event.kind == "deletion"
    assert event.before == "A"


def test_model_zero_rate_is_identity():
    rng = random.Random(0)
    model = MutationModel(0.0, 0.0, 0.0)
    seq = DNASequence("ATGCATGC")
    mutated, events = model.apply(seq, rng)
    assert mutated == seq
    assert events == []


def test_model_is_reproducible_with_seed():
    seq = DNASequence("ATGC" * 25)
    model = MutationModel(0.05, 0.01, 0.01)
    a, ea = model.apply(seq, random.Random(42))
    b, eb = model.apply(seq, random.Random(42))
    assert a == b
    assert ea == eb


def test_model_high_substitution_rate_mutates():
    rng = random.Random(1)
    model = MutationModel(substitution_rate=1.0, insertion_rate=0.0, deletion_rate=0.0)
    seq = DNASequence("AAAA")
    mutated, events = model.apply(seq, rng)
    # Every base substituted for a different base.
    assert all(b != "A" for b in mutated.bases)
    assert len(events) == 4


def test_model_rejects_rate_over_one():
    import pytest

    with pytest.raises(ValueError):
        MutationModel(0.6, 0.3, 0.3)
