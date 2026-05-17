import pytest

from kt45.truth import T, NIL, TruthValue, coerce


def test_bivalent_only():
    assert T is TruthValue.T
    assert NIL is TruthValue.NIL
    assert ~T is NIL
    assert ~NIL is T


def test_logical_ops():
    assert (T & T) is T
    assert (T & NIL) is NIL
    assert (NIL | NIL) is NIL
    assert (NIL | T) is T
    assert bool(T) is True and bool(NIL) is False


def test_coerce_accepts_known_values():
    assert coerce(True) is T
    assert coerce(False) is NIL
    assert coerce("T") is T
    assert coerce("NIL") is NIL
    assert coerce(T) is T


def test_coerce_rejects_three_valued():
    with pytest.raises(ValueError):
        coerce(None)
    with pytest.raises(ValueError):
        coerce("MAYBE")
    with pytest.raises(ValueError):
        coerce(0.5)
