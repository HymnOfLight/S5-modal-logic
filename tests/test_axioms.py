import pytest

from kt45.agent import CognitiveAgent
from kt45.axioms import KT45Checker, ViolationType
from kt45.facts import FactBase, Proposition
from kt45.transaction import Transaction, TransactionError


def make_world():
    fb = FactBase()
    fb.assert_fact(Proposition("p", ()))
    fb.assert_fact(Proposition("q", ()))
    return fb


def test_axiom_T_violated_by_unfounded_knowledge():
    world = make_world()
    a = CognitiveAgent("a")
    a._known.add(Proposition("not_in_world", ()))
    a._meta_knows_known.add(Proposition("not_in_world", ()))
    v = KT45Checker().check_all(a, world)
    assert any(x.kind is ViolationType.T_AXIOM for x in v)


def test_axiom_4_violated_when_meta_missing():
    world = make_world()
    a = CognitiveAgent("a")
    a._known.add(Proposition("p", ()))
    # deliberately omit meta entry
    v = KT45Checker().check_all(a, world)
    assert any(x.kind is ViolationType.AXIOM_4 for x in v)


def test_axiom_5_violated_when_meta_unknown_missing():
    world = make_world()
    a = CognitiveAgent("a")
    a._unknown.add(Proposition("p", ()))
    v = KT45Checker().check_all(a, world)
    assert any(x.kind is ViolationType.AXIOM_5 for x in v)


def test_repair_makes_invariants_hold():
    world = make_world()
    a = CognitiveAgent("a")
    a._known.add(Proposition("p", ()))
    a._known.add(Proposition("not_in_world", ()))  # T-violation
    a._unknown.add(Proposition("q", ()))           # 5-violation
    KT45Checker().repair(a, world)
    assert KT45Checker().check_all(a, world) == []


def test_strict_transaction_rolls_back_on_violation():
    world = make_world()
    a = CognitiveAgent("a")
    snap = (set(a._known), set(a._unknown))
    bogus = Proposition("bogus", ())
    with pytest.raises(TransactionError):
        with Transaction(a, world, mode="strict") as tx:
            tx.set_known(bogus)
    assert bogus not in a._known
    assert (a._known, a._unknown) == snap


def test_repair_transaction_succeeds():
    world = make_world()
    a = CognitiveAgent("a")
    bogus = Proposition("bogus", ())
    with Transaction(a, world, mode="repair") as tx:
        tx.set_known(bogus)
    # repair demoted bogus knowledge to UNKNOWN
    assert bogus not in a._known
    assert KT45Checker().check_all(a, world) == []
