from kt45.agent import CognitiveAgent
from kt45.facts import FactBase, Proposition
from kt45.reasoning import ForwardChainer, Rule, Z3Verifier
from kt45.transaction import Transaction


def test_forward_chainer_derives_with_axiom_T():
    world = FactBase()
    world.assert_fact(Proposition("mammal", ("Dog",)))
    world.assert_fact(Proposition("animal", ("Dog",)))  # required by axiom T

    rules = [
        Rule(
            name="mammal_implies_animal",
            premises=(Proposition("mammal", ("X",)),),
            conclusion=Proposition("animal", ("X",)),
        )
    ]
    a = CognitiveAgent("a")
    with Transaction(a, world, mode="repair") as tx:
        tx.set_known(Proposition("mammal", ("Dog",)))
    chainer = ForwardChainer(rules)
    derived = chainer.saturate(a, world)
    assert derived == 1
    assert Proposition("animal", ("Dog",)) in a._known
    assert Proposition("animal", ("Dog",)) in a._inferred


def test_forward_chainer_blocks_T_violating_inferences():
    world = FactBase()
    world.assert_fact(Proposition("mammal", ("Dog",)))
    # Note: do NOT add animal(Dog) to world.
    rules = [
        Rule(
            name="m_implies_a",
            premises=(Proposition("mammal", ("X",)),),
            conclusion=Proposition("animal", ("X",)),
        )
    ]
    a = CognitiveAgent("a")
    with Transaction(a, world, mode="repair") as tx:
        tx.set_known(Proposition("mammal", ("Dog",)))
    derived = ForwardChainer(rules).saturate(a, world)
    assert derived == 0
    assert Proposition("animal", ("Dog",)) not in a._known


def test_z3_verifier_basic_axiom_T():
    world = FactBase()
    p = Proposition("p", ())
    world.assert_fact(p)
    a = CognitiveAgent("a")
    with Transaction(a, world, mode="repair") as tx:
        tx.set_known(p)
    v = Z3Verifier()
    v.encode_agent_kt45(a, world, [p])
    assert v.check() == "sat"
    assert v.prove_axiom_T("a", p)
