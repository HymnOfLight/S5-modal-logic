from kt45.agent import CognitiveAgent
from kt45.facts import FactBase, Proposition
from kt45.group import AgentGroup
from kt45.transaction import Transaction


def world_with(propositions):
    fb = FactBase()
    for p in propositions:
        fb.assert_fact(p)
    return fb


def test_everyone_knows_is_intersection():
    p = Proposition("p", ())
    q = Proposition("q", ())
    r = Proposition("r", ())
    world = world_with([p, q, r])
    a = CognitiveAgent("a")
    b = CognitiveAgent("b")
    c = CognitiveAgent("c")
    for ag in (a, b, c):
        with Transaction(ag, world, mode="repair") as tx:
            tx.set_known(p)
    with Transaction(a, world, mode="repair") as tx:
        tx.set_known(q)
    with Transaction(b, world, mode="repair") as tx:
        tx.set_known(q)
    g = AgentGroup("G", [a, b, c])
    assert p in g.everyone_knows()
    assert q not in g.everyone_knows()
    assert r not in g.everyone_knows()
    assert p in g.distributed_knowledge()
    assert q in g.distributed_knowledge()


def test_common_knowledge_fixpoint_converges():
    world = FactBase()
    props = [Proposition(f"p{i}", ()) for i in range(20)]
    for p in props:
        world.assert_fact(p)
    agents = [CognitiveAgent(f"a{i}") for i in range(5)]
    # all agents know props[:5]
    for a in agents:
        for p in props[:5]:
            with Transaction(a, world, mode="repair") as tx:
                tx.set_known(p)
    g = AgentGroup("G", agents)
    fast = g.common_knowledge()
    fix = g.common_knowledge_fixpoint()
    assert fast.common_knowledge == fix.common_knowledge
    assert len(fast.common_knowledge) == 5
