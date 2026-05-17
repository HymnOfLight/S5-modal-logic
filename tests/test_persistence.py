import os
import tempfile

from kt45.agent import CognitiveAgent
from kt45.facts import FactBase, Proposition
from kt45.persistence import PersistenceManager
from kt45.transaction import Transaction


def test_world_round_trip():
    fb = FactBase()
    fb.assert_fact(Proposition("p", ("x",)))
    fb.assert_fact(Proposition("q", ("y", "z")))
    with tempfile.TemporaryDirectory() as d:
        pm = PersistenceManager(root=d)
        path = pm.save_world(fb)
        assert os.path.exists(path)
        fb2 = pm.load_world()
        assert set(fb._positive) == set(fb2._positive)


def test_agent_round_trip():
    fb = FactBase()
    p = Proposition("p", ())
    fb.assert_fact(p)
    a = CognitiveAgent("agent_x")
    with Transaction(a, fb, mode="repair") as tx:
        tx.set_known(p)
    with tempfile.TemporaryDirectory() as d:
        pm = PersistenceManager(root=d)
        pm.save_agents([a])
        loaded = pm.load_agents()
        assert len(loaded) == 1
        assert loaded[0].agent_id == "agent_x"
        assert p in loaded[0]._known
