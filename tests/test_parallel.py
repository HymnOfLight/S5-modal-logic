"""Regression tests for ParallelEngine.

Specifically guards the bug where ``repair_population`` /
``infer_population`` returned a *new* list of agents and silently
left the caller's original agent objects unrepaired.
"""
import multiprocessing as mp
import random

import pytest

from kt45 import (
    CognitiveAgent,
    FactBase,
    KT45Checker,
    ParallelEngine,
    Proposition,
    Rule,
)


@pytest.fixture(scope="module", autouse=True)
def _fork_start_method():
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass


def _world_with(positives):
    fb = FactBase()
    for p in positives:
        fb.assert_fact(p)
    return fb


def test_repair_population_mutates_originals_in_place():
    world = _world_with([Proposition("p", ()), Proposition("q", ())])
    bogus = Proposition("not_in_world", ("x",))
    agents = [CognitiveAgent(f"a{i}") for i in range(20)]
    for a in agents:
        a._known.add(Proposition("p", ()))
        a._known.add(bogus)

    original_ids = [id(a) for a in agents]
    engine = ParallelEngine(workers=4)
    out, repaired, _ = engine.repair_population(agents, world)

    # Same list object, same agent identities — the API must mutate in place.
    assert out is agents
    assert [id(a) for a in agents] == original_ids
    # Repair removed the bogus K p from every agent.
    for a in agents:
        assert bogus not in a._known
    # And the originals satisfy KT45 now.
    checker = KT45Checker()
    assert all(checker.check_all(a, world) == [] for a in agents)
    assert repaired >= len(agents)  # at least one fix per agent


def test_infer_population_mutates_originals_in_place():
    world = _world_with([
        Proposition("mammal", ("Dog",)),
        Proposition("animal", ("Dog",)),
        Proposition("mammal", ("Cat",)),
        Proposition("animal", ("Cat",)),
    ])
    rules = [Rule(
        name="m_implies_a",
        premises=(Proposition("mammal", ("X",)),),
        conclusion=Proposition("animal", ("X",)),
    )]

    agents = [CognitiveAgent(f"a{i}") for i in range(20)]
    for a in agents:
        a._known.add(Proposition("mammal", ("Dog",)))
        a._known.add(Proposition("mammal", ("Cat",)))
        # Ensure axioms 4/5 closure to keep agents pre-clean.
        a._meta_knows_known.update(a._known)

    original_ids = [id(a) for a in agents]
    engine = ParallelEngine(workers=4)
    out, derived, _ = engine.infer_population(agents, world, rules)

    assert out is agents
    assert [id(a) for a in agents] == original_ids
    for a in agents:
        assert Proposition("animal", ("Dog",)) in a._known
        assert Proposition("animal", ("Cat",)) in a._known
        assert Proposition("animal", ("Dog",)) in a._inferred
    assert derived >= 2 * len(agents)


def test_chained_repair_then_infer_keeps_KT45_invariant():
    """The exact regression that broke step 7 of the demo: plant
    T-violations, run parallel repair, then run parallel infer, and
    confirm the originals are clean (no bogus left in K)."""
    rng = random.Random(0)
    truths = [Proposition(f"t{i}", ()) for i in range(50)]
    world = _world_with(truths)
    bogus = [Proposition("not_in_world", (f"y{i}",)) for i in range(20)]
    agents = [CognitiveAgent(f"a{i}") for i in range(12)]
    for a in agents:
        a._known.update(rng.sample(truths, 10))
        a._known.update(rng.sample(bogus, 3))

    engine = ParallelEngine(workers=4)
    engine.repair_population(agents, world)
    engine.infer_population(agents, world, rules=[])
    bad = sum(1 for a in agents for p in a._known if p not in world._positive)
    assert bad == 0
    checker = KT45Checker()
    assert all(checker.check_all(a, world) == [] for a in agents)
