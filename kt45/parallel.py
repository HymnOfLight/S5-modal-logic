"""Parallel reasoning utilities.

CPython's GIL is the elephant in the room. For the KT45 workload we
parallelise *across agents* (independent L1 sets) and use
``multiprocessing`` so each worker gets its own GIL. Agents are
serialisable via :meth:`CognitiveAgent.to_state` / ``from_state``,
so we ship state to workers and merge results back.

Two task types are exposed:

* :func:`repair_population` — bulk axiom repair on a large agent
  cohort. Embarrassingly parallel.

* :func:`infer_population` — bulk forward chaining. Each worker
  saturates its assigned agents independently against a shared
  copy of the world.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .agent import CognitiveAgent
from .axioms import KT45Checker
from .facts import FactBase, Proposition
from .reasoning import ForwardChainer, Rule


# ---- worker helpers (must be top-level to be picklable) -------------------


def _repair_worker(args):
    states, world_state = args
    fb = FactBase()
    for v, s in world_state:
        fb.assert_fact(Proposition.parse(s),
                       __import__("kt45.truth", fromlist=["T", "NIL"]).T
                       if v == "T"
                       else __import__("kt45.truth", fromlist=["T", "NIL"]).NIL)
    checker = KT45Checker()
    out_states = []
    repaired_total = 0
    for st in states:
        a = CognitiveAgent.from_state(st)
        repaired = checker.repair(a, fb)
        repaired_total += len(repaired)
        out_states.append(a.to_state())
    return out_states, repaired_total


def _infer_worker(args):
    states, world_state, rules_payload = args
    from .truth import T, NIL
    fb = FactBase()
    for v, s in world_state:
        fb.assert_fact(Proposition.parse(s), T if v == "T" else NIL)
    rules = [
        Rule(
            name=r["name"],
            premises=tuple(Proposition.parse(p) for p in r["premises"]),
            conclusion=Proposition.parse(r["conclusion"]),
        )
        for r in rules_payload
    ]
    chainer = ForwardChainer(rules)
    out_states = []
    derived_total = 0
    for st in states:
        a = CognitiveAgent.from_state(st)
        derived_total += chainer.saturate(a, fb)
        out_states.append(a.to_state())
    return out_states, derived_total


def _serialize_world(world: FactBase):
    return [("T", str(p)) for p in world._positive] + \
           [("NIL", str(p)) for p in world._negative]


def _serialize_rules(rules: Sequence[Rule]):
    return [
        {
            "name": r.name,
            "premises": [str(p) for p in r.premises],
            "conclusion": str(r.conclusion),
        }
        for r in rules
    ]


# ---- public API ----------------------------------------------------------


@dataclass
class ParallelEngine:
    workers: int = max(1, (os.cpu_count() or 2))

    def repair_population(self, agents: List[CognitiveAgent],
                          world: FactBase) -> Tuple[List[CognitiveAgent], int, float]:
        """Run KT45 repair on every agent across ``workers`` processes.

        The original ``agents`` objects are mutated **in place** with the
        repaired state. The returned list is the same list that was
        passed in — this keeps caller-side references stable and avoids
        a class of bugs where post-parallel updates are silently lost
        because the caller forgot to reassign.
        """
        if not agents:
            return agents, 0, 0.0
        start = time.perf_counter()
        # Single-process fast path keeps small cohorts snappy.
        if self.workers == 1 or len(agents) <= 4:
            checker = KT45Checker()
            total = 0
            for a in agents:
                total += len(checker.repair(a, world))
            return agents, total, time.perf_counter() - start

        states = [a.to_state() for a in agents]
        world_state = _serialize_world(world)
        chunks = self._chunk(states, self.workers)
        with ProcessPoolExecutor(max_workers=self.workers) as ex:
            results = list(ex.map(
                _repair_worker,
                [(c, world_state) for c in chunks],
            ))
        repaired_total = 0
        # Index originals by id, then write worker output back into them.
        by_id = {a.agent_id: a for a in agents}
        for new_states, n in results:
            repaired_total += n
            for st in new_states:
                target = by_id.get(st["agent_id"])
                if target is not None:
                    self._copy_state_into(target, st)
                else:  # defensive: fall back to a fresh agent
                    agents.append(CognitiveAgent.from_state(st))
        return agents, repaired_total, time.perf_counter() - start

    def infer_population(self, agents: List[CognitiveAgent],
                         world: FactBase,
                         rules: Sequence[Rule]) -> Tuple[List[CognitiveAgent], int, float]:
        """Run forward-chaining saturation in parallel; mutates in place."""
        if not agents:
            return agents, 0, 0.0
        start = time.perf_counter()
        if self.workers == 1 or len(agents) <= 4:
            chainer = ForwardChainer(rules)
            total = 0
            for a in agents:
                total += chainer.saturate(a, world)
            return agents, total, time.perf_counter() - start

        states = [a.to_state() for a in agents]
        world_state = _serialize_world(world)
        rules_payload = _serialize_rules(rules)
        chunks = self._chunk(states, self.workers)
        with ProcessPoolExecutor(max_workers=self.workers) as ex:
            results = list(ex.map(
                _infer_worker,
                [(c, world_state, rules_payload) for c in chunks],
            ))
        derived_total = 0
        by_id = {a.agent_id: a for a in agents}
        for new_states, n in results:
            derived_total += n
            for st in new_states:
                target = by_id.get(st["agent_id"])
                if target is not None:
                    self._copy_state_into(target, st)
                else:
                    agents.append(CognitiveAgent.from_state(st))
        return agents, derived_total, time.perf_counter() - start

    @staticmethod
    def _copy_state_into(target: CognitiveAgent, state: Dict) -> None:
        """Replace ``target``'s mutable fields with those from ``state``."""
        fresh = CognitiveAgent.from_state(state)
        target._known = fresh._known
        target._unknown = fresh._unknown
        target._inferred = fresh._inferred
        target._meta_knows_known = fresh._meta_knows_known
        target._meta_knows_unknown = fresh._meta_knows_unknown
        target.metadata = fresh.metadata

    @staticmethod
    def _chunk(items: List, n: int) -> List[List]:
        if n <= 1 or len(items) <= n:
            return [items[i::n] for i in range(n)]
        size = (len(items) + n - 1) // n
        return [items[i:i + size] for i in range(0, len(items), size)]
