"""Three-layer KT45 cognitive agent.

Every agent maintains:

* **L0 — World beliefs**: which propositions the agent treats as true
  in its model of the world. In a fully-truthful KT45 setting L0 is a
  subset of the world's positive facts.

* **L1 — Epistemic state**: per-proposition tag in ``{KNOWN, UNKNOWN,
  INFERRED}``. KNOWN comes from direct observation, INFERRED comes
  from the forward chainer, UNKNOWN is the explicit "I don't know"
  mark required by axiom 5.

* **L2 — Meta-cognition**: the agent's belief about its own L1. We
  represent it as two sets — ``meta_knows_known`` (positive
  introspection, axiom 4) and ``meta_knows_unknown`` (negative
  introspection, axiom 5).

The agent never bypasses the axiom checker for non-trivial mutations:
all writes go through :class:`kt45.transaction.Transaction`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Optional, Set

from .facts import FactBase, Proposition


class EpistemicState(Enum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    INFERRED = "INFERRED"


class CognitiveAgent:
    """A KT45-compliant cognitive agent.

    The internal sets are exposed as ``_known`` / ``_unknown`` / ... so
    the axiom checker, the transaction layer, and the forward chainer
    can manipulate them in tight loops. External code should prefer the
    higher-level methods (``believe_known``, ``epistemic_state``, ...)
    or use a ``Transaction`` context manager.
    """

    __slots__ = (
        "agent_id",
        "_known",
        "_unknown",
        "_inferred",
        "_meta_knows_known",
        "_meta_knows_unknown",
        "metadata",
    )

    def __init__(self, agent_id: str) -> None:
        self.agent_id: str = agent_id
        self._known: Set[Proposition] = set()
        self._unknown: Set[Proposition] = set()
        self._inferred: Set[Proposition] = set()
        self._meta_knows_known: Set[Proposition] = set()
        self._meta_knows_unknown: Set[Proposition] = set()
        self.metadata: Dict[str, object] = {}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"<CognitiveAgent {self.agent_id} "
            f"K={len(self._known)} U={len(self._unknown)} I={len(self._inferred)}>"
        )

    # -- L1 query API ---------------------------------------------------
    def epistemic_state(self, prop: Proposition) -> EpistemicState:
        if prop in self._known and prop in self._inferred:
            return EpistemicState.INFERRED
        if prop in self._known:
            return EpistemicState.KNOWN
        return EpistemicState.UNKNOWN

    def knows(self, prop: Proposition) -> bool:
        return prop in self._known

    def knows_that_knows(self, prop: Proposition) -> bool:
        return prop in self._meta_knows_known

    def knows_that_unknown(self, prop: Proposition) -> bool:
        return prop in self._meta_knows_unknown

    # -- bulk-ish convenience APIs (still go through Transaction) -------
    def believe_known(self, world: FactBase, prop: Proposition,
                      mode: str = "strict") -> "list":
        """Convenience wrapper around a single-write transaction."""
        from .transaction import Transaction  # local import to avoid cycle
        with Transaction(self, world, mode=mode) as tx:
            tx.set_known(prop)
        return tx.violations

    def declare_unknown(self, world: FactBase, prop: Proposition,
                        mode: str = "strict") -> "list":
        from .transaction import Transaction
        with Transaction(self, world, mode=mode) as tx:
            tx.set_unknown(prop)
        return tx.violations

    def stats(self) -> Dict[str, int]:
        return {
            "known": len(self._known),
            "unknown": len(self._unknown),
            "inferred": len(self._inferred),
            "meta_known": len(self._meta_knows_known),
            "meta_unknown": len(self._meta_knows_unknown),
        }

    # -- snapshot helpers ----------------------------------------------
    def to_state(self) -> Dict[str, list]:
        return {
            "agent_id": self.agent_id,
            "known": [str(p) for p in self._known],
            "unknown": [str(p) for p in self._unknown],
            "inferred": [str(p) for p in self._inferred],
            "meta_known": [str(p) for p in self._meta_knows_known],
            "meta_unknown": [str(p) for p in self._meta_knows_unknown],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_state(cls, state: Dict) -> "CognitiveAgent":
        a = cls(state["agent_id"])
        a._known = {Proposition.parse(s) for s in state.get("known", [])}
        a._unknown = {Proposition.parse(s) for s in state.get("unknown", [])}
        a._inferred = {Proposition.parse(s) for s in state.get("inferred", [])}
        a._meta_knows_known = {Proposition.parse(s) for s in state.get("meta_known", [])}
        a._meta_knows_unknown = {Proposition.parse(s) for s in state.get("meta_unknown", [])}
        a.metadata = dict(state.get("metadata", {}))
        return a
