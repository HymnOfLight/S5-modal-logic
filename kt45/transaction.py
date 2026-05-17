"""Transactional writes against the cognitive store.

Every mutation that needs to preserve the KT45 invariants must go
through a :class:`Transaction`. Transactions are context managers:

    with Transaction(agent, world, mode='strict') as tx:
        tx.set_known(prop)

On commit the axiom checker is invoked. ``mode='strict'`` rolls back
the transaction on any violation; ``mode='repair'`` invokes
``KT45Checker.repair`` and lets the transaction succeed with the
repaired state. The list of violations is always available afterwards
via :pyattr:`Transaction.violations`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from .axioms import AxiomViolation, KT45Checker
from .facts import FactBase, Proposition

if TYPE_CHECKING:  # pragma: no cover
    from .agent import CognitiveAgent


class TransactionError(RuntimeError):
    """Raised when a strict transaction must be rolled back."""

    def __init__(self, message: str, violations: List[AxiomViolation]) -> None:
        super().__init__(message)
        self.violations = violations


@dataclass
class _AgentSnapshot:
    known: frozenset
    unknown: frozenset
    inferred: frozenset
    meta_known: frozenset
    meta_unknown: frozenset


@dataclass
class Transaction:
    agent: "CognitiveAgent"
    world: FactBase
    mode: str = "strict"  # 'strict' | 'repair'
    checker: KT45Checker = field(default_factory=KT45Checker)

    _agent_snap: Optional[_AgentSnapshot] = field(default=None, init=False, repr=False)
    _world_snap: Optional[tuple] = field(default=None, init=False, repr=False)
    violations: List[AxiomViolation] = field(default_factory=list)
    repaired: List[AxiomViolation] = field(default_factory=list)
    committed: bool = field(default=False, init=False)

    def __enter__(self) -> "Transaction":
        a = self.agent
        self._agent_snap = _AgentSnapshot(
            known=frozenset(a._known),
            unknown=frozenset(a._unknown),
            inferred=frozenset(a._inferred),
            meta_known=frozenset(a._meta_knows_known),
            meta_unknown=frozenset(a._meta_knows_unknown),
        )
        self._world_snap = self.world.snapshot()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self._rollback()
            return False
        try:
            self._commit()
        except TransactionError:
            self._rollback()
            raise
        return False

    def set_known(self, prop: Proposition) -> None:
        a = self.agent
        a._known.add(prop)
        a._unknown.discard(prop)
        a._inferred.discard(prop)
        a._meta_knows_known.add(prop)
        a._meta_knows_unknown.discard(prop)

    def set_unknown(self, prop: Proposition) -> None:
        a = self.agent
        a._unknown.add(prop)
        a._known.discard(prop)
        a._inferred.discard(prop)
        a._meta_knows_unknown.add(prop)
        a._meta_knows_known.discard(prop)

    def set_inferred(self, prop: Proposition) -> None:
        a = self.agent
        a._inferred.add(prop)
        a._known.add(prop)
        a._unknown.discard(prop)
        a._meta_knows_known.add(prop)
        a._meta_knows_unknown.discard(prop)

    def assert_world(self, prop: Proposition, value=None) -> None:
        from .truth import T
        self.world.assert_fact(prop, value if value is not None else T)

    def _commit(self) -> None:
        viol = self.checker.check_all(self.agent, self.world)
        self.violations = viol
        if not viol:
            self.committed = True
            return
        if self.mode == "repair":
            self.repaired = self.checker.repair(self.agent, self.world)
            self.violations = self.checker.check_all(self.agent, self.world)
            if self.violations:
                raise TransactionError(
                    "KT45 invariants still broken after repair pass",
                    self.violations,
                )
            self.committed = True
            return
        raise TransactionError(
            f"KT45 invariants violated ({len(viol)} issue(s)); strict rollback",
            viol,
        )

    def _rollback(self) -> None:
        if self._agent_snap is not None:
            a = self.agent
            s = self._agent_snap
            a._known = set(s.known)
            a._unknown = set(s.unknown)
            a._inferred = set(s.inferred)
            a._meta_knows_known = set(s.meta_known)
            a._meta_knows_unknown = set(s.meta_unknown)
        if self._world_snap is not None:
            self.world.restore(self._world_snap)
        self.committed = False
