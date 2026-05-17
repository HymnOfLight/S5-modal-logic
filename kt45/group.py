"""Agent groups and common knowledge.

For a group ``G`` of agents:

* ``E_G(p)`` — *everyone knows* ``p``  ↔  ∀ a ∈ G : K_a p
* ``D_G(p)`` — *distributed knowledge* of ``p`` ↔ ∃ derivation
  using ⋃ a∈G K_a (here: union of L1 KNOWN sets).
* ``C_G(p)`` — *common knowledge* of ``p`` is the greatest fixpoint
  of ``E_G``: the largest set ``X`` such that ``X ⊆ E_G(X)``. Under
  KT45/S5 with positive and negative introspection this fixpoint is
  exactly ``⋂_{a ∈ G} K_a`` (no further iteration needed because
  axiom 4 already gives nested knowledge), which is what we compute
  here. We still ship the iterative fixpoint version
  (:meth:`AgentGroup.common_knowledge_fixpoint`) for verification and
  for debugging non-S5 fragments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Sequence, Set

from .agent import CognitiveAgent
from .facts import Proposition


@dataclass
class CommonKnowledge:
    group_id: str
    everyone_knows: FrozenSet[Proposition]
    distributed_knowledge: FrozenSet[Proposition]
    common_knowledge: FrozenSet[Proposition]
    iterations: int = 0

    def stats(self) -> Dict[str, int]:
        return {
            "E": len(self.everyone_knows),
            "D": len(self.distributed_knowledge),
            "C": len(self.common_knowledge),
            "iterations": self.iterations,
        }


class AgentGroup:
    """A finite set of agents together with group-knowledge operators."""

    __slots__ = ("group_id", "members")

    def __init__(self, group_id: str, members: Sequence[CognitiveAgent]) -> None:
        self.group_id = group_id
        self.members: List[CognitiveAgent] = list(members)

    def __len__(self) -> int:
        return len(self.members)

    def add(self, agent: CognitiveAgent) -> None:
        self.members.append(agent)

    # -- E_G and D_G ----------------------------------------------------
    def everyone_knows(self) -> FrozenSet[Proposition]:
        if not self.members:
            return frozenset()
        it = iter(self.members)
        result: Set[Proposition] = set(next(it)._known)
        for a in it:
            result &= a._known
            if not result:
                return frozenset()
        return frozenset(result)

    def distributed_knowledge(self) -> FrozenSet[Proposition]:
        result: Set[Proposition] = set()
        for a in self.members:
            result |= a._known
        return frozenset(result)

    # -- C_G ------------------------------------------------------------
    def common_knowledge(self) -> CommonKnowledge:
        """Fast S5 common-knowledge: ⋂ K_a (since axiom 4 closes K_a)."""
        ek = self.everyone_knows()
        return CommonKnowledge(
            group_id=self.group_id,
            everyone_knows=ek,
            distributed_knowledge=self.distributed_knowledge(),
            common_knowledge=ek,
            iterations=1,
        )

    def common_knowledge_fixpoint(self, max_iter: int = 32) -> CommonKnowledge:
        """Greatest fixpoint computation of ``E_G``.

        Starts from ``E_G(world)`` and iterates ``X_{n+1} = E_G(X_n)``.
        The first time ``X_{n+1} == X_n`` we have a fixpoint. With
        axiom 4 in force this converges in 1 iteration; we keep the
        loop for diagnostics and for non-S5 use cases.
        """
        candidate: Set[Proposition] = set(self.distributed_knowledge())
        iterations = 0
        for _ in range(max_iter):
            iterations += 1
            new: Set[Proposition] = set()
            for prop in candidate:
                if all(prop in a._known for a in self.members):
                    new.add(prop)
            if new == candidate:
                break
            candidate = new
        return CommonKnowledge(
            group_id=self.group_id,
            everyone_knows=self.everyone_knows(),
            distributed_knowledge=self.distributed_knowledge(),
            common_knowledge=frozenset(candidate),
            iterations=iterations,
        )

    # -- introspection --------------------------------------------------
    def stats(self) -> Dict[str, int]:
        return {
            "members": len(self.members),
            "E": len(self.everyone_knows()),
            "D": len(self.distributed_knowledge()),
        }
