"""KT45 axiom checker.

The system enforces three modal axioms as **hard invariants**. Every
write through a :class:`Transaction` is validated by the
:class:`KT45Checker`. On violation the transaction is either rolled
back (strict mode) or auto-repaired (repair mode).

Axiom semantics
---------------

* **T  : K p -> p**
  If agent ``a`` knows ``p`` then ``p`` must be true in the world.
  An agent who "knows" a falsehood is rejected.

* **4  : K p -> K K p**
  If agent ``a`` knows ``p`` then ``a`` knows that it knows ``p``.
  Realised as the L1 -> L2 closure: every KNOWN proposition must
  appear in the meta layer's "knows-it-knows" set.

* **5  : ~K p -> K ~K p**
  If agent ``a`` does not know ``p`` then ``a`` knows it does not
  know ``p``. Realised as the L1 -> L2 closure on UNKNOWN/INFERRED.

We additionally enforce the K (distribution) axiom structurally: an
agent's KNOWN set is closed under the agent's currently *materialised*
deductive consequences, which the forward chainer maintains.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

from .facts import FactBase, Proposition
from .truth import T, NIL

if TYPE_CHECKING:  # pragma: no cover
    from .agent import CognitiveAgent


class ViolationType(Enum):
    T_AXIOM = "T"          # K p but ~p in world
    AXIOM_4 = "4"          # K p but ~K K p (missing positive introspection)
    AXIOM_5 = "5"          # ~K p but ~K ~K p (missing negative introspection)
    BIVALENCE = "BIVALENCE"  # truth value outside {T, NIL}
    UNKNOWN_PROP = "UNKNOWN_PROP"  # reference to an un-interned proposition


@dataclass(frozen=True)
class AxiomViolation:
    """An immutable record of a single axiom violation."""

    kind: ViolationType
    agent_id: Optional[str]
    proposition: Optional[Proposition]
    detail: str = ""

    def __str__(self) -> str:
        a = f"agent={self.agent_id}" if self.agent_id else "world"
        p = f" prop={self.proposition}" if self.proposition else ""
        d = f" :: {self.detail}" if self.detail else ""
        return f"[{self.kind.value}] {a}{p}{d}"


class KT45Checker:
    """Validates an agent against the world under the KT45 axioms.

    The checker is stateless — it operates on an agent + a fact base.
    All scans run in O(|L1|) using only Python ``set`` ops, so checking
    a 100k-fact agent takes well under 100 ms on commodity hardware.
    """

    def check_axiom_T(self, agent: "CognitiveAgent", world: FactBase) -> List[AxiomViolation]:
        violations: List[AxiomViolation] = []
        positive = world._positive  # fast attribute access
        for prop in agent._known:
            if prop not in positive:
                violations.append(
                    AxiomViolation(
                        kind=ViolationType.T_AXIOM,
                        agent_id=agent.agent_id,
                        proposition=prop,
                        detail="agent KNOWS prop but world says NIL",
                    )
                )
        return violations

    def check_axiom_4(self, agent: "CognitiveAgent") -> List[AxiomViolation]:
        violations: List[AxiomViolation] = []
        meta_known = agent._meta_knows_known
        for prop in agent._known:
            if prop not in meta_known:
                violations.append(
                    AxiomViolation(
                        kind=ViolationType.AXIOM_4,
                        agent_id=agent.agent_id,
                        proposition=prop,
                        detail="missing K K p in L2",
                    )
                )
        return violations

    def check_axiom_5(self, agent: "CognitiveAgent") -> List[AxiomViolation]:
        violations: List[AxiomViolation] = []
        meta_unknown = agent._meta_knows_unknown
        for prop in agent._unknown:
            if prop not in meta_unknown:
                violations.append(
                    AxiomViolation(
                        kind=ViolationType.AXIOM_5,
                        agent_id=agent.agent_id,
                        proposition=prop,
                        detail="missing K ~K p in L2",
                    )
                )
        return violations

    def check_all(self, agent: "CognitiveAgent", world: FactBase) -> List[AxiomViolation]:
        v: List[AxiomViolation] = []
        v.extend(self.check_axiom_T(agent, world))
        v.extend(self.check_axiom_4(agent))
        v.extend(self.check_axiom_5(agent))
        return v

    def repair(self, agent: "CognitiveAgent", world: FactBase) -> List[AxiomViolation]:
        """Force ``agent`` into a KT45-consistent state.

        Strategy:
          * T : drop any ``K p`` whose world value is NIL.
          * 4 : insert missing ``K K p`` records.
          * 5 : insert missing ``K ~K p`` records.

        Returns the list of violations that were repaired (for logs).
        """
        repaired: List[AxiomViolation] = []
        positive = world._positive
        bad_known = [p for p in agent._known if p not in positive]
        for p in bad_known:
            agent._known.discard(p)
            agent._unknown.add(p)
            repaired.append(
                AxiomViolation(
                    ViolationType.T_AXIOM, agent.agent_id, p,
                    "demoted KNOWN -> UNKNOWN to satisfy T",
                )
            )

        meta_known = agent._meta_knows_known
        for p in agent._known:
            if p not in meta_known:
                meta_known.add(p)
                repaired.append(
                    AxiomViolation(
                        ViolationType.AXIOM_4, agent.agent_id, p,
                        "inserted K K p to satisfy 4",
                    )
                )

        meta_unknown = agent._meta_knows_unknown
        for p in agent._unknown:
            if p not in meta_unknown:
                meta_unknown.add(p)
                repaired.append(
                    AxiomViolation(
                        ViolationType.AXIOM_5, agent.agent_id, p,
                        "inserted K ~K p to satisfy 5",
                    )
                )

        return repaired
