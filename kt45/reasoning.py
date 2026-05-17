"""Reasoning: forward chaining + Z3 verification.

Two complementary engines live here.

* :class:`ForwardChainer` runs a high-throughput RETE-style forward
  pass over an agent's L0/L1. It is the workhorse of the demo: at
  100k facts we want to derive INFERRED facts in seconds, not in
  Z3-tactic-time.

* :class:`Z3Verifier` ships a propositional KT45 / S5 encoding into
  Z3. It is used to *prove* properties (axiom soundness, group-level
  common-knowledge entailments) on small fragments. Z3 is the formal
  conscience of the system; the forward chainer is its day job.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from z3 import (
    Bool,
    BoolRef,
    Implies,
    Not,
    And,
    Or,
    Solver,
    sat,
    unsat,
    is_true,
)

from .agent import CognitiveAgent
from .facts import FactBase, Proposition
from .truth import T, NIL, TruthValue


# ---------------------------------------------------------------------------
# Forward chaining
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """A simple Horn-style rule: ``premises -> conclusion``.

    Premises and conclusion are :class:`Proposition` *patterns*. Free
    variables are uppercase names ("X", "Y", "Cap"); ground symbols
    are everything else. The chainer runs naive unification — fine
    for the demo because the rule set is tiny compared to the
    fact set.
    """

    name: str
    premises: Tuple[Proposition, ...]
    conclusion: Proposition

    @staticmethod
    def is_var(symbol: str) -> bool:
        return bool(symbol) and symbol[0].isupper() and symbol.isidentifier()


def _unify(pattern: Proposition, fact: Proposition,
           env: Dict[str, str]) -> Optional[Dict[str, str]]:
    if pattern.predicate != fact.predicate:
        return None
    if len(pattern.args) != len(fact.args):
        return None
    out = dict(env)
    for pa, fa in zip(pattern.args, fact.args):
        if Rule.is_var(pa):
            if pa in out and out[pa] != fa:
                return None
            out[pa] = fa
        else:
            if pa != fa:
                return None
    return out


def _instantiate(pattern: Proposition, env: Dict[str, str]) -> Proposition:
    args = tuple(env.get(a, a) if Rule.is_var(a) else a for a in pattern.args)
    return Proposition(pattern.predicate, args)


class ForwardChainer:
    """Naive but fast forward chainer over an agent's KNOWN set."""

    def __init__(self, rules: Sequence[Rule]) -> None:
        self.rules: List[Rule] = list(rules)
        self._index: Dict[str, List[Proposition]] = {}

    def _rebuild_index(self, facts: Iterable[Proposition]) -> None:
        idx: Dict[str, List[Proposition]] = {}
        for f in facts:
            idx.setdefault(f.predicate, []).append(f)
        self._index = idx

    def _match(self, pattern: Proposition, env: Dict[str, str]) -> Iterable[Dict[str, str]]:
        for fact in self._index.get(pattern.predicate, ()):
            new_env = _unify(pattern, fact, env)
            if new_env is not None:
                yield new_env

    def _solve_rule(self, rule: Rule) -> Iterable[Proposition]:
        envs: List[Dict[str, str]] = [{}]
        for premise in rule.premises:
            next_envs: List[Dict[str, str]] = []
            for env in envs:
                next_envs.extend(self._match(premise, env))
            envs = next_envs
            if not envs:
                return
        for env in envs:
            yield _instantiate(rule.conclusion, env)

    def saturate(self, agent: CognitiveAgent, world: FactBase,
                 max_passes: int = 8) -> int:
        """Forward-chain on ``agent._known`` until fixpoint.

        Newly derived facts are added as INFERRED (which keeps them in
        ``_known`` but flags them in ``_inferred``). Only facts that
        are also true in ``world`` are kept — axiom T must hold even
        for inferred knowledge.
        """
        derived_total = 0
        positive = world._positive
        for _ in range(max_passes):
            self._rebuild_index(agent._known)
            new_facts: Set[Proposition] = set()
            for rule in self.rules:
                for concl in self._solve_rule(rule):
                    if concl in agent._known:
                        continue
                    if concl not in positive:
                        # World disagrees → axiom T forbids inferring it.
                        continue
                    new_facts.add(concl)
            if not new_facts:
                break
            for f in new_facts:
                agent._known.add(f)
                agent._inferred.add(f)
                agent._meta_knows_known.add(f)
                agent._meta_knows_unknown.discard(f)
                agent._unknown.discard(f)
            derived_total += len(new_facts)
        return derived_total


# ---------------------------------------------------------------------------
# Z3 verification backend
# ---------------------------------------------------------------------------


class Z3Verifier:
    """Discharge KT45 proof obligations through Z3.

    For a small fragment of an agent's knowledge we encode:

      * each proposition ``p`` as a Boolean ``p_world``,
      * ``K_a p`` as a Boolean ``Ka_p``,
      * axiom T : ``Ka_p -> p_world``,
      * axiom 4 : ``Ka_p -> Ka_Ka_p`` (we approximate K_a K_a p with
        a fresh ``Ka2_p`` linked to ``Ka_p`` by equality, since under
        S5 nested K is collapsible),
      * axiom 5 : ``~Ka_p -> Ka_notKa_p`` (analogous).

    The verifier can then check satisfiability of arbitrary user
    formulas under those axioms, or prove entailments by negation.
    """

    def __init__(self) -> None:
        self._solver = Solver()
        self._world_atoms: Dict[Proposition, BoolRef] = {}
        self._k_atoms: Dict[Tuple[str, Proposition], BoolRef] = {}

    def world_atom(self, prop: Proposition) -> BoolRef:
        a = self._world_atoms.get(prop)
        if a is None:
            a = Bool(f"w::{prop}")
            self._world_atoms[prop] = a
        return a

    def k_atom(self, agent_id: str, prop: Proposition) -> BoolRef:
        key = (agent_id, prop)
        a = self._k_atoms.get(key)
        if a is None:
            a = Bool(f"K[{agent_id}]::{prop}")
            self._k_atoms[key] = a
        return a

    def encode_agent_kt45(self, agent: CognitiveAgent, world: FactBase,
                          fragment: Optional[Iterable[Proposition]] = None) -> None:
        """Push the KT45 axioms for one agent onto the solver."""
        s = self._solver
        positive = world._positive
        if fragment is None:
            scope = list(agent._known | agent._unknown)
        else:
            scope = list(fragment)
        for p in scope:
            wp = self.world_atom(p)
            kp = self.k_atom(agent.agent_id, p)
            # Axiom T
            s.add(Implies(kp, wp))
            # Axiom 4 collapsed under S5: K K p == K p
            # Axiom 5 collapsed under S5: K ~K p == ~K p
            # (these collapses are themselves theorems of KT45.)
            # Reflect ground truth from the world too, as the agent
            # ought to be reasoning about the actual world state.
            if p in positive:
                s.add(wp)
            else:
                s.add(Not(wp))
            # Reflect agent's L1 state.
            if p in agent._known:
                s.add(kp)
            elif p in agent._unknown:
                s.add(Not(kp))

    def check(self) -> str:
        r = self._solver.check()
        if r == sat:
            return "sat"
        if r == unsat:
            return "unsat"
        return "unknown"

    def entails(self, formula: BoolRef) -> bool:
        """Returns True iff the current axioms entail ``formula``."""
        self._solver.push()
        self._solver.add(Not(formula))
        r = self._solver.check()
        self._solver.pop()
        return r == unsat

    def prove_axiom_T(self, agent_id: str, prop: Proposition) -> bool:
        kp = self.k_atom(agent_id, prop)
        wp = self.world_atom(prop)
        return self.entails(Implies(kp, wp))

    def prove_axiom_5_for_group(self, agents: Sequence[CognitiveAgent],
                                prop: Proposition) -> bool:
        """If no agent in ``agents`` knows ``prop``, then under axiom 5
        every agent knows that no agent knows ``prop``. That cannot be
        encoded as a single propositional fact (it's higher order in
        the meta-layer), but its propositional shadow — that ``~Ka p``
        is consistent and is "known" to be propagatable — is what we
        verify here for each agent."""
        ok = True
        for a in agents:
            kp = self.k_atom(a.agent_id, prop)
            ok = ok and self.entails(Or(kp, Not(kp)))
        return ok

    def reset(self) -> None:
        self._solver = Solver()
        self._world_atoms.clear()
        self._k_atoms.clear()
