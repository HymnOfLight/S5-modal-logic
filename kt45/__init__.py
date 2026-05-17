"""KT45 Cognitive System.

An engineering-grade multi-agent, temporal, parallel, persistent symbolic
engine built around the modal logic KT45 (a.k.a. S5).

KT45 axioms (system-level hard invariants):
    K : K(p -> q) -> (K p -> K q)         (distribution; baseline of K)
    T : K p -> p                            (truth / reflexivity)
    4 : K p -> K K p                        (positive introspection)
    5 : ~K p -> K ~K p                      (negative introspection)

Truth is two-valued (T / NIL).
"""

from .truth import T, NIL, TruthValue
from .facts import Fact, FactBase, Proposition
from .axioms import KT45Checker, AxiomViolation, ViolationType
from .transaction import Transaction, TransactionError
from .agent import CognitiveAgent, EpistemicState
from .group import AgentGroup, CommonKnowledge
from .reasoning import ForwardChainer, Rule, Z3Verifier
from .world import WorldFactory
from .persistence import PersistenceManager
from .parallel import ParallelEngine

__version__ = "1.0.0"

__all__ = [
    "T",
    "NIL",
    "TruthValue",
    "Fact",
    "FactBase",
    "Proposition",
    "KT45Checker",
    "AxiomViolation",
    "ViolationType",
    "Transaction",
    "TransactionError",
    "CognitiveAgent",
    "EpistemicState",
    "AgentGroup",
    "CommonKnowledge",
    "ForwardChainer",
    "Rule",
    "Z3Verifier",
    "WorldFactory",
    "PersistenceManager",
    "ParallelEngine",
]
