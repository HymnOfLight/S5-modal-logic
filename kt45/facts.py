"""Fact representation and fact base.

The bulk of facts in a KT45 world are atomic propositions of the form
``predicate(arg1, arg2, ...)``. We use string-interned tuples for cheap
hashing and 64-bit integer ids for the fast paths (sets, bitmaps).

The ``FactBase`` is closed-world: a proposition that is not present is
implicitly NIL. Negative facts can still be stored explicitly via the
``negative`` set, allowing the axiom checker to distinguish "open" from
"asserted-false" when needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, Optional, Set, Tuple

from .truth import T, NIL, TruthValue


@dataclass(frozen=True, slots=True)
class Proposition:
    """An atomic proposition: a predicate plus a tuple of argument symbols."""

    predicate: str
    args: Tuple[str, ...] = ()

    def __str__(self) -> str:
        if not self.args:
            return self.predicate
        return f"{self.predicate}(" + ",".join(self.args) + ")"

    @classmethod
    def parse(cls, text: str) -> "Proposition":
        text = text.strip()
        if "(" not in text:
            return cls(text, ())
        head, rest = text.split("(", 1)
        if not rest.endswith(")"):
            raise ValueError(f"Malformed proposition: {text!r}")
        inner = rest[:-1]
        args = tuple(a.strip() for a in inner.split(",")) if inner else ()
        return cls(head.strip(), args)


@dataclass(frozen=True, slots=True)
class Fact:
    """A signed atomic fact: ``proposition`` together with truth value."""

    proposition: Proposition
    value: TruthValue = T

    def __str__(self) -> str:
        return f"{self.proposition}={self.value.value}"


class FactBase:
    """High-throughput in-memory store of atomic facts.

    Backed by two ``set[Proposition]``s plus a string -> int interner so
    that downstream layers (agent L0/L1, group bitmaps) can index facts
    by integer ids in O(1).
    """

    __slots__ = ("_positive", "_negative", "_id_of", "_by_id", "_next_id")

    def __init__(self) -> None:
        self._positive: Set[Proposition] = set()
        self._negative: Set[Proposition] = set()
        self._id_of: Dict[Proposition, int] = {}
        self._by_id: Dict[int, Proposition] = {}
        self._next_id: int = 0

    def __len__(self) -> int:
        return len(self._positive) + len(self._negative)

    def __iter__(self) -> Iterator[Fact]:
        for p in self._positive:
            yield Fact(p, T)
        for p in self._negative:
            yield Fact(p, NIL)

    def positive(self) -> Iterable[Proposition]:
        return self._positive

    def negative(self) -> Iterable[Proposition]:
        return self._negative

    def has(self, prop: Proposition) -> bool:
        return prop in self._positive or prop in self._negative

    def truth_of(self, prop: Proposition) -> TruthValue:
        if prop in self._positive:
            return T
        return NIL

    def id_of(self, prop: Proposition) -> int:
        pid = self._id_of.get(prop)
        if pid is None:
            pid = self._next_id
            self._next_id += 1
            self._id_of[prop] = pid
            self._by_id[pid] = prop
        return pid

    def proposition_of(self, pid: int) -> Optional[Proposition]:
        return self._by_id.get(pid)

    def assert_fact(self, prop: Proposition, value: TruthValue = T) -> None:
        """Direct write — bypasses the axiom checker. Only use inside a
        ``Transaction`` or when you have already proved the write safe."""
        self.id_of(prop)
        if value is T:
            self._positive.add(prop)
            self._negative.discard(prop)
        else:
            self._negative.add(prop)
            self._positive.discard(prop)

    def retract(self, prop: Proposition) -> None:
        self._positive.discard(prop)
        self._negative.discard(prop)

    def snapshot(self) -> Tuple[frozenset, frozenset]:
        return frozenset(self._positive), frozenset(self._negative)

    def restore(self, snap: Tuple[frozenset, frozenset]) -> None:
        pos, neg = snap
        self._positive = set(pos)
        self._negative = set(neg)

    def stats(self) -> Dict[str, int]:
        return {
            "positive": len(self._positive),
            "negative": len(self._negative),
            "interned": len(self._id_of),
        }
