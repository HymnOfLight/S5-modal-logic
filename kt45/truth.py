"""Bivalent truth values for KT45.

KT45 demands a strictly two-valued semantics. We expose only ``T`` and
``NIL`` as truth tokens. Anything else is rejected at runtime by the
axiom checker. ``TruthValue`` is intentionally a tiny enum-like object
because it sits on the hot path of every assertion.
"""
from __future__ import annotations

from enum import Enum
from typing import Final


class TruthValue(Enum):
    """The only two admissible truth values in the system."""

    T = "T"
    NIL = "NIL"

    def __invert__(self) -> "TruthValue":
        return TruthValue.NIL if self is TruthValue.T else TruthValue.T

    def __and__(self, other: "TruthValue") -> "TruthValue":
        return TruthValue.T if (self is TruthValue.T and other is TruthValue.T) else TruthValue.NIL

    def __or__(self, other: "TruthValue") -> "TruthValue":
        return TruthValue.T if (self is TruthValue.T or other is TruthValue.T) else TruthValue.NIL

    def __bool__(self) -> bool:
        return self is TruthValue.T

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


T: Final[TruthValue] = TruthValue.T
NIL: Final[TruthValue] = TruthValue.NIL


def coerce(value) -> TruthValue:
    """Coerce a Python-ish value into a strict ``TruthValue``.

    Anything outside ``{T, NIL, True, False, 'T', 'NIL'}`` raises so we
    never silently smuggle three-valued semantics in.
    """
    if isinstance(value, TruthValue):
        return value
    if value is True:
        return T
    if value is False:
        return NIL
    if isinstance(value, str):
        v = value.strip().upper()
        if v == "T":
            return T
        if v in ("NIL", "F"):
            return NIL
    raise ValueError(f"Non-bivalent value rejected by KT45: {value!r}")
