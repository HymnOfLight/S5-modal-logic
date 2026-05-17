"""Persistence: dump and reload entire cognitive states.

We use newline-delimited JSON so a 100k-fact world dumps in a few
hundred milliseconds and the on-disk representation stays diff-able.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .agent import CognitiveAgent
from .facts import FactBase, Proposition


@dataclass
class PersistenceManager:
    """Save / load worlds and agent populations to a directory."""

    root: str

    def __post_init__(self) -> None:
        os.makedirs(self.root, exist_ok=True)

    def _path(self, name: str) -> str:
        return os.path.join(self.root, name)

    # ------------------------------------------------------------------
    def save_world(self, fb: FactBase, name: str = "world.jsonl") -> str:
        path = self._path(name)
        with open(path, "w", encoding="utf-8") as f:
            for prop in fb._positive:
                f.write(json.dumps({"v": "T", "p": str(prop)}) + "\n")
            for prop in fb._negative:
                f.write(json.dumps({"v": "NIL", "p": str(prop)}) + "\n")
        return path

    def load_world(self, name: str = "world.jsonl") -> FactBase:
        from .truth import T, NIL
        path = self._path(name)
        fb = FactBase()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                fb.assert_fact(Proposition.parse(row["p"]),
                               T if row["v"] == "T" else NIL)
        return fb

    # ------------------------------------------------------------------
    def save_agents(self, agents: Iterable[CognitiveAgent],
                    name: str = "agents.jsonl") -> str:
        path = self._path(name)
        with open(path, "w", encoding="utf-8") as f:
            for a in agents:
                f.write(json.dumps(a.to_state()) + "\n")
        return path

    def load_agents(self, name: str = "agents.jsonl") -> List[CognitiveAgent]:
        path = self._path(name)
        out: List[CognitiveAgent] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                out.append(CognitiveAgent.from_state(json.loads(line)))
        return out

    # ------------------------------------------------------------------
    def save_summary(self, payload: Dict, name: str = "summary.json") -> str:
        path = self._path(name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        return path
