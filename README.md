# KT45 Cognitive System

An engineering-grade **multi-agent, temporal, parallel, persistent symbolic
engine** for the modal logic **KT45** (a.k.a. **S5**), implemented in
Python with **Z3** as the constraint solver providing logical support.

The KT45 axioms (T / 4 / 5) are enforced as **system-level hard
invariants**: every write to the cognitive store goes through an axiom
checker, and on violation the transaction is either rolled back
(`strict` mode) or auto-repaired (`repair` mode).

| Axiom | Modal form           | Operational meaning                            |
| ----- | -------------------- | ---------------------------------------------- |
| K     | K(p→q) → (Kp → Kq)   | distribution (kept by forward chainer)         |
| **T** | **K p → p**          | knowledge of *p* requires *p* in the world     |
| **4** | **K p → K K p**      | positive introspection (L1 → L2)               |
| **5** | **¬K p → K ¬K p**    | negative introspection (L1 → L2)               |

Truth is strictly **two-valued (T / NIL)** — anything else is rejected
at runtime. With 64 GB of RAM the engine can hold millions of facts
fully in memory; with 16 GB the bundled demo (100 000 facts × 100
agents) finishes in roughly **8 seconds wall-clock** on 4 CPU cores.

---

## Architecture

```
kt45/
├── truth.py        # Bivalent T / NIL (enum + coercion)
├── facts.py        # Proposition, Fact, FactBase (with int interning)
├── axioms.py       # KT45Checker, AxiomViolation, ViolationType
├── transaction.py  # Transactional writes with rollback / repair
├── agent.py        # CognitiveAgent: 3 layers (L0 / L1 / L2)
├── group.py        # AgentGroup: E_G, D_G, C_G (common knowledge)
├── reasoning.py    # ForwardChainer + Z3Verifier
├── world.py        # WorldFactory (geography + math + commonsense)
├── persistence.py  # PersistenceManager (JSONL save / load)
└── parallel.py     # ParallelEngine (multiprocessing pool)
demo.py             # Large-scale demo (100k facts, 100 agents)
tests/              # pytest suite (17 tests)
```

### Three-layer cognitive agent

* **L0 — World beliefs.** Which propositions the agent holds true.
  Closed-world: absence ⇒ NIL.
* **L1 — Epistemic state** per proposition: `KNOWN`, `UNKNOWN`, or
  `INFERRED` (i.e. KNOWN, but obtained via the forward chainer).
* **L2 — Meta-cognition.** Two sets — `meta_knows_known` (positive
  introspection, axiom 4) and `meta_knows_unknown` (negative
  introspection, axiom 5). The repair pass keeps L2 closed under L1.

### Group-level operators

For a group `G ⊆ Agents`:

* `E_G(p)` — *everyone knows*: `∩ K_a` over `a ∈ G`.
* `D_G(p)` — *distributed knowledge*: `∪ K_a` over `a ∈ G`.
* `C_G(p)` — *common knowledge*: greatest fixpoint of `E_G`. Under
  KT45/S5 (axiom 4) this collapses to `∩ K_a`. The iterative fixpoint
  algorithm is shipped alongside as `common_knowledge_fixpoint` for
  cross-checking and non-S5 fragments.

### Z3 backend

`Z3Verifier` propositionally encodes `K_a p`, `p_world`, axiom T (and
S5-collapsed 4 & 5) into Z3 and uses entailment-by-negation
(`¬φ` is unsat ⇒ `φ` holds) to prove obligations. The forward chainer
does the heavy lifting at scale; Z3 is the **formal conscience** of the
system that proves correctness on small fragments.

---

## Install

```bash
pip install -r requirements.txt          # only z3-solver
pip install pytest                       # for tests (optional)
```

## Demo

```bash
python demo.py                                # defaults: 100k facts, 100 agents
python demo.py --facts 200000 --agents 200    # bigger run
```

Sample output (16 GB RAM, 4 CPU cores):

```
STEP 1 — Build world (100,000 facts)              0.15 s
STEP 2 — Create 100 cognitive agents              instant
STEP 3 — Random knowledge + S5 repair             1.2 s
        pre-repair violations  : 207,463
        repaired actions       : 207,463
        post-repair violations : 0
        strict-mode rollback demo : OK
STEP 4 — 10 groups + common knowledge             0.02 s
        every group: E = D = C = 50
        fixpoint cross-check converges in 2 iters
STEP 5 — Forward chaining inference               0.42 s
        90,347 derivations  (≈215 K/s)
        post-inference violations : 0
STEP 6 — Parallel stress test (4 workers)         5.9 s
        parallel repair  : 54,408 fixes
        parallel infer   : 21,629 derivations
STEP 7 — Z3 spot-check                            <1 s
        encoding sat ; axiom T proved 10/10
STEP 8 — Persist artefacts                        0.4 s
        kt45_run/world.jsonl
        kt45_run/agents.jsonl
        kt45_run/summary.json

TOTAL wall clock : ~8 s
```

## Tests

```bash
python -m pytest tests/ -v
```

17 tests cover bivalent truth, every axiom violation kind, repair,
strict rollback, group operators, forward chaining (including the case
where axiom T blocks a derivation), Z3 verification, and JSONL
persistence round-trip.

---

## Programming the engine

Minimal example — observe a fact, get axiom 4 enforced automatically:

```python
from kt45 import (
    CognitiveAgent, FactBase, Proposition, Transaction, KT45Checker,
)

world = FactBase()
world.assert_fact(Proposition("capital_of", ("Beijing", "China")))

alice = CognitiveAgent("alice")
with Transaction(alice, world, mode="strict") as tx:
    tx.set_known(Proposition("capital_of", ("Beijing", "China")))

assert alice.knows(Proposition("capital_of", ("Beijing", "China")))
assert alice.knows_that_knows(Proposition("capital_of", ("Beijing", "China")))  # axiom 4
assert KT45Checker().check_all(alice, world) == []
```

Strict mode rejects axiom-violating writes by rollback:

```python
from kt45 import TransactionError
try:
    with Transaction(alice, world, mode="strict") as tx:
        tx.set_known(Proposition("capital_of", ("Atlantis", "Mu")))
except TransactionError as e:
    print(len(e.violations), "violation(s) — transaction rolled back")
```

Repair mode auto-fixes:

```python
with Transaction(alice, world, mode="repair") as tx:
    tx.set_known(Proposition("capital_of", ("Atlantis", "Mu")))
# Bogus knowledge demoted from KNOWN to UNKNOWN; meta sets updated.
```

Common knowledge over a group:

```python
from kt45 import AgentGroup
g = AgentGroup("team", [alice, bob, carol])
ck = g.common_knowledge()
print("|C_G| =", len(ck.common_knowledge))
```

Parallel inference across a 100-agent cohort:

```python
from kt45 import ParallelEngine
engine = ParallelEngine(workers=4)
agents, derived, dt = engine.infer_population(agents, world, rules)
```

---

## Design notes

* **Hot-path data structures.** Agents store sets of `Proposition`
  objects (frozen dataclasses with slots). The axiom checker runs in
  `O(|L1|)` set membership tests — checking a 100k-fact agent takes
  well under 100 ms.
* **Closed-world world model.** A proposition that is not in
  `FactBase._positive` is implicitly NIL. Negative facts can be made
  explicit when needed; the forward chainer respects axiom T by
  blocking any derivation whose conclusion is not positively present
  in the world.
* **Multiprocessing > threading.** Workers receive serialised agent
  states via `CognitiveAgent.to_state()`, sidestepping the GIL.
  `ProcessPoolExecutor` chunks agents across `os.cpu_count()` workers.
* **Persistence.** JSONL on disk — diff-friendly, streamable, and
  reload time scales linearly with the file size.

## License

MIT.
