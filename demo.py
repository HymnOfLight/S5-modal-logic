"""Engineering-grade KT45 demo.

Sequence:
  1. Build a 100,000-fact world (geography + math + commonsense).
  2. Spawn 100 cognitive agents (3-layer L0/L1/L2).
  3. Randomly assign knowledge, then run S5 axiom repair.
  4. Build 10 groups, compute E_G / D_G / C_G.
  5. Run forward chaining to materialise inferred knowledge.
  6. Parallel stress test (multi-process).

Run with::

    python demo.py
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import time
from typing import Dict, List

from kt45 import (
    AgentGroup,
    CognitiveAgent,
    EpistemicState,
    FactBase,
    ForwardChainer,
    KT45Checker,
    ParallelEngine,
    PersistenceManager,
    Proposition,
    Transaction,
    TransactionError,
    WorldFactory,
    Z3Verifier,
)


def banner(text: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n>> {text}\n{line}")


def step1_world(target_size: int) -> tuple:
    banner(f"STEP 1 — Build world ({target_size:,} facts)")
    t0 = time.perf_counter()
    fb, rules = WorldFactory(target_size=target_size).build()
    dt = time.perf_counter() - t0
    print(f"  facts asserted : {len(fb):>9,}")
    print(f"  predicates     : {len({p.predicate for p in fb._positive}):>9,}")
    print(f"  rules loaded   : {len(rules):>9,}")
    print(f"  build time     : {dt:>9.2f} s")
    return fb, rules


def step2_agents(n_agents: int) -> List[CognitiveAgent]:
    banner(f"STEP 2 — Create {n_agents} cognitive agents")
    agents = [CognitiveAgent(agent_id=f"agent_{i:03d}") for i in range(n_agents)]
    print(f"  {len(agents)} agents instantiated, each with empty L0/L1/L2")
    return agents


def step3_random_knowledge(agents: List[CognitiveAgent], world: FactBase,
                           per_agent: int, seed: int = 42,
                           shared_core_size: int = 50) -> Dict[str, int]:
    banner(f"STEP 3 — Random knowledge + S5 repair (~{per_agent} facts/agent)")
    rng = random.Random(seed)
    positive = list(world._positive)

    # A small "common-sense core" that EVERY agent learns. This guarantees
    # C_G(p) is non-empty for every group and lets us showcase common
    # knowledge in step 4 instead of an empty intersection.
    shared_core = rng.sample(positive, k=min(shared_core_size, len(positive)))

    t0 = time.perf_counter()
    bogus_props = [
        Proposition("not_in_world", (f"x{i}",)) for i in range(50)
    ]
    for a in agents:
        sample = rng.sample(positive, min(per_agent, len(positive)))
        bad = rng.sample(bogus_props, k=3)  # axiom-T violations
        a._known.update(sample)
        a._known.update(shared_core)
        a._known.update(bad)
        unknowns = rng.sample(positive, k=min(20, len(positive)))
        for p in unknowns:
            if p not in a._known:
                a._unknown.add(p)
    asg_dt = time.perf_counter() - t0
    print(f"  random assignment            : {asg_dt:.2f} s")

    # Validate: this should report many violations.
    checker = KT45Checker()
    t0 = time.perf_counter()
    pre_violations = sum(len(checker.check_all(a, world)) for a in agents)
    pre_dt = time.perf_counter() - t0
    print(f"  pre-repair KT45 violations   : {pre_violations:,} ({pre_dt:.2f} s)")

    # Run repair (the strict-then-repair pattern: each agent goes through
    # a Transaction to demonstrate the rollback path).
    t0 = time.perf_counter()
    repaired_total = 0
    for a in agents:
        with Transaction(a, world, mode="repair") as tx:
            pass  # commit triggers checker.check_all -> repair
        repaired_total += len(tx.repaired)
    repair_dt = time.perf_counter() - t0
    print(f"  repaired actions             : {repaired_total:,} ({repair_dt:.2f} s)")

    t0 = time.perf_counter()
    post_violations = sum(len(checker.check_all(a, world)) for a in agents)
    post_dt = time.perf_counter() - t0
    print(f"  post-repair KT45 violations  : {post_violations:,} ({post_dt:.2f} s)")
    assert post_violations == 0, "KT45 invariants must be restored after repair"

    # Demonstrate rollback path on a single bad transaction.
    a = agents[0]
    bogus = Proposition("definitely_false", ("z",))
    rolled_back = False
    try:
        with Transaction(a, world, mode="strict") as tx:
            tx.set_known(bogus)
    except TransactionError as e:
        rolled_back = True
        print(f"  strict-mode rollback demo    : {len(e.violations)} violation(s) detected, rolled back")
    assert rolled_back, "strict transaction must roll back on violation"
    assert bogus not in a._known, "rollback failed to restore agent state"

    return {
        "pre_violations": pre_violations,
        "post_violations": post_violations,
        "repaired_total": repaired_total,
        "assignment_seconds": round(asg_dt, 3),
        "repair_seconds": round(repair_dt, 3),
    }


def step4_groups(agents: List[CognitiveAgent], n_groups: int = 10) -> Dict:
    banner(f"STEP 4 — Build {n_groups} groups + common knowledge")
    rng = random.Random(7)
    # Stratified groups: each agent in roughly 1-2 groups.
    groups: List[AgentGroup] = []
    size = max(2, len(agents) // n_groups)
    for g in range(n_groups):
        members = rng.sample(agents, k=min(size + rng.randint(0, 5), len(agents)))
        groups.append(AgentGroup(group_id=f"G{g:02d}", members=members))

    summary: List[Dict] = []
    t0 = time.perf_counter()
    for g in groups:
        ck = g.common_knowledge()
        summary.append({"group": g.group_id, **ck.stats(),
                        "members": len(g)})
    fast_dt = time.perf_counter() - t0
    print(f"  fast S5 common-knowledge     : {fast_dt:.3f} s for {n_groups} groups")
    for s in summary:
        print(f"    {s['group']}  members={s['members']:>3}  "
              f"E={s['E']:>6,}  D={s['D']:>6,}  C={s['C']:>6,}")

    # Cross-check via the iterative greatest fixpoint algorithm on one group.
    t0 = time.perf_counter()
    ck_fix = groups[0].common_knowledge_fixpoint()
    fix_dt = time.perf_counter() - t0
    print(f"  fixpoint check on {groups[0].group_id}      : "
          f"|C|={len(ck_fix.common_knowledge):,} iter={ck_fix.iterations} ({fix_dt:.3f} s)")
    return {"groups": summary, "fixpoint_iter": ck_fix.iterations}


def step5_inference(agents: List[CognitiveAgent], world: FactBase, rules) -> Dict:
    banner("STEP 5 — Forward chaining inference")
    chainer = ForwardChainer(rules)
    t0 = time.perf_counter()
    total = 0
    for a in agents:
        total += chainer.saturate(a, world)
    dt = time.perf_counter() - t0
    inferred = sum(len(a._inferred) for a in agents)
    print(f"  total derivations            : {total:,}")
    print(f"  total INFERRED across cohort : {inferred:,}")
    print(f"  forward-chaining time        : {dt:.2f} s "
          f"({total / max(dt, 1e-9):,.0f} derivations/s)")

    # Final axiom check — inference must keep the agents KT45-compliant.
    checker = KT45Checker()
    bad = sum(len(checker.check_all(a, world)) for a in agents)
    print(f"  post-inference violations    : {bad}")
    assert bad == 0
    return {"derivations": total, "inferred_total": inferred,
            "inference_seconds": round(dt, 3)}


def step6_parallel(agents: List[CognitiveAgent], world: FactBase, rules) -> Dict:
    banner("STEP 6 — Parallel stress test")
    cpu = os.cpu_count() or 2
    engine = ParallelEngine(workers=cpu)
    print(f"  worker processes : {cpu}")
    # Re-randomise + plant fresh axiom violations so the parallel pipeline
    # has both repair and inference work to do.
    rng = random.Random(123)
    positive = list(world._positive)
    bogus = [Proposition("not_in_world_par", (f"y{i}",)) for i in range(200)]
    for a in agents:
        a._known.update(rng.sample(positive, k=min(500, len(positive))))
        a._known.update(rng.sample(bogus, k=5))   # T violations
        new_unknowns = rng.sample(positive, k=50)
        for p in new_unknowns:
            if p not in a._known:
                a._unknown.add(p)                 # 5 violations until repair
        # We deliberately do NOT update meta sets so axioms 4/5 will
        # require repair too.

    t0 = time.perf_counter()
    agents, repaired, par_repair_dt = engine.repair_population(agents, world)
    print(f"  parallel repair  : repaired={repaired:,} in {par_repair_dt:.2f} s")

    agents, derived, par_infer_dt = engine.infer_population(agents, world, rules)
    print(f"  parallel infer   : derived={derived:,} in {par_infer_dt:.2f} s")

    total_dt = time.perf_counter() - t0
    print(f"  total wall clock : {total_dt:.2f} s")
    return {
        "workers": cpu,
        "parallel_repair_seconds": round(par_repair_dt, 3),
        "parallel_infer_seconds": round(par_infer_dt, 3),
        "parallel_total_seconds": round(total_dt, 3),
        "parallel_repair_count": repaired,
        "parallel_infer_count": derived,
    }


def step7_z3(agents: List[CognitiveAgent], world: FactBase) -> Dict:
    banner("STEP 7 — Z3 spot-checks of KT45 axioms")
    verifier = Z3Verifier()
    sample_agent = agents[0]
    fragment = list(sample_agent._known)[:50]
    verifier.encode_agent_kt45(sample_agent, world, fragment)
    status = verifier.check()
    print(f"  agent={sample_agent.agent_id}  fragment_size={len(fragment)}  status={status}")
    assert status == "sat", "KT45 encoding must be satisfiable"
    proven = 0
    for p in fragment[:10]:
        if verifier.prove_axiom_T(sample_agent.agent_id, p):
            proven += 1
    print(f"  axiom T proved on {proven}/10 sampled propositions")
    return {"z3_status": status, "axiom_T_proved": proven}


def step8_persist(world: FactBase, agents: List[CognitiveAgent], summary: Dict) -> Dict:
    banner("STEP 8 — Persist run artefacts")
    pm = PersistenceManager(root="./kt45_run")
    t0 = time.perf_counter()
    p1 = pm.save_world(world)
    p2 = pm.save_agents(agents)
    p3 = pm.save_summary(summary)
    dt = time.perf_counter() - t0
    print(f"  wrote: {p1}\n         {p2}\n         {p3}\n  in {dt:.2f} s")
    return {"world_path": p1, "agents_path": p2, "summary_path": p3,
            "persist_seconds": round(dt, 3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", type=int, default=100_000)
    parser.add_argument("--agents", type=int, default=100)
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--per-agent-known", type=int, default=2_000,
                        help="approx. number of known facts per agent")
    args = parser.parse_args()

    overall_start = time.perf_counter()
    world, rules = step1_world(args.facts)
    agents = step2_agents(args.agents)
    s3 = step3_random_knowledge(agents, world, args.per_agent_known)
    s4 = step4_groups(agents, args.groups)
    s5 = step5_inference(agents, world, rules)
    s6 = step6_parallel(agents, world, rules)
    s7 = step7_z3(agents, world)
    summary = {
        "world_facts": len(world),
        "agents": len(agents),
        "groups": args.groups,
        "step3": s3, "step4": s4, "step5": s5,
        "step6": s6, "step7": s7,
        "wall_clock_seconds": round(time.perf_counter() - overall_start, 3),
    }
    s8 = step8_persist(world, agents, summary)
    summary["step8"] = s8

    banner("DONE")
    print(f"  total wall clock : {summary['wall_clock_seconds']:.2f} s")
    print(f"  artefacts under  : ./kt45_run/")


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
