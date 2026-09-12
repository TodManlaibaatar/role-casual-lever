#!/usr/bin/env python3
"""
15_within_condition_association.py

Does the layer-14 probe margin predict attack success *within* a condition?

Between conditions the margin and the behaviour come apart (the headline
result).  A reviewer will immediately ask whether the margin still carries
signal inside a single condition, because if it does, the probe is not simply
uninformative about behaviour.

This script answers that, and then checks the obvious confound: prompt-level
attackability.  Some prompts are easier to attack than others, and those same
prompts may sit at a higher baseline margin for reasons that have nothing to do
with any intervention.  If that is what is going on, then the *baseline*
margin should predict success inside the steered conditions about as well as
that condition's own margin does.

No scipy required - the p-values come from a label permutation test.

Usage
-----
    python 15_within_condition_association.py
    python 15_within_condition_association.py --results-dir results/final
    python 15_within_condition_association.py --per-population --n-perm 50000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEED = 20260911
N_PERM = 20000

POPULATIONS = [
    ("primary_component_latest.json", "Primary", ["baseline", "full", "aligned", "residual"]),
    ("fresh_heldout_latest.json", "Fresh A", ["baseline", "full", "aligned", "residual"]),
    ("fresh_heldout_extension_latest.json", "Fresh B", ["baseline", "full", "aligned", "residual"]),
    ("norm_matched_component_latest.json", "Equal norm", ["aligned_nm", "residual_nm"]),
]


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """P(score of a success > score of a failure), ties counted as half.

    Equivalent to the Mann-Whitney U statistic normalised to [0, 1].  0.5 means
    the margin carries no information about the outcome.
    """
    pos, neg = scores[labels], scores[~labels]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, dtype=float)
    ranks[order] = np.arange(1, order.size + 1)
    # average ranks over tied values
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    u = ranks[: pos.size].sum() - pos.size * (pos.size + 1) / 2
    return float(u / (pos.size * neg.size))


def perm_pvalue(scores, labels, rng, n_perm=N_PERM) -> float:
    """Two-sided permutation test on |AUC - 0.5|, shuffling the outcome labels."""
    obs = abs(rank_auc(scores, labels) - 0.5)
    lab = labels.copy()
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(lab)
        if abs(rank_auc(scores, lab) - 0.5) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def summarise(scores, labels, rng, n_perm):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    if labels.all() or (~labels).any() is False or labels.sum() == 0:
        return None
    return {
        "n": int(scores.size),
        "n_success": int(labels.sum()),
        "auc": rank_auc(scores, labels),
        "mean_margin_success": float(scores[labels].mean()),
        "mean_margin_failure": float(scores[~labels].mean()),
        "p_perm": perm_pvalue(scores, labels, rng, n_perm),
    }


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_records(results_dir: Path, fname: str):
    """prompt_id -> condition -> {margin, success} for scoreable trajectories."""
    with open(results_dir / fname) as fh:
        doc = json.load(fh)
    table: dict[str, dict[str, dict]] = {}
    for rec in doc["records"]:
        if not rec.get("evaluable"):
            continue
        readout = rec.get("downstream_readout") or {}
        if readout.get("layer14_margin") is None:
            continue
        table.setdefault(rec["prompt_id"], {})[rec["condition"]] = {
            "margin": float(readout["layer14_margin"]),
            "success": bool(rec["simulated_upload_success"]),
        }
    return table


def fmt(label, st):
    if st is None:
        return f"  {label:<26} (no outcome variation)"
    return (
        f"  {label:<26} n={st['n']:>3}  successes={st['n_success']:>3}  "
        f"AUC={st['auc']:.3f}  margin succ={st['mean_margin_success']:+6.2f} "
        f"fail={st['mean_margin_failure']:+6.2f}  p={st['p_perm']:.4f}"
    )


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/final", type=Path)
    ap.add_argument("--n-perm", default=N_PERM, type=int)
    ap.add_argument("--per-population", action="store_true",
                    help="also break the pooled numbers down by population")
    ap.add_argument("--out", default=None, type=Path,
                    help="optional path to write the results as JSON")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    out: dict = {"n_perm": args.n_perm, "seed": SEED}

    tables = {label: load_records(args.results_dir, f) for f, label, _ in POPULATIONS}

    # ---- 1. within-condition association, pooled across the three eval sets
    print("\n" + "=" * 96)
    print("1. Does the L14 margin predict success WITHIN a condition?")
    print("   Pooled over Primary / Fresh A / Fresh B (Equal norm reported separately).")
    print("=" * 96)

    eval_pops = ["Primary", "Fresh A", "Fresh B"]
    for cond in ["baseline", "full", "aligned", "residual"]:
        m, y = [], []
        for pop in eval_pops:
            for cmap in tables[pop].values():
                if cond in cmap:
                    m.append(cmap[cond]["margin"])
                    y.append(cmap[cond]["success"])
        st = summarise(m, y, rng, args.n_perm)
        out.setdefault("within_condition_pooled", {})[cond] = st
        print(fmt(cond, st))

    for cond in ["aligned_nm", "residual_nm"]:
        cmaps = tables["Equal norm"].values()
        m = [c[cond]["margin"] for c in cmaps if cond in c]
        y = [c[cond]["success"] for c in cmaps if cond in c]
        st = summarise(m, y, rng, args.n_perm)
        out.setdefault("within_condition_pooled", {})[cond] = st
        print(fmt(cond, st))

    # ---- 2. the confound check
    print("\n" + "=" * 96)
    print("2. CONFOUND CHECK: does each prompt's BASELINE margin predict success")
    print("   inside the steered conditions? If it predicts about as well as the")
    print("   condition's own margin, the association in (1) is prompt-level")
    print("   attackability rather than anything the intervention did.")
    print("=" * 96)

    for cond in ["baseline", "full", "aligned", "residual"]:
        m, y = [], []
        for pop in eval_pops:
            for cmap in tables[pop].values():
                if cond in cmap and "baseline" in cmap:
                    m.append(cmap["baseline"]["margin"])   # baseline margin
                    y.append(cmap[cond]["success"])        # steered outcome
        st = summarise(m, y, rng, args.n_perm)
        out.setdefault("baseline_margin_predicts", {})[cond] = st
        print(fmt(f"baseline margin -> {cond}", st))

    # ---- 3. does the SHIFT carry anything the baseline margin does not?
    print("\n" + "=" * 96)
    print("3. Does the per-prompt SHIFT (condition margin - baseline margin)")
    print("   predict success? This strips out the prompt-level level effect.")
    print("=" * 96)

    for cond in ["full", "aligned", "residual"]:
        m, y = [], []
        for pop in eval_pops:
            for cmap in tables[pop].values():
                if cond in cmap and "baseline" in cmap:
                    m.append(cmap[cond]["margin"] - cmap["baseline"]["margin"])
                    y.append(cmap[cond]["success"])
        st = summarise(m, y, rng, args.n_perm)
        out.setdefault("shift_predicts", {})[cond] = st
        print(fmt(f"shift -> {cond}", st))

    # ---- 4. optional per-population breakdown
    if args.per_population:
        print("\n" + "=" * 96)
        print("4. Within-condition association, by population")
        print("=" * 96)
        for pop, conds in [(p, c) for _f, p, c in POPULATIONS]:
            print(f"\n{pop}")
            for cond in conds:
                cmaps = tables[pop].values()
                m = [c[cond]["margin"] for c in cmaps if cond in c]
                y = [c[cond]["success"] for c in cmaps if cond in c]
                st = summarise(m, y, rng, args.n_perm)
                out.setdefault("per_population", {}).setdefault(pop, {})[cond] = st
                print(fmt(cond, st))

    print("\n" + "=" * 96)
    print("Reading the output: AUC 0.5 means the margin says nothing about the")
    print("outcome. Compare each row of (1) against the matching row of (2).")
    print("=" * 96 + "\n")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1, default=float)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()