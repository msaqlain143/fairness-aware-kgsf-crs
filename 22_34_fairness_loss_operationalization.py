#!/usr/bin/env python
"""Experiment 22.34: operationalize proposal L_fair from frozen 22.32 aggregates.

Read-only: no training, retrieval, candidate generation, ranking, target use,
or optimization. The proposal does not uniquely prescribe omega_c or R_min.
The default R_min=0 is a positive-support feasibility baseline only, not a
validated threshold. Select the final support threshold on validation data.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

EPSILON_0 = 1e-12


def find_input(root: Path) -> Path:
    candidates = [
        root / "results_final_test_m2/experiment_22_32/R_cg_rho_cg_reconstruction.csv",
        root / "results_final_test_m2/experiment_22_32/R_cg_reconstruction.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "22.32 reconstruction CSV not found under "
        "results_final_test_m2/experiment_22_32."
    )


def choose_col(df, names, label):
    for name in names:
        if name in df.columns:
            return name
    raise RuntimeError(
        f"Missing {label}; tried {names}. Available: {list(df.columns)}"
    )


def calculate(data: pd.DataFrame, r_min: float):
    # Aggregate repeated rows defensively.
    cells = (data.groupby(["category", "popularity_group"], as_index=False,
                          dropna=False)
             .agg(R_cg=("R_cg", "sum"), E_cg=("E_cg", "sum")))
    vals = cells[["R_cg", "E_cg"]].to_numpy(dtype=float)
    if not np.isfinite(vals).all() or (vals < 0).any():
        raise RuntimeError("R_cg/E_cg must be finite and non-negative.")

    # G_c+ support rule: R_cg >= R_min and strictly positive.
    active = cells[(cells.R_cg >= r_min) & (cells.R_cg > 0)].copy()
    if active.empty:
        raise RuntimeError(f"No supported cells at R_min={r_min}.")

    # Recompute proposal quantities on the active group set.
    active["rho_cg"] = active.E_cg / (active.R_cg + EPSILON_0)
    active["category_R"] = active.groupby("category").R_cg.transform("sum")
    active["p_cg"] = active.R_cg / active.category_R
    active["p_rho"] = active.p_cg * active.rho_cg
    means = (active.groupby("category", as_index=False).p_rho.sum()
             .rename(columns={"p_rho": "rho_bar_c"}))
    active = active.merge(means, on="category", how="left")
    active["variance_contribution"] = (
        active.p_cg * (active.rho_cg - active.rho_bar_c) ** 2
    )

    cat = (active.groupby("category", as_index=False)
           .agg(category_relevance_mass=("R_cg", "sum"),
                active_groups=("popularity_group", "nunique"),
                category_variance=("variance_contribution", "sum")))
    n_cat = len(cat)
    total_R = float(cat.category_relevance_mass.sum())
    cat["omega_uniform"] = 1.0 / n_cat
    cat["omega_relevance_mass"] = cat.category_relevance_mass / total_R

    loss_uniform = float((cat.omega_uniform * cat.category_variance).sum())
    loss_mass = float((cat.omega_relevance_mass * cat.category_variance).sum())

    summary = {
        "R_min": float(r_min),
        "epsilon_0": EPSILON_0,
        "input_category_group_cells": int(len(cells)),
        "active_category_group_cells": int(len(active)),
        "active_categories": int(n_cat),
        "total_active_relevance_mass": total_R,
        "L_fair_uniform_category_weighting": loss_uniform,
        "L_fair_relevance_mass_category_weighting": loss_mass,
        "omega_uniform": "1 / number of active categories",
        "omega_relevance_mass": "sum_g R_cg / sum_c sum_g R_cg",
        "R_min_status": (
            "POSITIVE_SUPPORT_BASELINE_ONLY_NOT_VALIDATED"
            if r_min == 0 else
            "USER_SUPPLIED_THRESHOLD_NOT_VALIDATED_BY_THIS_TEST_AUDIT"
        ),
    }
    return active, cat, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--input", default=None)
    ap.add_argument("--out-dir", default="results_final_test_m2/experiment_22_34")
    ap.add_argument("--r-min", type=float, default=0.0)
    args = ap.parse_args()
    if args.r_min < 0:
        raise ValueError("--r-min must be non-negative.")

    root = Path(args.project_root).resolve()
    source_path = Path(args.input).resolve() if args.input else find_input(root)
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(source_path)
    source.columns = [str(c).strip() for c in source.columns]
    cols = {
        "category": choose_col(source, ["category", "category_name", "genre"], "category"),
        "popularity_group": choose_col(
            source, ["popularity_group", "group", "pop_group"], "popularity group"),
        "R_cg": choose_col(source, ["R_cg", "R_c_g", "relevance_mass"], "R_cg"),
        "E_cg": choose_col(source, ["E_cg", "E_c_g", "exposure_mass"], "E_cg"),
    }
    data = source[list(cols.values())].copy()
    data.columns = list(cols.keys())
    data["category"] = data.category.astype(str)
    data["popularity_group"] = data.popularity_group.astype(str)
    data["R_cg"] = pd.to_numeric(data.R_cg, errors="raise")
    data["E_cg"] = pd.to_numeric(data.E_cg, errors="raise")

    active, cat, s = calculate(data, args.r_min)
    active.to_csv(out / "active_category_group_fairness_terms.csv", index=False)
    cat.to_csv(out / "category_weighting_and_variance.csv", index=False)
    pd.DataFrame([
        {"omega_scheme": "uniform_active_category",
         "L_fair": s["L_fair_uniform_category_weighting"],
         "primary_recommendation": True},
        {"omega_scheme": "category_relevance_mass",
         "L_fair": s["L_fair_relevance_mass_category_weighting"],
         "primary_recommendation": False},
    ]).to_csv(out / "L_fair_weighting_comparison.csv", index=False)

    decision = {
        "objective_computation": "COMPUTABLE_FROM_FROZEN_22_32_AGGREGATES",
        "primary_omega_operationalization": "UNIFORM_ACTIVE_CATEGORY",
        "sensitivity_omega": "CATEGORY_RELEVANCE_MASS",
        "R_min_finally_validated": False,
        "held_out_test_used_to_select_threshold": False,
        "next_step": (
            "Freeze R_min and relevance/eligibility thresholds using validation data; "
            "then implement the exposure-ledger reranker and evaluate on locked test."
        ),
        **s,
    }
    (out / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    manifest = {
        "experiment": "22.34_fairness_loss_operationalization",
        "proposal": "KG_FairCRS_Research_Proposal_20261117",
        "input": str(source_path),
        "read_only": True, "training": False, "retrieval": False,
        "ranking_changes": False, "target_used": False,
        "omega_c_unique_value_prescribed_by_proposal": False,
        "outputs": [
            "active_category_group_fairness_terms.csv",
            "category_weighting_and_variance.csv",
            "L_fair_weighting_comparison.csv", "decision.json", "report.txt",
            "experiment_manifest.json",
        ],
        **s,
    }
    (out / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    report = f"""Experiment 22.34 — Proposal Fairness-Loss Operationalization

Read-only: YES | Training: NO | Retrieval: NO | Ranking changes: NO
Target use: NO

Input: {source_path}
Input category-group cells: {s['input_category_group_cells']}
Active category-group cells: {s['active_category_group_cells']}
Active categories: {s['active_categories']}
R_min: {args.r_min}
R_min status: {s['R_min_status']}

L_fair, uniform active-category omega:
  {s['L_fair_uniform_category_weighting']:.12g}

L_fair, category relevance-mass omega (sensitivity):
  {s['L_fair_relevance_mass_category_weighting']:.12g}

Primary operationalization: omega_c = 1 / number of active categories.
Sensitivity: omega_c proportional to category relevance mass.
The proposal does not uniquely prescribe a numerical omega_c.

The default R_min=0 is only a positive-support feasibility baseline.
It is not a validated low-support threshold. Select and freeze R_min and
relevance/eligibility thresholds using validation data before final test.

Decision: L_fair is computationally operational from frozen 22.32
R_cg/E_cg aggregates under explicit omega_c and R_min choices. This
completes mathematical operationalization, not algorithm validation.

Next: implement eligibility gating, category/group exposure-deficit
calculation, fairness-adjusted scoring, Top-K ranking, and update the
exposure ledger only after recommendations are delivered.
"""
    (out / "report.txt").write_text(report, encoding="utf-8")

    print("=" * 72)
    print("EXPERIMENT 22.34 COMPLETE")
    print("=" * 72)
    print(f"Input cells: {s['input_category_group_cells']}")
    print(f"Active cells: {s['active_category_group_cells']}")
    print(f"Active categories: {s['active_categories']}")
    print(f"L_fair (uniform omega): {s['L_fair_uniform_category_weighting']:.12g}")
    print(f"L_fair (relevance-mass omega): {s['L_fair_relevance_mass_category_weighting']:.12g}")
    print(f"R_min status: {s['R_min_status']}")
    print("Read-only: YES | Training: NO | Retrieval: NO | Ranking changes: NO")
    print(f"Output: {out.resolve()}")


if __name__ == "__main__":
    main()
