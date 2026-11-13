#!/usr/bin/env python
"""
Experiment 22.17: Cross-backbone generalization evidence synthesis.

Reuses completed validation artifacts from:
  - KGSF primary-backbone Experiment 22.00
  - KBRD second-backbone Experiment 22.09

No training, retrieval, coefficient search, reranking, or test-set evaluation
is performed. This script audits comparable baseline-to-treatment changes
within each backbone and reports whether their directions agree.

Important: the two backbones use separately selected validation policies.
This is a descriptive cross-backbone generalization assessment, not a
pooled causal estimate or a formal backbone-by-treatment interaction test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a Markdown table without requiring the optional tabulate package."""
    columns = [str(column) for column in frame.columns]

    def format_cell(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6g}"
        return str(value).replace("|", r"\\|").replace("\\r", " ").replace("\\n", " ")

    rows = [
        [format_cell(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\\n".join([header, separator, *body])



DEFAULT_KGSF = Path("results_corrected/experiment_22_00")
DEFAULT_KBRD = Path("results_corrected/experiment_22_09")
DEFAULT_OUT = Path("results_corrected/experiment_22_17")


def load_csv(path: Path, required: set[str], label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{label} is missing columns {sorted(missing)}; "
            f"available={frame.columns.tolist()}"
        )
    if frame.empty:
        raise ValueError(f"{label} is empty: {path}")
    return frame


def load_json(path: Path, label: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def one_row(frame: pd.DataFrame, method: str, label: str) -> pd.Series:
    rows = frame.loc[frame["method"].astype(str) == method]
    if len(rows) != 1:
        raise ValueError(
            f"{label}: expected exactly one row for {method!r}, got {len(rows)}"
        )
    return rows.iloc[0]


def finite_float(value, label: str) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} is not finite: {value}")
    return result


def compare_backbone(
    backbone: str,
    baseline_method: str,
    treatment_method: str,
    baseline: pd.Series,
    treatment: pd.Series,
    selected_lambda: float | None,
) -> dict:
    b_hr = finite_float(baseline["HR@10"], f"{backbone} baseline HR@10")
    t_hr = finite_float(treatment["HR@10"], f"{backbone} treatment HR@10")
    b_ndcg = finite_float(baseline["NDCG@10"], f"{backbone} baseline NDCG@10")
    t_ndcg = finite_float(treatment["NDCG@10"], f"{backbone} treatment NDCG@10")
    b_fair = finite_float(
        baseline["category_group_log_rate_MAD"],
        f"{backbone} baseline fairness MAD",
    )
    t_fair = finite_float(
        treatment["category_group_log_rate_MAD"],
        f"{backbone} treatment fairness MAD",
    )

    return {
        "backbone": backbone,
        "baseline_method": baseline_method,
        "treatment_method": treatment_method,
        "selected_lambda": selected_lambda,
        "events_baseline": int(baseline["events"]) if "events" in baseline else None,
        "events_treatment": int(treatment["events"]) if "events" in treatment else None,
        "baseline_HR@10": b_hr,
        "treatment_HR@10": t_hr,
        "delta_HR@10": t_hr - b_hr,
        "baseline_NDCG@10": b_ndcg,
        "treatment_NDCG@10": t_ndcg,
        "delta_NDCG@10": t_ndcg - b_ndcg,
        "NDCG_retention": t_ndcg / (b_ndcg + EPS),
        "baseline_category_group_log_rate_MAD": b_fair,
        "treatment_category_group_log_rate_MAD": t_fair,
        "delta_category_group_log_rate_MAD": t_fair - b_fair,
        "fairness_relative_change_pct": (
            100.0 * (t_fair - b_fair) / (abs(b_fair) + EPS)
        ),
        "fairness_improved": bool(t_fair < b_fair),
        "utility_retained_at_95pct": bool(t_ndcg >= 0.95 * b_ndcg),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kgsf-dir", type=Path, default=DEFAULT_KGSF)
    parser.add_argument("--kbrd-dir", type=Path, default=DEFAULT_KBRD)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # KGSF: the primary-backbone fairness policy is A3_KG_FairCRS,
    # compared with its frozen KG-rank-fusion reference A1_frequency_3.
    kgsf_metrics = load_csv(
        args.kgsf_dir / "relevance_fairness_comparison.csv",
        {"method", "HR@10", "NDCG@10", "category_group_log_rate_MAD"},
        "KGSF comparison",
    )
    kgsf_checks = load_json(
        args.kgsf_dir / "validation_checks.json", "KGSF validation checks"
    )
    kgsf_summary = load_json(
        args.kgsf_dir / "validation_summary.json", "KGSF validation summary"
    )
    kgsf_baseline = one_row(kgsf_metrics, "A1_frequency_3", "KGSF comparison")
    kgsf_treatment = one_row(kgsf_metrics, "A3_KG_FairCRS", "KGSF comparison")

    kgsf_lambda = kgsf_summary.get("selected_lambda")
    if isinstance(kgsf_lambda, dict):
        kgsf_lambda = kgsf_lambda.get("A3_KG_FairCRS")
    if kgsf_lambda is None:
        kgsf_lambda = 0.8  # documented Experiment 22.00 selection
    kgsf_lambda = float(kgsf_lambda)

    kgsf_result = compare_backbone(
        "KGSF",
        "A1_frequency_3",
        "A3_KG_FairCRS",
        kgsf_baseline,
        kgsf_treatment,
        kgsf_lambda,
    )

    # KBRD: use the already completed cross-backbone experiment.
    kbrd_points = load_csv(
        args.kbrd_dir / "point_estimates.csv",
        {"method", "HR@10", "NDCG@10"},
        "KBRD point estimates",
    )
    kbrd_fair = load_csv(
        args.kbrd_dir / "category_fairness_results.csv",
        {"method", "category_group_log_rate_MAD"},
        "KBRD category fairness",
    )
    kbrd_checks = load_json(
        args.kbrd_dir / "validation_checks.json", "KBRD validation checks"
    )

    kbrd_baseline_point = one_row(kbrd_points, "B0_KBRD", "KBRD point estimates")
    kbrd_treatment_point = one_row(
        kbrd_points, "B2_KBRD_KG_gated", "KBRD point estimates"
    )
    kbrd_baseline_fair = one_row(
        kbrd_fair, "B0_KBRD", "KBRD fairness results"
    )
    kbrd_treatment_fair = one_row(
        kbrd_fair, "B2_KBRD_KG_gated", "KBRD fairness results"
    )

    # Join metric rows only after independently checking unique method rows.
    kbrd_baseline = pd.concat([kbrd_baseline_point, kbrd_baseline_fair]).to_dict()
    kbrd_treatment = pd.concat([kbrd_treatment_point, kbrd_treatment_fair]).to_dict()

    kbrd_selected = kbrd_checks.get("selected_lambda", {})
    kbrd_lambda = (
        kbrd_selected.get("B2_KBRD_KG_gated")
        if isinstance(kbrd_selected, dict) else None
    )
    kbrd_result = compare_backbone(
        "KBRD",
        "B0_KBRD",
        "B2_KBRD_KG_gated",
        pd.Series(kbrd_baseline),
        pd.Series(kbrd_treatment),
        None if kbrd_lambda is None else float(kbrd_lambda),
    )

    # Confirm that both inputs are validation-only and did not access test data.
    kgsf_validation_only = (
        kgsf_checks.get("split") == "validation_only"
        or kgsf_checks.get("validation_only") is True
    )
    kgsf_test_not_evaluated = (
        kgsf_checks.get("test_split_evaluated") is False
    )
    kbrd_validation_only = kbrd_checks.get("validation_only") is True
    kbrd_test_not_evaluated = kbrd_checks.get("test_split_evaluated") is False

    rows = [kgsf_result, kbrd_result]
    result_df = pd.DataFrame(rows)
    result_df.to_csv(args.out / "cross_backbone_comparison.csv", index=False)

    direction_agreement = (
        kgsf_result["fairness_improved"] == kbrd_result["fairness_improved"]
    )
    both_fairness_improved = (
        kgsf_result["fairness_improved"] and kbrd_result["fairness_improved"]
    )
    both_utility_feasible = (
        kgsf_result["utility_retained_at_95pct"]
        and kbrd_result["utility_retained_at_95pct"]
    )

    # Do not call the model-agnostic claim confirmed merely because point
    # estimates are directionally aligned. This is a two-backbone validation
    # with separately selected policies and no pooled interaction test.
    checks = {
        "experiment": "22.17_cross_backbone_generalization_synthesis",
        "overall_pass": bool(
            kgsf_validation_only and kgsf_test_not_evaluated
            and kbrd_validation_only and kbrd_test_not_evaluated
        ),
        "no_training_performed": True,
        "no_retrieval_rerun": True,
        "no_coefficient_search": True,
        "no_test_set_tuning": True,
        "kgsf_validation_only": kgsf_validation_only,
        "kgsf_test_split_not_evaluated": kgsf_test_not_evaluated,
        "kbrd_validation_only": kbrd_validation_only,
        "kbrd_test_split_not_evaluated": kbrd_test_not_evaluated,
        "backbones_compared": ["KGSF", "KBRD"],
        "within_backbone_baseline_treatment_comparisons": True,
        "fairness_direction_agrees_across_backbones": bool(direction_agreement),
        "fairness_improves_on_both_backbones": bool(both_fairness_improved),
        "utility_retained_at_95pct_on_both_backbones": bool(both_utility_feasible),
        "formal_backbone_by_treatment_interaction_test": False,
        "pooled_causal_effect_estimated": False,
        "interpretation": (
            "Descriptive cross-backbone generalization synthesis. "
            "The fairness intervention is supported across backbones only if "
            "fairness improves and utility retention is at least 95% on both. "
            "Directional agreement alone is insufficient."
        ),
    }
    with (args.out / "validation_checks.json").open("w", encoding="utf-8") as stream:
        json.dump(checks, stream, indent=2)

    report = [
        "# Experiment 22.17 — Cross-Backbone Generalization",
        "",
        "## Scope",
        "",
        "This synthesis reuses the frozen validation outputs from Experiments "
        "22.00 (KGSF) and 22.09 (KBRD). It performs no training, retrieval, "
        "reranking, coefficient search, or test-set evaluation.",
        "",
        "## Within-backbone comparisons",
        "",
        dataframe_to_markdown(result_df),
        "",
        "## Decision checks",
        "",
        f"- Fairness direction agrees across backbones: {direction_agreement}",
        f"- Fairness improves on both backbones: {both_fairness_improved}",
        f"- At least 95% NDCG retention on both: {both_utility_feasible}",
        "",
        "## Interpretation boundary",
        "",
        "This is descriptive evidence across two backbones with separately "
        "selected validation policies. It is not a pooled causal estimate, "
        "a formal backbone-by-treatment interaction test, or proof of "
        "generalization to all conversational recommender architectures.",
        "",
        "Experiment 22.16 is not used as positive evidence: its selected "
        "lambda was zero for all calibration variants.",
        "",
    ]
    (args.out / "cross_backbone_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    manifest = {
        **checks,
        "inputs": {
            "kgsf_comparison": str(args.kgsf_dir / "relevance_fairness_comparison.csv"),
            "kgsf_validation_checks": str(args.kgsf_dir / "validation_checks.json"),
            "kgsf_validation_summary": str(args.kgsf_dir / "validation_summary.json"),
            "kbrd_point_estimates": str(args.kbrd_dir / "point_estimates.csv"),
            "kbrd_fairness": str(args.kbrd_dir / "category_fairness_results.csv"),
            "kbrd_validation_checks": str(args.kbrd_dir / "validation_checks.json"),
        },
        "outputs": [
            "cross_backbone_comparison.csv",
            "cross_backbone_report.md",
            "validation_checks.json",
            "experiment_manifest.json",
        ],
        "backbone_comparisons": rows,
    }
    with (args.out / "experiment_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)

    print("\n========== EXPERIMENT 22.17 COMPLETE ==========")
    print(result_df.to_string(index=False))
    print("\nDecision checks:")
    print(json.dumps(checks, indent=2))
    print("\nOutput:", args.out.resolve())


if __name__ == "__main__":
    main()
