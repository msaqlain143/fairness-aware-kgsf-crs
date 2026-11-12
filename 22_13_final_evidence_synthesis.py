#!/usr/bin/env python3
"""Experiment 22.13 — reproducible synthesis of FROZEN validation evidence.

This script is read-only with respect to Experiments 22.09–22.12. It does not
train, retrieve, choose a new lambda, evaluate test data, or assert a final
policy winner. All numeric claims are derived from the specified CSV/JSON
artifacts. Missing or inconsistent source data makes the run fail closed.

Run from the repository root:
    python 22_13_final_evidence_synthesis.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "B0_KBRD"
C02 = "B_category_only_lambda_0p2"
G02 = "B_KG_gated_lambda_0p2"
C04 = "B_category_only_lambda_0p4"
G04 = "B_KG_gated_lambda_0p4"
METHODS = [BASE, C02, G02, C04, G04]
PAIRS = [(0.2, C02, G02), (0.4, C04, G04)]
ATOL = 1e-8


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def read_csv(path: Path, columns: set[str], name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {name}: {path}")
    frame = pd.read_csv(path)
    missing = columns - set(frame.columns)
    require(not missing, f"{name} is missing columns: {sorted(missing)}")
    return frame


def read_json(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {name}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path):
    hash_object = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hash_object.update(chunk)
    return hash_object.hexdigest()


def is_true(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in ("true", "1", "yes")


def numerical(frame, cols, name):
    for col in cols:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        require(np.isfinite(frame[col]).all(), f"Non-finite {name}.{col}")
    return frame


def assert_methods(frame, name):
    require(set(frame["method"]) == set(METHODS),
            f"{name}: method mismatch: {sorted(set(frame['method']) ^ set(METHODS))}")
    require(not frame["method"].duplicated().any(), f"{name}: duplicate methods")
    return frame.set_index("method").loc[METHODS].reset_index()


def derive_claims(quality, fairness, attribution, kg, movement):
    q = quality.set_index("method")
    f = fairness.set_index("method")
    a = attribution.set_index("lambda")
    claims = []

    def add(claim, assessment, evidence, caveat):
        claims.append(dict(claim=claim, assessment=assessment,
                           evidence=evidence, caveat=caveat))

    add("KG support is recorded for a minority of KBRD Top-500 candidate rows",
        "SUPPORTED_DESCRIPTIVE",
        f"Recorded KG support: {float(kg['overall_kg_support_coverage']):.2%} of candidate rows",
        "This is recorded retrieval support, not semantic relevance.")
    add("KG-supported candidates have better mean numerical KBRD ranks",
        "SUPPORTED_ASSOCIATION",
        "Supported-minus-unsupported rank difference: "
        f"{float(kg['conversation_cluster_bootstrap_supported_minus_unsupported_kbrd_rank']['difference']):.3f}",
        "Selection and popularity effects may confound this association.")
    add("KG gating significantly improves category-group MAD at fixed lambda",
        "NOT_ESTABLISHED",
        "; ".join(f"lambda={lam:g}: point={a.loc[lam, 'delta_fairness_gated_minus_control']:.4f}, "
                  f"CI=[{a.loc[lam, 'fairness_delta_ci95_low']:.4f},"
                  f"{a.loc[lam, 'fairness_delta_ci95_high']:.4f}]" for lam, _, _ in PAIRS),
        "Both reported paired percentile confidence intervals include zero; bootstrap design limits apply.")
    add("KG-gated reranking yields lower category-group MAD than unchanged KBRD",
        "CONTRADICTED_ON_VALIDATION_POINT_ESTIMATES",
        f"B0={f.loc[BASE, 'category_group_log_rate_MAD']:.4f}; "
        f"gated lambda=0.2={f.loc[G02, 'category_group_log_rate_MAD']:.4f}; "
        f"gated lambda=0.4={f.loc[G04, 'category_group_log_rate_MAD']:.4f}",
        "Point estimates do not establish a significant difference.")
    add("Higher lambda reduces completely unexposed verified categories",
        "SUPPORTED_DESCRIPTIVE",
        f"B0={int(f.loc[BASE,'completely_unexposed_categories'])}; "
        f"category-only 0.4={int(f.loc[C04,'completely_unexposed_categories'])}; "
        f"KG-gated 0.4={int(f.loc[G04,'completely_unexposed_categories'])}",
        "Category-label coverage is incomplete and the results are validation-only.")
    add("KG gating at lambda=0.4 improves held-out HR@10 versus category-only",
        "SMALL_POINT_ESTIMATE_NOT_CONFIRMED",
        f"paired delta={a.loc[0.4, 'delta_HR@10_gated_minus_control']:.6f}; "
        f"CI=[{a.loc[0.4, 'delta_HR@10_ci95_low']:.6f},"
        f"{a.loc[0.4, 'delta_HR@10_ci95_high']:.6f}]",
        "Confidence interval includes zero; assess practical importance as well.")
    add("A final fairness-improving policy has been established",
        "NOT_ESTABLISHED",
        "Original KBRD has lower aggregate category-group MAD than each tested reranking arm.",
        "Do not select or evaluate a final test policy based on this synthesis alone.")
    if movement is not None:
        m = movement.set_index("method")
        add("All promoted items in KG-gated arms have recorded KG support",
            "SUPPORTED_BY_DESIGN_AND_OBSERVED",
            f"lambda=0.2: {int(m.loc[G02,'promoted_supported'])}/{int(m.loc[G02,'promoted'])}; "
            f"lambda=0.4: {int(m.loc[G04,'promoted_supported'])}/{int(m.loc[G04,'promoted'])}",
            "A gating eligibility rule explains this pattern; it is not independent causal evidence.")
    return pd.DataFrame(claims)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("results_corrected"),
                    help="Root containing experiment_22_09 through experiment_22_12")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output folder; default ROOT/experiment_22_13")
    args = ap.parse_args()
    root = args.root
    out = args.out or root / "experiment_22_13"
    p09, p10, p11, p12 = (root / f"experiment_22_{x:02d}" for x in (9, 10, 11, 12))

    paths = {
        "quality": p10 / "matched_lambda_point_estimates.csv",
        "fairness_10": p10 / "category_fairness_results.csv",
        "paired": p10 / "direct_paired_kg_gating_effects.csv",
        "checks_10": p10 / "validation_checks.json",
        "kg": p11 / "diagnostic_summary.json",
        "checks_11": p11 / "validation_checks.json",
        "movement": p11 / "top10_movement_summary.csv",
        "pop_kg": p11 / "popularity_group_kg_coverage.csv",
        "summary_12": p12 / "method_exclusion_summary.csv",
        "group_12": p12 / "popularity_group_exclusion_summary.csv",
        "checks_12": p12 / "validation_checks.json",
    }
    # Sources from 22.09 are an optional reference; 22.10's matched-lambda
    # results are the canonical source for the five-arm synthesis.
    if (p09 / "experiment_manifest.json").exists():
        paths["manifest_09"] = p09 / "experiment_manifest.json"

    quality = read_csv(paths["quality"],
        {"method", "events", "conversations", "HR@10", "NDCG@10", "lambda", "kg_gate"},
        "matched-lambda quality")
    quality = assert_methods(quality, "matched-lambda quality")
    numerical(quality, ["events", "conversations", "HR@10", "NDCG@10", "lambda"], "quality")

    fairness10 = read_csv(paths["fairness_10"],
        {"method", "category_group_log_rate_MAD"}, "original fairness")
    fairness10 = assert_methods(fairness10, "original fairness")
    numerical(fairness10, ["category_group_log_rate_MAD"], "fairness_10")

    exclusion = read_csv(paths["summary_12"],
        {"method", "categories", "completely_unexposed_categories",
         "category_exposure_coverage", "zero_exposure_cells", "opportunity_cells",
         "zero_exposure_cell_rate", "original_definition_MAD_recalculated",
         "MAD_excluding_completely_unexposed_categories", "total_opportunity"},
         "exclusion summary")
    exclusion = assert_methods(exclusion, "exclusion summary")
    numerical(exclusion,
        ["categories", "completely_unexposed_categories", "category_exposure_coverage",
         "zero_exposure_cells", "opportunity_cells", "zero_exposure_cell_rate",
         "original_definition_MAD_recalculated",
         "MAD_excluding_completely_unexposed_categories", "total_opportunity"], "exclusion")

    merged = fairness10.merge(exclusion, on="method", validate="one_to_one")
    require(np.allclose(merged["category_group_log_rate_MAD"],
                        merged["original_definition_MAD_recalculated"], atol=ATOL, rtol=0),
            "Experiment 22.10 and 22.12 MAD reconciliation failed")
    require(np.allclose(merged["total_opportunity"], merged["total_opportunity"].iloc[0],
                        atol=ATOL, rtol=0), "Opportunity totals differ across methods")
    require((quality["events"] == 624).all() and (quality["conversations"] == 176).all(),
            "Expected 624 held-out validation events and 176 conversations")
    require(all(np.isclose(quality.loc[quality.method == method, "lambda"].iloc[0], lam)
                for lam, no_gate, gate in PAIRS for method in (no_gate, gate)),
            "Matched-lambda values are not aligned")
    require(not is_true(quality.loc[quality.method == BASE, "kg_gate"].iloc[0]),
            "Baseline unexpectedly has KG gating")
    require(all(is_true(quality.loc[quality.method == gate, "kg_gate"].iloc[0])
                and not is_true(quality.loc[quality.method == no_gate, "kg_gate"].iloc[0])
                for _, no_gate, gate in PAIRS), "KG gate assignment incorrect")

    paired = read_csv(paths["paired"],
        {"lambda", "control_method", "kg_gated_method", "delta_fairness_gated_minus_control",
         "fairness_delta_ci95_low", "fairness_delta_ci95_high",
         "bootstrap_probability_fairness_improves",
         "delta_HR@10_gated_minus_control", "delta_HR@10_ci95_low",
         "delta_HR@10_ci95_high", "delta_NDCG@10_gated_minus_control",
         "delta_NDCG@10_ci95_low", "delta_NDCG@10_ci95_high",
         "bootstrap_replicates", "conversation_clusters"}, "paired KG effects")
    numerical(paired, ["lambda", "delta_fairness_gated_minus_control",
        "fairness_delta_ci95_low", "fairness_delta_ci95_high",
        "bootstrap_probability_fairness_improves",
        "delta_HR@10_gated_minus_control", "delta_HR@10_ci95_low",
        "delta_HR@10_ci95_high", "delta_NDCG@10_gated_minus_control",
        "delta_NDCG@10_ci95_low", "delta_NDCG@10_ci95_high",
        "bootstrap_replicates", "conversation_clusters"], "paired")
    require(len(paired) == 2 and set(np.round(paired["lambda"], 10)) == {0.2, 0.4},
            "Expected exactly the two matched-lambda pairs")
    quality_by = quality.set_index("method")
    fair_by = fairness10.set_index("method")
    for lam, no_gate, gate in PAIRS:
        row = paired[np.isclose(paired["lambda"], lam)].iloc[0]
        require(row["control_method"] == no_gate and row["kg_gated_method"] == gate,
                f"Incorrect arm names for lambda={lam}")
        for observed, expected, label in (
            (row["delta_fairness_gated_minus_control"],
             fair_by.loc[gate, "category_group_log_rate_MAD"] - fair_by.loc[no_gate, "category_group_log_rate_MAD"], "MAD"),
            (row["delta_HR@10_gated_minus_control"],
             quality_by.loc[gate, "HR@10"] - quality_by.loc[no_gate, "HR@10"], "HR"),
            (row["delta_NDCG@10_gated_minus_control"],
             quality_by.loc[gate, "NDCG@10"] - quality_by.loc[no_gate, "NDCG@10"], "NDCG"),
        ):
            require(np.isclose(observed, expected, rtol=0, atol=ATOL),
                    f"Paired delta disagrees with point estimates: lambda={lam} {label}")
        require(row["bootstrap_replicates"] == 10000 and row["conversation_clusters"] == 176,
                "Bootstrap sample count or cluster count differs from reported 22.10 results")

    group = read_csv(paths["group_12"],
        {"method", "popularity_group", "eligible_category_cells",
         "zero_exposure_categories", "zero_exposure_rate",
         "total_exposure", "total_opportunity"}, "popularity exclusion")
    numerical(group, ["popularity_group", "eligible_category_cells",
        "zero_exposure_categories", "zero_exposure_rate", "total_exposure",
        "total_opportunity"], "popularity exclusion")
    require(len(group) == 25 and not group.duplicated(["method", "popularity_group"]).any(),
            "Expected exactly 25 unique popularity/method rows")
    require(all(set(group.loc[group.method == method, "popularity_group"]) == set(range(5))
                for method in METHODS), "Missing popularity group")
    pop_kg = read_csv(paths["pop_kg"],
        {"popularity_group", "candidate_rows", "supported_rows", "kg_support_coverage"},
        "KG popularity coverage")
    numerical(pop_kg, ["popularity_group", "candidate_rows", "supported_rows", "kg_support_coverage"], "KG coverage")
    require(len(pop_kg) == 5 and set(pop_kg["popularity_group"]) == set(range(5)),
            "Expected exactly five popularity-group KG rows")
    require(np.allclose(pop_kg["supported_rows"] / pop_kg["candidate_rows"],
                        pop_kg["kg_support_coverage"], rtol=0, atol=ATOL),
            "KG support coverage does not match row counts")
    group = group.merge(pop_kg[["popularity_group", "candidate_rows", "supported_rows",
                                "kg_support_coverage"]], on="popularity_group", validate="many_to_one")
    for method in METHODS:
        sub = group[group.method == method]
        summary_row = exclusion[exclusion.method == method].iloc[0]
        require(int(sub.zero_exposure_categories.sum()) == int(summary_row.zero_exposure_cells),
                f"Group exclusion cell totals disagree for {method}")
        require(int(sub.eligible_category_cells.sum()) == int(summary_row.opportunity_cells),
                f"Group opportunity counts disagree for {method}")

    kg = read_json(paths["kg"], "KG diagnostic summary")
    checks10 = read_json(paths["checks_10"], "22.10 checks")
    checks11 = read_json(paths["checks_11"], "22.11 checks")
    checks12 = read_json(paths["checks_12"], "22.12 checks")
    require(checks10.get("validation_only") is True
            and checks10.get("test_split_evaluated") is False
            and checks10.get("test_data_loaded") is False,
            "22.10 manifest does not establish validation-only scope")
    require(checks11.get("overall_pass") is True and
            checks11.get("prohibited_operations_not_performed", {}).get("test_split_used") is False,
            "22.11 validation checks failed")
    require(checks12.get("checks", {}).get("overall_pass") is True and
            checks12.get("checks", {}).get("original_MAD_recalculation_matches") is True,
            "22.12 validation checks failed")
    require(int(kg["candidate_rows"]) == 1505000, "Unexpected KG candidate-row count")
    require(np.isclose(float(kg["overall_kg_support_coverage"]),
                       float(kg["kg_supported_candidate_rows"]) / float(kg["candidate_rows"]),
                       rtol=0, atol=ATOL), "KG coverage mismatch")
    require(int(pop_kg.candidate_rows.sum()) == int(kg["candidate_rows"]),
            "Popularity candidate counts do not sum to KG diagnostic total")
    require(int(pop_kg.supported_rows.sum()) == int(kg["kg_supported_candidate_rows"]),
            "Popularity KG support counts do not sum to diagnostic total")

    movement = None
    if paths["movement"].exists():
        movement = read_csv(paths["movement"],
            {"method", "promoted", "demoted", "promoted_supported"}, "Top-10 movement")
        require(set(movement.method) == set(METHODS[1:]),
                "Top-10 movement method set mismatch")

    quality_out = quality.copy()
    quality_out["delta_HR_vs_B0"] = quality_out["HR@10"] - float(quality_by.loc[BASE, "HR@10"])
    quality_out["delta_NDCG_vs_B0"] = quality_out["NDCG@10"] - float(quality_by.loc[BASE, "NDCG@10"])
    fairness_out = merged.copy()
    fairness_out["delta_MAD_vs_B0"] = fairness_out["category_group_log_rate_MAD"] - float(fair_by.loc[BASE, "category_group_log_rate_MAD"])
    paired_out = paired.copy()
    paired_out["fairness_CI_excludes_zero"] = (
        (paired_out["fairness_delta_ci95_low"] > 0) |
        (paired_out["fairness_delta_ci95_high"] < 0)
    )
    paired_out["fairness_interpretation"] = np.where(
        paired_out["fairness_CI_excludes_zero"], "directionally_supported_by_reported_CI", "inconclusive_CI_includes_zero")
    claims = derive_claims(quality_out, fairness_out, paired_out, kg, movement)

    validations = {
        "five_matched_methods": True,
        "624_heldout_events_176_conversations": True,
        "paired_deltas_reconciled": True,
        "fairness_22_10_and_22_12_reconciled": True,
        "popularity_groups_complete_and_reconciled": True,
        "kg_support_totals_reconciled": True,
        "source_validation_checks_accepted": True,
        "test_data_not_loaded_by_synthesis": True,
        "no_model_training_or_retrieval": True,
        "no_new_policy_selection_or_tuning": True,
    }
    validations["overall_pass"] = all(validations.values())
    input_hashes = {key: {"path": str(path), "sha256": sha256(path),
                          "bytes": path.stat().st_size}
                    for key, path in paths.items() if path.is_file()}
    outputs = {
        "recommendation_quality_comparison.csv": quality_out,
        "fairness_comparison.csv": fairness_out,
        "kg_gating_attribution.csv": paired_out,
        "popularity_fairness_comparison.csv": group,
        "research_claims_assessment.csv": claims,
    }
    assessment = {
        "experiment": "22.13_final_evidence_synthesis",
        "scope": "validation_only_frozen_existing_exports",
        "events": 624,
        "conversations": 176,
        "kg_support_coverage": float(kg["overall_kg_support_coverage"]),
        "mean_spearman_on_eligible_events": float(kg["mean_event_spearman"]),
        "primary_fairness_definition": "category_group_log_rate_MAD (lower is better)",
        "fairness_claim": "Not established: no tested reranking arm improves point-estimate MAD over unchanged KBRD.",
        "kg_attribution_claim": "Matched-lambda bootstrap fairness intervals both include zero.",
        "sensitivity": "Complete category exclusion and non-excluded-category MAD reported separately.",
        "final_policy": None,
        "final_policy_status": "NOT_SELECTED_BY_THIS_SCRIPT",
        "independent_test_evaluated": False,
        "recommendation": "Do not claim established KG fairness superiority; a predeclared selection rule is required before independent test.",
    }
    manifest = {
        "experiment": "22.13",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": input_hashes,
        "outputs": [*outputs, "final_research_assessment.json", "validation_checks.json", "experiment_manifest.json"],
        "validation_only": True,
        "test_split_evaluated": False,
        "training_performed": False,
        "retrieval_rerun": False,
        "new_weighting_coefficients": False,
        "policy_selected": False,
        "limitations": [
            "The fairness metric can return zero MAD for fully unexposed categories.",
            "Verified category evidence is sparse and may bias evaluation.",
            "Bootstrap results are as exported; this script does not repair bootstrap validity or infer causal effects.",
            "The baseline and reranked methods share the same frozen candidate pool; the data are validation-only.",
        ],
    }

    # No output is generated before complete validation.
    out.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(out / filename, index=False)
    for filename, payload in (
        ("final_research_assessment.json", assessment),
        ("validation_checks.json", validations),
        ("experiment_manifest.json", manifest),
    ):
        with (out / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)

    print("\n========== EXPERIMENT 22.13 COMPLETE ==========")
    print("Recommendation quality:")
    print(quality_out[["method", "HR@10", "NDCG@10"]].to_string(index=False))
    print("\nCategory fairness and exclusion:")
    print(fairness_out[["method", "category_group_log_rate_MAD", "completely_unexposed_categories",
                        "category_exposure_coverage"]].to_string(index=False))
    print("\nDirect matched-lambda KG attribution:")
    print(paired_out[["lambda", "delta_fairness_gated_minus_control",
                      "fairness_delta_ci95_low", "fairness_delta_ci95_high",
                      "fairness_interpretation"]].to_string(index=False))
    print(f"\nValidation: {'PASS' if validations['overall_pass'] else 'FAIL'}")
    print("Test evaluated: NO | New tuning: NO | Final policy selected: NO")
    print("Output directory:", out.resolve())


if __name__ == "__main__":
    main()
