#!/usr/bin/env python
"""Experiment 22.20 corrected: paired cluster bootstrap + cluster sign-flip tests.

Input: frozen held-out outputs from Experiment 22.19.
Utility: paired conversation-cluster bootstrap percentile CIs and a
Monte-Carlo cluster-level sign-flip randomization test.
Multiplicity: Holm correction across six utility tests.
Fairness MAD: descriptive only; it is a sequential ledger aggregate and is
not assigned an event-level CI.
No training, retrieval, tuning, or independent-test evaluation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT = Path("results_corrected/experiment_22_19")
DEFAULT_OUT = Path("results_corrected/experiment_22_20")
FULL = "FULL_22_18"
ABLATIONS = [
    "A1_no_category_context",
    "A2_no_KG_evidence",
    "A3_no_exposure_deficit",
]
METRICS = ["hit10", "ndcg10"]


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


def cluster_deltas(frame, treatment, control, metric):
    wide = frame.pivot(
        index=["event_id", "conv_id"], columns="method", values=metric
    ).reset_index()
    wide = wide.dropna(subset=[treatment, control]).copy()
    wide["delta"] = (
        wide[treatment].astype(float) - wide[control].astype(float)
    )
    if wide.empty:
        raise ValueError(f"No paired observations for {treatment}, {control}, {metric}")
    grouped = wide.groupby("conv_id", sort=False)["delta"].agg(["sum", "count"])
    return wide, grouped["sum"].to_numpy(float), grouped["count"].to_numpy(float)


def bootstrap_ci(cluster_sums, cluster_counts, n_boot, rng):
    n = len(cluster_sums)
    draws = rng.integers(0, n, size=(n_boot, n))
    numer = cluster_sums[draws].sum(axis=1)
    denom = cluster_counts[draws].sum(axis=1)
    return numer / denom


def sign_flip_pvalue(cluster_sums, cluster_counts, observed, n_perm, rng):
    """Two-sided cluster sign-flip test on the event-weighted mean delta."""
    n = len(cluster_sums)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    perm_num = (signs * cluster_sums[None, :]).sum(axis=1)
    # Keep the observed cluster event counts fixed in every permutation.
    perm_stat = perm_num / cluster_counts.sum()
    obs = abs(float(observed))
    # Add-one correction prevents a zero Monte-Carlo p-value.
    return float((1 + np.sum(np.abs(perm_stat) >= obs)) / (n_perm + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=22020)
    args = parser.parse_args()

    if args.bootstrap < 1000 or args.permutations < 1000:
        raise ValueError("Bootstrap and permutations must each be >= 1000.")

    event_path = args.input_dir / "heldout_event_results.csv"
    summary_path = args.input_dir / "ablation_comparison.csv"
    checks_path = args.input_dir / "validation_checks.json"
    for path in (event_path, summary_path, checks_path):
        if not path.exists():
            raise FileNotFoundError(path)

    source_checks = json.loads(checks_path.read_text(encoding="utf-8"))
    if not source_checks.get("overall_pass", False):
        raise RuntimeError("Experiment 22.19 source validation did not pass.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(event_path)
    summary = pd.read_csv(summary_path)

    required = {"event_id", "conv_id", "method", "hit10", "ndcg10"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Missing event columns: {sorted(missing)}")
    if events.duplicated(["event_id", "method"]).any():
        raise ValueError("Duplicate event_id/method records.")

    expected = {"B0_KBRD", FULL, *ABLATIONS}
    found = set(events.method.astype(str))
    if not expected.issubset(found):
        raise ValueError(f"Missing methods: {sorted(expected - found)}")

    event_sets = {
        m: set(events.loc[events.method == m, "event_id"]) for m in expected
    }
    same_events = all(event_sets[m] == event_sets[FULL] for m in expected)
    event_mapping_unique = bool(
        (events.groupby("event_id").conv_id.nunique() == 1).all()
    )
    if not same_events or not event_mapping_unique:
        raise ValueError("Held-out event alignment or conversation mapping failed.")

    result_rows = []
    seed_counter = 0
    for treatment in ABLATIONS:
        for metric in METRICS:
            wide, sums, counts = cluster_deltas(
                events, treatment, FULL, metric
            )
            observed = float(wide.delta.mean())
            rng_boot = np.random.default_rng(args.seed + seed_counter)
            rng_perm = np.random.default_rng(args.seed + 1000 + seed_counter)
            boot = bootstrap_ci(sums, counts, args.bootstrap, rng_boot)
            ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
            p_value = sign_flip_pvalue(
                sums, counts, observed, args.permutations, rng_perm
            )
            result_rows.append({
                "treatment": treatment,
                "control": FULL,
                "metric": metric,
                "point_delta_treatment_minus_full": observed,
                "bootstrap_mean_delta": float(boot.mean()),
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "cluster_sign_flip_p_two_sided": p_value,
                "conversation_clusters": int(len(sums)),
                "paired_events": int(len(wide)),
                "bootstrap_replicates": int(args.bootstrap),
                "permutation_replicates": int(args.permutations),
            })
            seed_counter += 1

    inferential = pd.DataFrame(result_rows)
    inferential["holm_adjusted_p"] = holm_adjust(
        inferential.cluster_sign_flip_p_two_sided.to_numpy()
    )
    inferential["significant_after_holm_0p05"] = (
        inferential.holm_adjusted_p < 0.05
    )
    inferential.to_csv(
        args.out_dir / "paired_utility_cluster_inference.csv", index=False
    )

    fairness_metric = "category_group_log_rate_MAD"
    if fairness_metric not in summary.columns:
        raise ValueError(f"Missing {fairness_metric} in ablation_comparison.csv")
    sidx = summary.set_index("method")
    if FULL not in sidx.index:
        raise ValueError(f"Missing {FULL} in ablation summary")
    full_mad = float(sidx.loc[FULL, fairness_metric])
    fairness_rows = []
    for method in ABLATIONS:
        mad = float(sidx.loc[method, fairness_metric])
        fairness_rows.append({
            "treatment": method,
            "control": FULL,
            "full_fairness_MAD": full_mad,
            "ablation_fairness_MAD": mad,
            "delta_MAD_ablation_minus_full": mad - full_mad,
            "confidence_interval": "NOT_ESTIMATED",
            "interpretation": (
                "ablation worse" if mad > full_mad else
                "ablation better" if mad < full_mad else "equal point estimate"
            ),
            "reason": (
                "Sequential aggregate exposure/opportunity ledger; valid "
                "uncertainty requires recomputing the entire ledger per "
                "conversation-cluster resample."
            ),
        })
    fairness = pd.DataFrame(fairness_rows)
    fairness.to_csv(
        args.out_dir / "fairness_ablation_descriptive.csv", index=False
    )

    checks = {
        "experiment_22_19_source_validation_passed": True,
        "required_methods_present": bool(expected.issubset(found)),
        "same_heldout_event_universe": bool(same_events),
        "event_to_conversation_mapping_unique": event_mapping_unique,
        "paired_event_alignment_used": True,
        "conversation_cluster_bootstrap_used": True,
        "cluster_sign_flip_randomization_test_used": True,
        "holm_correction_applied_to_six_utility_tests": len(inferential) == 6,
        "fairness_MAD_kept_descriptive": True,
        "bootstrap_and_permutation_counts_valid": (
            args.bootstrap >= 1000 and args.permutations >= 1000
        ),
        "no_training_performed": True,
        "no_retrieval_rerun": True,
        "no_parameter_search": True,
        "no_independent_test_evaluation": True,
    }
    checks["overall_pass"] = all(checks.values())

    assessment = {
        "experiment": "22.20_paired_ablation_cluster_inference",
        "scope": "frozen_heldout_outputs_from_experiment_22_19",
        "heldout_events": int(event_sets[FULL].__len__()),
        "conversation_clusters": int(
            events.loc[events.method == FULL, "conv_id"].nunique()
        ),
        "bootstrap_replicates": args.bootstrap,
        "permutation_replicates": args.permutations,
        "utility_inference": (
            "Paired conversation-cluster percentile bootstrap CIs plus "
            "two-sided cluster sign-flip randomization tests; Holm correction "
            "across six utility comparisons."
        ),
        "fairness_inference": (
            "Aggregate fairness MAD differences remain descriptive. A valid "
            "fairness CI requires full sequential ledger replay inside each "
            "cluster resample."
        ),
        "independent_test_evaluated": False,
        "training_performed": False,
        "retrieval_rerun": False,
        "parameter_search": False,
    }
    (args.out_dir / "validation_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    (args.out_dir / "final_research_assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )
    manifest = {
        **assessment,
        "inputs": {
            "heldout_event_results": str(event_path),
            "ablation_comparison": str(summary_path),
            "source_validation_checks": str(checks_path),
        },
        "outputs": [
            "paired_utility_cluster_inference.csv",
            "fairness_ablation_descriptive.csv",
            "validation_checks.json",
            "final_research_assessment.json",
            "experiment_manifest.json",
        ],
    }
    (args.out_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\n========== EXPERIMENT 22.20 COMPLETE ==========")
    print("\nPaired utility inference:")
    print(inferential.to_string(index=False))
    print("\nDescriptive fairness:")
    print(fairness.to_string(index=False))
    print("\nValidation:")
    print(json.dumps(checks, indent=2))
    print("\nOutput:", args.out_dir.resolve())
    if not checks["overall_pass"]:
        raise RuntimeError("Experiment 22.20 validation failed.")


if __name__ == "__main__":
    main()



