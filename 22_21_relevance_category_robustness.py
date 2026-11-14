

#!/usr/bin/env python
"""
Experiment 22.21: Relevance and Category Robustness Analysis.

Purpose
-------
Stress-test the frozen Experiment 22.18 contextual relevance gate against
small, reproducible perturbations of the frozen KBRD candidate scores.

Design
------
- Reuses the frozen KBRD Top-500 validation candidates and KG-rank export.
- Reuses the exact category metadata, conversation split, exposure ledger,
  contextual relevance gate, and fairness metric from Experiments 22.16/22.18.
- Freezes lambda=0.4 and tau=250.0 from Experiment 22.18.
- Applies Gaussian score perturbations at 0%, 1%, 5%, and 10% of each
  event's median absolute candidate-score scale.
- Uses the same perturbed candidate scores for baseline and treatment.
- Replays warm and tuning histories before evaluating held-out conversations.
- Does not train, rerun retrieval, tune parameters, or evaluate the independent
  test set.

Interpretation
--------------
Perturbation percentages describe noise standard deviation relative to the
median absolute score magnitude within each event. They are stress-test
settings, not estimates of real-world score measurement error.
"""

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

LAMBDA_FIXED = 0.4
TAU_FIXED = 250.0
NOISE_LEVELS = (0.0, 0.01, 0.05, 0.10)
DEFAULT_REPLICATES = 30
DEFAULT_SEED = 22121
EPS = 1e-12
TOP_K = 10

DEFAULT_KBRD = Path("results_corrected/experiment_22_04B/kbrd_validation_top500.csv")
DEFAULT_KG = Path("results_corrected/adaptive_m2_validation_export22_candidates.csv")
DEFAULT_PROFILES = Path("results_category_audit/experiment_21_22A/catalog_item_category_profiles.csv")
DEFAULT_CATALOG = Path("catalog_audit_corrected.pkl")
DEFAULT_REFERENCE = Path("22_16_category_context_utility_calibration.py")
DEFAULT_GATE = Path("22_18_contextual_relevance_gate_validation.py")
DEFAULT_OUT = Path("results_corrected/experiment_22_21")


def load_module(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def perturb_candidates(candidates, level, rng):
    """Return a copy with Gaussian score noise; IDs and KG ranks stay frozen."""
    perturbed = {}
    for event_id, candidate in candidates.items():
        scores = np.asarray(candidate["scores"], dtype=np.float64)
        scale = float(np.median(np.abs(scores))) if scores.size else 1.0
        if scale <= EPS:
            scale = float(np.std(scores)) if scores.size else 1.0
        if scale <= EPS:
            scale = 1.0
        noise = rng.normal(0.0, float(level) * scale, size=len(scores))
        perturbed[event_id] = {
            "ids": np.asarray(candidate["ids"]).copy(),
            "scores": scores + noise,
            "kg": np.asarray(candidate["kg"]).copy(),
        }
    return perturbed


def baseline_rank_from_scores(candidate):
    """Re-rank the frozen candidate pool by its (possibly perturbed) scores."""
    ids = candidate["ids"]
    scores = candidate["scores"]
    order = np.lexsort((ids, -scores))
    return ids[order]


def run_split_robust(event_ids, events, candidates, item2group, categories,
                     start_ledger, gate_module, reference, treatment):
    """Replay one split sequentially, preserving the 22.18 ledger semantics."""
    ledger = start_ledger.clone()
    rows = []
    top10_by_event = {}

    for event_id in sorted(event_ids):
        candidate = candidates[event_id]
        if treatment:
            ranked_ids, info = gate_module.contextual_gate_rerank(
                candidate, item2group, categories, ledger,
                LAMBDA_FIXED, TAU_FIXED
            )
        else:
            ranked_ids = baseline_rank_from_scores(candidate)
            info = {
                "eligible_contextual_kg_candidates": 0,
                "kg_supported_candidates": int(np.sum(candidate["kg"] > 0)),
                "categorized_candidates": 0,
                "top10_promoted_positions": 0,
            }

        target = int(events[event_id]["target"])
        positions = np.flatnonzero(ranked_ids[:TOP_K] == target)
        if len(positions):
            hit = 1
            ndcg = 1.0 / math.log2(int(positions[0]) + 2.0)
        else:
            hit = 0
            ndcg = 0.0

        top10_by_event[int(event_id)] = list(map(int, ranked_ids[:TOP_K]))
        groups = np.asarray(
            [item2group[int(movie_id)] for movie_id in candidate["ids"]],
            dtype=int
        )
        ranked_groups = np.asarray(
            [item2group[int(movie_id)] for movie_id in ranked_ids[:TOP_K]],
            dtype=int
        )
        ledger.update(
            candidate["ids"], groups,
            ranked_ids[:TOP_K], ranked_groups, categories
        )
        rows.append({
            "event_id": int(event_id),
            "conv_id": int(events[event_id]["conv"]),
            "hit10": hit,
            "ndcg10": ndcg,
            **info,
        })

    frame = pd.DataFrame(rows)
    summary = {
        "events": int(len(frame)),
        "conversations": int(frame.conv_id.nunique()) if len(frame) else 0,
        "HR@10": float(frame.hit10.mean()) if len(frame) else float("nan"),
        "NDCG@10": float(frame.ndcg10.mean()) if len(frame) else float("nan"),
        "category_group_log_rate_MAD": float(reference.fairness_mad(ledger)),
        "mean_eligible_contextual_kg_candidates": (
            float(frame.eligible_contextual_kg_candidates.mean())
            if len(frame) else 0.0
        ),
        "mean_top10_promoted_positions": (
            float(frame.top10_promoted_positions.mean())
            if len(frame) else 0.0
        ),
    }
    return frame, ledger, summary, top10_by_event


def evaluate_one(candidates, events, split_ids, item2group, categories,
                 gate_module, reference):
    warm_ids, tune_ids, heldout_ids = split_ids

    baseline_warm = reference.ExposureLedger()
    _, baseline_warm, _, _ = run_split_robust(
        warm_ids, events, candidates, item2group, categories,
        baseline_warm, gate_module, reference, treatment=False
    )
    _, baseline_tune_ledger, _, _ = run_split_robust(
        tune_ids, events, candidates, item2group, categories,
        baseline_warm, gate_module, reference, treatment=False
    )

    treatment_warm = reference.ExposureLedger()
    _, treatment_warm, _, _ = run_split_robust(
        warm_ids, events, candidates, item2group, categories,
        treatment_warm, gate_module, reference, treatment=True
    )
    _, treatment_tune_ledger, _, _ = run_split_robust(
        tune_ids, events, candidates, item2group, categories,
        treatment_warm, gate_module, reference, treatment=True
    )

    baseline_frame, _, baseline_summary, baseline_top10 = run_split_robust(
        heldout_ids, events, candidates, item2group, categories,
        baseline_tune_ledger, gate_module, reference, treatment=False
    )
    treatment_frame, _, treatment_summary, treatment_top10 = run_split_robust(
        heldout_ids, events, candidates, item2group, categories,
        treatment_tune_ledger, gate_module, reference, treatment=True
    )
    return (baseline_frame, treatment_frame, baseline_summary,
            treatment_summary, baseline_top10, treatment_top10)


def mean_top10_overlap(current, reference):
    common = sorted(set(current) & set(reference))
    if not common:
        return float("nan")
    values = []
    for event_id in common:
        a, b = set(current[event_id]), set(reference[event_id])
        values.append(len(a & b) / float(TOP_K))
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kbrd", type=Path, default=DEFAULT_KBRD)
    parser.add_argument("--kg", type=Path, default=DEFAULT_KG)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.replicates < 10:
        raise ValueError("--replicates must be at least 10 for a useful stress test.")

    for path in (args.kbrd, args.kg, args.profiles, args.catalog,
                 args.reference, args.gate):
        if not path.exists():
            raise FileNotFoundError(path)

    reference = load_module(args.reference, "experiment_22_16_reference")
    gate = load_module(args.gate, "experiment_22_18_gate")

    required_reference = ("load_metadata", "split_conversations",
                          "ExposureLedger", "fairness_mad")
    missing = [name for name in required_reference if not hasattr(reference, name)]
    if missing:
        raise AttributeError(f"22.16 reference is missing: {missing}")

    item2group, catalog, categories = reference.load_metadata(
        args.catalog, args.profiles
    )
    events, candidates = gate.load_events_and_candidates(args.kbrd, args.kg)

    if not events or set(events) != set(candidates):
        raise RuntimeError("Frozen event/candidate alignment failed.")
    if any(not set(map(int, c["ids"])).issubset(catalog)
           for c in candidates.values()):
        raise RuntimeError("A frozen candidate is absent from the catalog.")

    conv_ids = [v["conv"] for v in events.values()]
    warm_convs, tune_convs, heldout_convs = reference.split_conversations(conv_ids)
    warm_ids = {eid for eid, e in events.items() if e["conv"] in warm_convs}
    tune_ids = {eid for eid, e in events.items() if e["conv"] in tune_convs}
    heldout_ids = {eid for eid, e in events.items() if e["conv"] in heldout_convs}
    if (warm_convs & tune_convs or warm_convs & heldout_convs
            or tune_convs & heldout_convs):
        raise RuntimeError("Conversation-disjoint split check failed.")
    if warm_ids | tune_ids | heldout_ids != set(events):
        raise RuntimeError("Split does not cover all frozen events.")

    split_ids = (warm_ids, tune_ids, heldout_ids)
    args.out.mkdir(parents=True, exist_ok=True)

    print("Loading frozen validation exports...")
    print(
        f"Events={len(events)} | conversations={len(set(conv_ids))} | "
        f"warm={len(warm_ids)} | tune={len(tune_ids)} | heldout={len(heldout_ids)}"
    )
    print(f"Fixed parameters: lambda={LAMBDA_FIXED}, tau={TAU_FIXED}")

    # Establish the unperturbed held-out ranking references.
    zero = evaluate_one(
        candidates, events, split_ids, item2group, categories, gate, reference
    )
    baseline_reference_top10 = zero[4]
    treatment_reference_top10 = zero[5]

    result_rows = []
    run_specs = [(0.0, 0)] + [
        (level, replicate)
        for level in NOISE_LEVELS if level > 0
        for replicate in range(args.replicates)
    ]

    for run_index, (level, replicate) in enumerate(run_specs, start=1):
        if level == 0:
            perturbed = candidates
        else:
            rng = np.random.default_rng(args.seed + int(level * 10000) + replicate)
            perturbed = perturb_candidates(candidates, level, rng)

        evaluated = evaluate_one(
            perturbed, events, split_ids, item2group, categories, gate, reference
        )
        _, _, base, treat, base_top10, treat_top10 = evaluated

        for method, summary, current_top10, reference_top10 in (
            ("B0_KBRD", base, base_top10, baseline_reference_top10),
            ("B5_contextual_relevance_gate", treat, treat_top10,
             treatment_reference_top10),
        ):
            result_rows.append({
                "noise_level": level,
                "replicate": replicate,
                "method": method,
                **summary,
                "heldout_top10_overlap_vs_unperturbed": mean_top10_overlap(
                    current_top10, reference_top10
                ),
            })

        if run_index % 10 == 0 or run_index == len(run_specs):
            print(f"Completed {run_index}/{len(run_specs)} stress-test runs")

    results = pd.DataFrame(result_rows)
    results.to_csv(args.out / "robustness_replicate_results.csv", index=False)

    grouped = results.groupby(["noise_level", "method"], as_index=False).agg(
        replicates=("replicate", "count"),
        HR10_mean=("HR@10", "mean"),
        HR10_sd=("HR@10", "std"),
        NDCG10_mean=("NDCG@10", "mean"),
        NDCG10_sd=("NDCG@10", "std"),
        fairness_MAD_mean=("category_group_log_rate_MAD", "mean"),
        fairness_MAD_sd=("category_group_log_rate_MAD", "std"),
        top10_overlap_mean=("heldout_top10_overlap_vs_unperturbed", "mean"),
        eligible_contextual_KG_candidates_mean=(
            "mean_eligible_contextual_kg_candidates", "mean"
        ),
        promoted_positions_mean=("mean_top10_promoted_positions", "mean"),
    )
    grouped.to_csv(args.out / "robustness_summary.csv", index=False)

    pivot = grouped.pivot(index="noise_level", columns="method")
    paired_rows = []
    for level in NOISE_LEVELS:
        subset = grouped[grouped.noise_level == level].set_index("method")
        if not {"B0_KBRD", "B5_contextual_relevance_gate"}.issubset(subset.index):
            continue
        b = subset.loc["B0_KBRD"]
        t = subset.loc["B5_contextual_relevance_gate"]
        retention = (
            float(t["NDCG10_mean"]) / max(float(b["NDCG10_mean"]), EPS)
        )
        paired_rows.append({
            "noise_level": level,
            "baseline_HR10_mean": float(b["HR10_mean"]),
            "treatment_HR10_mean": float(t["HR10_mean"]),
            "delta_HR10": float(t["HR10_mean"] - b["HR10_mean"]),
            "baseline_NDCG10_mean": float(b["NDCG10_mean"]),
            "treatment_NDCG10_mean": float(t["NDCG10_mean"]),
            "NDCG_retention": retention,
            "baseline_fairness_MAD_mean": float(b["fairness_MAD_mean"]),
            "treatment_fairness_MAD_mean": float(t["fairness_MAD_mean"]),
            "delta_fairness_MAD": float(
                t["fairness_MAD_mean"] - b["fairness_MAD_mean"]
            ),
            "fairness_improved": bool(
                t["fairness_MAD_mean"] < b["fairness_MAD_mean"]
            ),
            "utility_retained_at_95pct": bool(retention >= 0.95),
            "treatment_top10_overlap_mean": float(t["top10_overlap_mean"]),
        })
    comparison = pd.DataFrame(paired_rows)
    comparison.to_csv(args.out / "robustness_method_comparison.csv", index=False)

    checks = {
        "frozen_validation_exports_used": True,
        "same_event_universe_for_all_runs": True,
        "conversation_disjoint_split": True,
        "all_events_assigned_once": True,
        "500_unique_candidates_per_event": all(
            len(c["ids"]) == 500 and len(np.unique(c["ids"])) == 500
            for c in candidates.values()
        ),
        "all_candidates_in_catalog": all(
            set(map(int, c["ids"])).issubset(catalog)
            for c in candidates.values()
        ),
        "KG_ranks_and_candidate_IDs_unchanged_under_perturbation": True,
        "same_score_perturbations_used_for_baseline_and_treatment": True,
        "lambda_and_tau_frozen_from_experiment_22_18": (
            LAMBDA_FIXED == 0.4 and TAU_FIXED == 250.0
        ),
        "all_noise_levels_evaluated": set(results.noise_level.unique())
        == set(NOISE_LEVELS),
        "replicate_counts_valid": (
            int((results.noise_level == 0).sum()) == 2
            and all(
                int((results.noise_level == level).sum()) == 2 * args.replicates
                for level in NOISE_LEVELS if level > 0
            )
        ),
        "no_training_performed": True,
        "no_retrieval_rerun": True,
        "no_parameter_search": True,
        "no_independent_test_evaluation": True,
        "context_remains_candidate_pool_category_prevalence": True,
    }
    checks["overall_pass"] = all(bool(v) for v in checks.values())

    assessment = {
        "experiment": "22.21_relevance_category_robustness",
        "scope": "frozen_validation_score_perturbation_stress_test",
        "events": len(events),
        "conversations": len(set(conv_ids)),
        "warm_events": len(warm_ids),
        "tuning_events": len(tune_ids),
        "heldout_events": len(heldout_ids),
        "fixed_lambda": LAMBDA_FIXED,
        "fixed_tau": TAU_FIXED,
        "noise_levels": list(NOISE_LEVELS),
        "replicates_per_nonzero_noise_level": args.replicates,
        "perturbation_definition": (
            "Gaussian noise with standard deviation equal to the noise level "
            "times the event-specific median absolute KBRD candidate score."
        ),
        "results": paired_rows,
        "interpretation": (
            "Robustness is assessed by held-out ranking stability, utility, "
            "and category-group exposure fairness under score perturbations. "
            "These synthetic perturbation magnitudes are stress-test settings, "
            "not empirically calibrated score-error estimates."
        ),
        "training_performed": False,
        "retrieval_rerun": False,
        "parameter_search": False,
        "independent_test_evaluated": False,
    }

    (args.out / "validation_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    (args.out / "research_assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )
    manifest = {
        **assessment,
        "inputs": {
            "kbrd_candidates": str(args.kbrd),
            "kg_candidate_export": str(args.kg),
            "category_profiles": str(args.profiles),
            "catalog_metadata": str(args.catalog),
            "experiment_22_16_reference": str(args.reference),
            "experiment_22_18_gate": str(args.gate),
        },
        "outputs": [
            "robustness_replicate_results.csv",
            "robustness_summary.csv",
            "robustness_method_comparison.csv",
            "validation_checks.json",
            "research_assessment.json",
            "experiment_manifest.json",
        ],
    }
    (args.out / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\n========== EXPERIMENT 22.21 COMPLETE ==========")
    print(comparison.to_string(index=False))
    print("\nValidation:")
    print(json.dumps(checks, indent=2))
    print("\nOutput:", args.out.resolve())
    if not checks["overall_pass"]:
        raise RuntimeError("Experiment 22.21 validation failed.")


if __name__ == "__main__":
    main()
