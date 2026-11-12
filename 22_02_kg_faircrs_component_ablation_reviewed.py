#!/usr/bin/env python
"""
Experiment 22.2: proposal-aligned component ablation for KG-FairCRS.

Uses the frozen Experiment 22.1 inputs and the same deterministic 60/20/20
conversation split. No model inference, retraining, candidate regeneration,
checkpoint modification, or test-set access is performed.

The full KG-supported policy is selected ONLY on the tuning partition from
B1-B5 (never from held-out results). Ablations remove one component at a time:
- no KG contextual support;
- no adaptive category calibration;
- no online exposure-history updates;
- no category calibration (unchanged A1 reference).

All policy parameters are chosen on tuning data under the same 95% NDCG@10
retention constraint, then replayed before held-out evaluation.
Opportunity accounting uses the original candidate order; exposure uses
the final reranked Top-10.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_22_1_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("kgfaircrs_exp22_1", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Experiment 22.1 source: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tune_one_policy(M, data, warm_ledger, tune_events, baseline_tune_ndcg,
                    label, mode, threshold, lambdas, update_history=True):
    rows = []
    for lam in lambdas:
        ledger = M.clone_ledger(warm_ledger)
        result, _ = M.run_policy_22_1(
            data, tune_events, initial_ledger=ledger, lam=lam,
            policy_mode=mode, threshold=threshold,
            update_ledger=update_history
        )
        disparity, _, _, _ = M.measure_partition_fairness(result, tune_events, data)
        ndcg = float(result.ndcg10.mean())
        rows.append({
            "component": label,
            "policy_mode": mode,
            "kg_rank_threshold": threshold,
            "lambda": float(lam),
            "HR@10": float(result.hit10.mean()),
            "NDCG@10": ndcg,
            "retention_vs_A1": ndcg / (baseline_tune_ndcg + M.EPS),
            "category_group_log_rate_MAD": disparity,
            "meets_95pct_utility": bool(
                ndcg >= 0.95 * baseline_tune_ndcg
            ),
            "mean_eligible_categorized_candidates": float(
                result.eligible_categorized_candidates.mean()
            ),
        })

    frame = pd.DataFrame(rows)
    feasible = frame[
        frame.meets_95pct_utility
        & frame.category_group_log_rate_MAD.notna()
    ]
    if feasible.empty:
        return {
            "component": label,
            "policy_mode": mode,
            "kg_rank_threshold": threshold,
            "lambda": 0.0,
            "selection_status": "no_feasible_lambda_fallback",
            "tuning_fairness": None,
            "tuning_ndcg10": None,
        }, frame

    best = feasible.sort_values(
        ["category_group_log_rate_MAD", "NDCG@10", "lambda"],
        ascending=[True, False, True]
    ).iloc[0]
    chosen = {
        "component": label,
        "policy_mode": mode,
        "kg_rank_threshold": threshold,
        "lambda": float(best["lambda"]),
        "selection_status": "selected_on_tuning",
        "tuning_fairness": float(best["category_group_log_rate_MAD"]),
        "tuning_ndcg10": float(best["NDCG@10"]),
    }
    return chosen, frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-script", type=Path,
        default=Path("22_01_kg_faircrs_relevance_gate_ablation.py")
    )
    parser.add_argument("--candidates", type=Path,
        default=Path("results_corrected/adaptive_m2_validation_export22_candidates.csv"))
    parser.add_argument("--events", type=Path,
        default=Path("results_corrected/adaptive_m2_validation_export22_events.csv"))
    parser.add_argument("--mapping", type=Path,
        default=Path("results_corrected/m2_event_conversation_mapping.csv"))
    parser.add_argument("--catalog-pickle", type=Path,
        default=Path("catalog_audit_corrected.pkl"))
    parser.add_argument("--category-profiles", type=Path,
        default=Path("results_category_audit/experiment_21_22A/catalog_item_category_profiles.csv"))
    parser.add_argument("--out", type=Path,
        default=Path("results_corrected/experiment_22_02"))
    parser.add_argument("--lambdas", type=str, default="0,0.05,0.1,0.2,0.4,0.8,1.2")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    M = load_22_1_module(args.base_script)
    lambdas = sorted(set(float(x) for x in args.lambdas.split(",")))
    if not lambdas or any(not np.isfinite(x) or x < 0 for x in lambdas):
        raise ValueError("Lambda grid must contain finite nonnegative values.")
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates.")

    # Reuse the exact, already validated input loader from Experiment 22.1.
    class InputArgs:
        pass
    inp = InputArgs()
    inp.candidates = args.candidates
    inp.events = args.events
    inp.mapping = args.mapping
    inp.catalog_pickle = args.catalog_pickle
    inp.category_profiles = args.category_profiles

    args.out.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("EXPERIMENT 22.2 — KG-FairCRS COMPONENT ABLATION")
    print("=" * 78)
    print("Loading frozen Experiment 22.1 inputs...")
    data = M.load_inputs(inp)

    events = data["events"]
    event_ids = sorted(map(int, events.event_id.unique()))
    conv_for_event = {
        int(r.event_id): int(r.conv_id)
        for r in events[["event_id", "conv_id"]].drop_duplicates().itertuples(index=False)
    }
    if len(event_ids) != 3010:
        raise ValueError(f"Expected 3,010 events, found {len(event_ids)}.")
    n_conversations = len(set(conv_for_event.values()))
    if n_conversations != 883:
        raise ValueError(f"Expected 883 conversations, found {n_conversations}.")

    warm, tune, held = M.conversation_split(list(conv_for_event.values()))
    ids_by_conv = {}
    for event_id, conv_id in conv_for_event.items():
        ids_by_conv.setdefault(conv_id, []).append(event_id)
    warm_events = sorted(e for c in warm for e in ids_by_conv[c])
    tune_events = sorted(e for c in tune for e in ids_by_conv[c])
    held_events = sorted(e for c in held for e in ids_by_conv[c])

    print(
        f"Events={len(event_ids)} conversations={n_conversations} | "
        f"warm={len(warm_events)} tune={len(tune_events)} heldout={len(held_events)}"
    )

    warm_ledger = M.seed_ledger_with_m2(data, warm_events)
    baseline_tune = M.evaluate_static_baselines(data, tune_events)
    baseline_a1_tune = baseline_tune[baseline_tune.method == M.METHOD_M2]
    baseline_tune_ndcg = float(baseline_a1_tune.ndcg10.mean())
    utility_floor = 0.95 * baseline_tune_ndcg

    # Select the full method ONLY among KG-supported policies B1-B5 using
    # tuning data. B0 is deliberately excluded from the full KG-FairCRS choice.
    kg_candidates = [
        ("B1_rank_le_50", "threshold", 50),
        ("B2_rank_le_100", "threshold", 100),
        ("B3_rank_le_250", "threshold", 250),
        ("B4_rank_le_500", "threshold", 500),
        ("B5_graded_support", "graded", None),
    ]
    tuning_frames = []
    selected = []
    for label, mode, threshold in kg_candidates:
        choice, frame = tune_one_policy(
            M, data, warm_ledger, tune_events, baseline_tune_ndcg,
            label, mode, threshold, lambdas, update_history=True
        )
        selected.append(choice)
        tuning_frames.append(frame)

    kg_feasible = [
        x for x in selected if x["selection_status"] == "selected_on_tuning"
    ]
    if not kg_feasible:
        raise RuntimeError(
            "No KG-supported policy satisfies the 95% NDCG utility constraint "
            "on tuning data. Do not claim a validated KG-gated full method."
        )
    full_choice = sorted(
        kg_feasible,
        key=lambda x: (x["tuning_fairness"], -x["tuning_ndcg10"], x["lambda"])
    )[0]

    # Define one-factor-at-a-time ablations. Each adaptive variant gets its own
    # lambda selected on tuning; no held-out values are used for selection.
    component_specs = [
        ("FULL_KG_FairCRS", full_choice["policy_mode"],
         full_choice["kg_rank_threshold"], True, True),
        ("ABL_no_KG_support", "none", None, True, True),
        ("ABL_no_adaptive_calibration", full_choice["policy_mode"],
         full_choice["kg_rank_threshold"], False, True),
        ("ABL_frozen_exposure_history", full_choice["policy_mode"],
         full_choice["kg_rank_threshold"], True, False),
    ]

    parameter_rows = []
    tuning_rows = list(tuning_frames)
    for label, mode, threshold, adaptive, update_history in component_specs:
        if label == "FULL_KG_FairCRS":
            parameter_rows.append({
                "component": label,
                "policy_mode": mode,
                "kg_rank_threshold": threshold,
                "lambda": float(full_choice["lambda"]),
                "selection_status": "selected_on_tuning_among_B1_to_B5",
                "tuning_fairness": full_choice["tuning_fairness"],
                "tuning_ndcg10": full_choice["tuning_ndcg10"],
            })
            continue

        if label == "ABL_no_adaptive_calibration":
            choice, frame = tune_one_policy(
                M, data, warm_ledger, tune_events, baseline_tune_ndcg,
                label, mode, threshold, [0.0], update_history=True
            )
        else:
            choice, frame = tune_one_policy(
                M, data, warm_ledger, tune_events, baseline_tune_ndcg,
                label, mode, threshold, lambdas,
                update_history=update_history
            )
        parameter_rows.append(choice)
        tuning_rows.append(frame)

    pd.concat(tuning_rows, ignore_index=True).to_csv(
        args.out / "tuning_ablation_results.csv", index=False
    )
    parameter_df = pd.DataFrame(parameter_rows)
    parameter_df.to_csv(args.out / "selected_ablation_parameters.csv", index=False)

    # Held-out replay: warm-up -> tuning under each selected policy -> held-out.
    held_results = []
    for label, mode, threshold, adaptive, update_history in component_specs:
        choice = next(x for x in parameter_rows if x["component"] == label)
        lam = float(choice["lambda"]) if adaptive else 0.0

        ledger = M.clone_ledger(warm_ledger)
        _, ledger = M.run_policy_22_1(
            data, tune_events, initial_ledger=ledger, lam=lam,
            policy_mode=mode, threshold=threshold,
            update_ledger=update_history
        )
        held, _ = M.run_policy_22_1(
            data, held_events, initial_ledger=M.clone_ledger(ledger), lam=lam,
            policy_mode=mode, threshold=threshold,
            update_ledger=update_history
        )
        held["method"] = label
        held["lambda"] = lam
        held_results.append(held)

    held_all = pd.concat(held_results, ignore_index=True)
    held_all.to_csv(args.out / "heldout_event_recommendations.csv", index=False)

    # Unmodified A1 reference.
    a1_held = M.evaluate_static_baselines(data, held_events)
    a1_held = a1_held[a1_held.method == M.METHOD_M2].copy()
    a1_source = data["all_events"][
        (data["all_events"].method == M.METHOD_M2)
        & (data["all_events"].event_id.isin(held_events))
    ][["event_id", "top10_movie_ids", "top10_group_ids"]].copy()
    a1_source = a1_source.merge(
        data["mapping"][["event_id", "conv_id"]],
        on="event_id", validate="one_to_one"
    )
    a1_disparity, a1_detail, a1_exp, a1_opp = M.measure_partition_fairness(
        a1_source, held_events, data
    )

    summaries = [{
        "method": M.METHOD_M2,
        "component_removed": "none_reference",
        "events": int(len(a1_held)),
        "HR@10": float(a1_held.hit10.mean()),
        "NDCG@10": float(a1_held.ndcg10.mean()),
        "retention_vs_A1": 1.0,
        "category_group_log_rate_MAD": a1_disparity,
        "mean_eligible_categorized_candidates": np.nan,
        "mean_top10_promoted_positions": 0.0,
        "verified_categories": int(len({c for c, g in a1_opp})),
        "supported_category_group_cells": int(sum(v > M.EPS for v in a1_opp.values())),
    }]
    group_rows = [{"method": M.METHOD_M2, **r} for r in a1_detail]
    bootstrap_rows = []
    paired_rows = []

    for label, mode, threshold, adaptive, update_history in component_specs:
        result = held_all[held_all.method == label].copy()
        disparity, detail, exp, opp = M.measure_partition_fairness(
            result, held_events, data
        )
        summaries.append({
            "method": label,
            "component_removed": (
                "none" if label == "FULL_KG_FairCRS"
                else label.replace("ABL_", "")
            ),
            "events": int(len(result)),
            "HR@10": float(result.hit10.mean()),
            "NDCG@10": float(result.ndcg10.mean()),
            "retention_vs_A1": float(
                result.ndcg10.mean() / (a1_held.ndcg10.mean() + M.EPS)
            ),
            "category_group_log_rate_MAD": disparity,
            "mean_eligible_categorized_candidates": float(
                result.eligible_categorized_candidates.mean()
            ),
            "mean_top10_promoted_positions": float(
                result.top10_promoted_positions.mean()
            ),
            "verified_categories": int(len({c for c, g in opp})),
            "supported_category_group_cells": int(sum(v > M.EPS for v in opp.values())),
        })
        group_rows.extend({"method": label, **r} for r in detail)
        bootstrap_rows.append({
            "method": label,
            **M.cluster_bootstrap(result, n_boot=args.bootstrap)
        })
        paired_rows.append({
            "method": label,
            **M.paired_cluster_bootstrap_22_1(
                a1_held, result, n_boot=args.bootstrap,
                seed=M.SEED + len(paired_rows)
            )
        })

    comparison = pd.DataFrame(summaries)
    comparison.to_csv(args.out / "heldout_component_comparison.csv", index=False)
    pd.DataFrame(group_rows).to_csv(
        args.out / "heldout_category_group_exposure.csv", index=False
    )
    pd.DataFrame(bootstrap_rows).to_csv(
        args.out / "conversation_cluster_bootstrap.csv", index=False
    )
    pd.DataFrame(paired_rows).to_csv(
        args.out / "paired_bootstrap_vs_A1.csv", index=False
    )

    checks = {
        "experiment": "22.2_proposal_aligned_component_ablation",
        "split": "validation_only",
        "test_split_evaluated": False,
        "checkpoint_modified": False,
        "events": len(event_ids),
        "conversations": n_conversations,
        "warmup_events": len(warm_events),
        "tuning_events": len(tune_events),
        "heldout_events": len(held_events),
        "same_split_seed_as_22_0_and_22_1": True,
        "candidate_pool": "frozen M2 Top-500",
        "full_method_gate_selected_only_from_tuning": True,
        "full_method_selected_gate": full_choice["component"],
        "full_method_selected_lambda": full_choice["lambda"],
        "utility_constraint": "NDCG@10 >= 95% of A1_frequency_3 on tuning partition",
        "ablation_components": [
            "KG contextual support",
            "adaptive category exposure calibration",
            "online exposure-history updates",
        ],
        "no_category_calibration_reference": "unchanged A1_frequency_3",
        "paired_bootstrap_unit": "conversation",
        "bootstrap_replicates": args.bootstrap,
        "test_set_access": False,
        "limitations": [
            "This experiment uses the same validation events and split as 22.0/22.1.",
            "It is a component ablation, not independent external validation.",
            "Category disparity applies only to items with accepted category evidence.",
            "KG rank is a contextual retrieval-support proxy, not a semantic relevance label.",
            "The opportunity denominator is the existing rank-discounted candidate-pool proxy."
        ],
    }
    (args.out / "validation_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    summary = {
        "experiment": "22.2_proposal_aligned_component_ablation",
        "full_method_tuning_choice": full_choice,
        "selected_parameters": parameter_rows,
        "heldout_component_comparison": comparison.to_dict("records"),
        "checks": checks,
    }
    (args.out / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print("EXPERIMENT 22.2 COMPLETE")
    print("=" * 78)
    print("Full KG-supported policy selected on tuning:", full_choice["component"])
    print("Selected lambda:", full_choice["lambda"])
    print("\nHeld-out component comparison:")
    print(comparison[[
        "method", "HR@10", "NDCG@10", "retention_vs_A1",
        "category_group_log_rate_MAD"
    ]].to_string(index=False))
    print("\nOutputs:", args.out.resolve())
    print("Test split evaluated: NO")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nEXPERIMENT 22.2 FAILED: {exc}", file=sys.stderr)
        raise
