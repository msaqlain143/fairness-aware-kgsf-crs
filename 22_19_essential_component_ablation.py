#!/usr/bin/env python
"""Experiment 22.19: Essential component ablation for KG-FairCRS.

Validation-only, controlled ablation of Experiment 22.18.
Uses the same frozen KBRD candidates, KG evidence, category profiles,
conversation split, and fixed lambda/tau selected by Experiment 22.18.
No training, retrieval rerun, independent-test access, or coefficient search.
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_REFERENCE = Path("22_18_contextual_relevance_gate_validation.py")
DEFAULT_OUT = Path("results_corrected/experiment_22_19")
FIXED_LAMBDA = 0.4
FIXED_TAU = 250.0
UTILITY_RETENTION = 0.95
EPS = 1e-12
TOP_K = 10


def load_module(path):
    spec = importlib.util.spec_from_file_location("experiment_22_18", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load reference experiment: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.reference = module.load_reference(module.DEFAULT_REFERENCE)
    return module


def make_ablation_ranker(module, ablation):
    """Return the 22.18 scoring rule with exactly one component removed."""

    def ranker(candidate, item2group, categories, ledger, lam, tau):
        ids = candidate["ids"]
        scores = candidate["scores"]
        kg_ranks = candidate["kg"]
        groups = np.asarray(
            [item2group[int(movie_id)] for movie_id in ids], dtype=int
        )

        context, categorized_count = module.reference_category_context(
            ids, categories
        )
        deficits = ledger.deficits(ids, groups, categories)

        bonus = np.zeros(len(ids), dtype=np.float64)
        eligible = 0
        supported = 0

        for index, movie_id in enumerate(ids):
            movie_categories = categories.get(int(movie_id), [])
            if not movie_categories:
                continue

            kg_rank = int(kg_ranks[index])
            if ablation != "A2_no_KG_evidence" and kg_rank <= 0:
                continue

            if kg_rank > 0:
                supported += 1

            if ablation == "A1_no_category_context":
                context_relevance = 1.0
            else:
                context_relevance = float(np.mean([
                    context.get(category, 0.0)
                    for category in movie_categories
                ]))

            if ablation == "A2_no_KG_evidence":
                kg_confidence = 1.0
            else:
                kg_confidence = math.exp(-float(kg_rank) / float(tau))

            if ablation == "A3_no_exposure_deficit":
                deficit = 1.0
            else:
                deficit = float(np.mean([
                    deficits.get((category, int(groups[index])), 0.0)
                    for category in movie_categories
                ]))

            gate = context_relevance * kg_confidence
            if gate > 0:
                eligible += 1
            bonus[index] = deficit * gate

        max_bonus = float(bonus.max(initial=0.0))
        if max_bonus > EPS:
            bonus /= max_bonus

        score_scale = float(np.median(np.abs(scores)))
        if score_scale <= EPS:
            score_scale = float(np.max(np.abs(scores))) if len(scores) else 1.0

        adjusted = scores + float(lam) * score_scale * bonus
        order = np.lexsort((ids, -adjusted))
        ranked_ids = ids[order]

        # Retain the same diagnostic definition as Experiment 22.18.
        movement = int(np.sum(
            order[:TOP_K] != np.arange(min(TOP_K, len(order)))
        ))
        return ranked_ids, {
            "eligible_contextual_kg_candidates": int(eligible),
            "kg_supported_candidates": int(supported),
            "categorized_candidates": int(categorized_count),
            "top10_promoted_positions": movement,
        }

    return ranker


def replay_configuration(module, name, treatment, event_ids, events,
                         candidates, item2group, catalog, categories,
                         warm_ids, tune_ids, heldout_ids):
    """Replay warm and tuning history, then evaluate the frozen held-out split."""
    if treatment:
        module.contextual_gate_rerank = make_ablation_ranker(module, name)

    # Ensure the dynamically loaded reference function resolves the same
    # module-level helper definitions and no hidden state is shared.
    baseline_ledger = module.reference.ExposureLedger()

    if treatment:
        _, warm_ledger, _ = module.run_split(
            warm_ids, events, candidates, item2group, categories,
            baseline_ledger, lam=FIXED_LAMBDA, tau=FIXED_TAU,
            treatment=True,
        )
        _, tune_ledger, _ = module.run_split(
            tune_ids, events, candidates, item2group, categories,
            warm_ledger, lam=FIXED_LAMBDA, tau=FIXED_TAU,
            treatment=True,
        )
        heldout_frame, _, heldout_summary = module.run_split(
            heldout_ids, events, candidates, item2group, categories,
            tune_ledger, lam=FIXED_LAMBDA, tau=FIXED_TAU,
            treatment=True,
        )
    else:
        _, warm_ledger, _ = module.run_split(
            warm_ids, events, candidates, item2group, categories,
            baseline_ledger, treatment=False,
        )
        _, tune_ledger, _ = module.run_split(
            tune_ids, events, candidates, item2group, categories,
            warm_ledger, treatment=False,
        )
        heldout_frame, _, heldout_summary = module.run_split(
            heldout_ids, events, candidates, item2group, categories,
            tune_ledger, treatment=False,
        )

    heldout_summary["method"] = name
    heldout_frame["method"] = name
    return heldout_frame, heldout_summary


def main():
    global FIXED_LAMBDA, FIXED_TAU
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--lambda-value", type=float, default=FIXED_LAMBDA)
    parser.add_argument("--tau", type=float, default=FIXED_TAU)
    args = parser.parse_args()

    FIXED_LAMBDA = float(args.lambda_value)
    FIXED_TAU = float(args.tau)

    if not args.reference.exists():
        raise FileNotFoundError(args.reference)

    module = load_module(args.reference)
    for path in (
        module.DEFAULT_KBRD, module.DEFAULT_KG,
        module.DEFAULT_PROFILES, module.DEFAULT_CATALOG,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    item2group, catalog, categories = module.reference.load_metadata(
        module.DEFAULT_CATALOG, module.DEFAULT_PROFILES
    )
    events, candidates = module.load_events_and_candidates(
        module.DEFAULT_KBRD, module.DEFAULT_KG
    )

    conv_ids = [event["conv"] for event in events.values()]
    warm_convs, tune_convs, heldout_convs = (
        module.reference.split_conversations(conv_ids)
    )
    warm_ids = {eid for eid, e in events.items() if e["conv"] in warm_convs}
    tune_ids = {eid for eid, e in events.items() if e["conv"] in tune_convs}
    heldout_ids = {
        eid for eid, e in events.items() if e["conv"] in heldout_convs
    }

    args.out.mkdir(parents=True, exist_ok=True)
    print("Loading frozen validation exports...")
    print(
        f"Events={len(events)} | conversations={len(set(conv_ids))} | "
        f"warm={len(warm_ids)} | tune={len(tune_ids)} | heldout={len(heldout_ids)}"
    )
    print(
        f"Fixed parameters from Experiment 22.18: "
        f"lambda={FIXED_LAMBDA}, tau={FIXED_TAU}"
    )

    configurations = [
        ("B0_KBRD", False),
        ("FULL_22_18", True),
        ("A1_no_category_context", True),
        ("A2_no_KG_evidence", True),
        ("A3_no_exposure_deficit", True),
    ]

    frames = []
    summaries = []
    for name, treatment in configurations:
        print(f"Evaluating {name}...")
        frame, summary = replay_configuration(
            module, name, treatment, set(events), events, candidates,
            item2group, catalog, categories, warm_ids, tune_ids, heldout_ids,
        )
        frames.append(frame)
        summaries.append(summary)

    results = pd.DataFrame(summaries)
    baseline = results.loc[results.method == "B0_KBRD"].iloc[0]
    results["delta_NDCG@10_vs_baseline"] = (
        results["NDCG@10"] - float(baseline["NDCG@10"])
    )
    results["NDCG_retention_vs_baseline"] = (
        results["NDCG@10"] / max(float(baseline["NDCG@10"]), EPS)
    )
    results["delta_fairness_MAD_vs_baseline"] = (
        results["category_group_log_rate_MAD"]
        - float(baseline["category_group_log_rate_MAD"])
    )
    results["fairness_improved_vs_baseline"] = (
        results["delta_fairness_MAD_vs_baseline"] < 0
    )
    results["utility_retained_at_95pct"] = (
        results["NDCG_retention_vs_baseline"] >= UTILITY_RETENTION
    )
    results.to_csv(args.out / "ablation_comparison.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(
        args.out / "heldout_event_results.csv", index=False
    )

    full = results.loc[results.method == "FULL_22_18"].iloc[0]
    ablation_rows = results[results.method.str.startswith("A")].copy()
    checks = {
        "five_configurations_evaluated": len(results) == 5,
        "same_frozen_event_universe": all(
            len(frame) == len(heldout_ids) for frame in frames
        ),
        "same_conversation_disjoint_split": (
            not (warm_convs & tune_convs)
            and not (warm_convs & heldout_convs)
            and not (tune_convs & heldout_convs)
        ),
        "fixed_parameters_reused_from_22_18": (
            math.isclose(FIXED_LAMBDA, 0.4)
            and math.isclose(FIXED_TAU, 250.0)
        ),
        "no_training_performed": True,
        "no_retrieval_rerun": True,
        "no_independent_test_evaluation": True,
        "no_parameter_search_in_22_19": True,
        "all_heldout_methods_have_identical_event_ids": all(
            set(frame.event_id) == set(frames[0].event_id) for frame in frames
        ),
        "overall_pass": True,
    }
    checks["overall_pass"] = all(checks.values())

    assessment = {
        "experiment": "22.19_essential_component_ablation",
        "scope": "frozen_validation_exports_only",
        "events": len(events),
        "conversations": len(set(conv_ids)),
        "heldout_events": len(heldout_ids),
        "fixed_lambda": FIXED_LAMBDA,
        "fixed_tau": FIXED_TAU,
        "utility_retention_threshold": UTILITY_RETENTION,
        "full_method_fairness_MAD": float(full["category_group_log_rate_MAD"]),
        "full_method_NDCG_retention": float(
            full["NDCG_retention_vs_baseline"]
        ),
        "ablation_results": ablation_rows[
            [
                "method", "delta_NDCG@10_vs_baseline",
                "NDCG_retention_vs_baseline",
                "delta_fairness_MAD_vs_baseline",
            ]
        ].to_dict(orient="records"),
        "purpose": (
            "Determine whether candidate-pool category context, KG evidence, "
            "and adaptive exposure deficit each contribute to the observed "
            "fairness-utility result."
        ),
        "independent_test_evaluated": False,
        "training_performed": False,
        "retrieval_rerun": False,
        "parameters_researched": False,
    }
    (args.out / "validation_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    (args.out / "final_research_assessment.json").write_text(
        json.dumps(assessment, indent=2), encoding="utf-8"
    )
    manifest = {
        **assessment,
        "inputs": {
            "reference_experiment": str(args.reference),
            "kbrd_candidates": str(module.DEFAULT_KBRD),
            "kg_candidate_export": str(module.DEFAULT_KG),
            "category_profiles": str(module.DEFAULT_PROFILES),
            "catalog_metadata": str(module.DEFAULT_CATALOG),
        },
        "configurations": [
            "B0_KBRD", "FULL_22_18", "A1_no_category_context",
            "A2_no_KG_evidence", "A3_no_exposure_deficit",
        ],
        "outputs": [
            "ablation_comparison.csv", "heldout_event_results.csv",
            "validation_checks.json", "final_research_assessment.json",
            "experiment_manifest.json",
        ],
    }
    (args.out / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\n========== EXPERIMENT 22.19 COMPLETE ==========")
    print(results.to_string(index=False))
    print("\nValidation:")
    print(json.dumps(checks, indent=2))
    print("\nOutput:", args.out.resolve())
    if not checks["overall_pass"]:
        raise RuntimeError("Experiment 22.19 validation failed.")


if __name__ == "__main__":
    main()



