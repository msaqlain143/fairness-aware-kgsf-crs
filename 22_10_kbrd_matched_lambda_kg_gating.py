#!/usr/bin/env python
"""
Experiment 22.10: matched-lambda attribution of KG gating in KBRD reranking.

Purpose
-------
Isolate the incremental effect of KG retrieval-support gating while holding
the reranking strength (lambda) fixed.

Comparisons:
  lambda=0.20: category-only vs KG-gated
  lambda=0.40: category-only vs KG-gated

The script reuses the frozen inputs, split, ranking implementation, ledger,
and category/fairness definitions from 22.09. No model training, candidate
generation, hyperparameter selection, or test-split evaluation is performed.

Important interpretation:
KG rank > 0 means retrieval support in the KG candidate export. It is not
proof of semantic relevance or correctness of a movie-entity link.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 22100
BOOT_SEED = 22101
MATCHED_LAMBDAS = (0.20, 0.40)
EPS = 1e-12


def load_base_module(path: Path):
    spec = importlib.util.spec_from_file_location("exp22_09_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Experiment 22.09 source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = (
        "load_metadata", "as_list", "split_conversations", "Ledger",
        "baseline_frame", "run_policy", "evaluate_fairness",
        "summarize", "contribution_vectors", "disparity_vectors",
        "TOP_K", "CANDIDATE_K",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"22.09 source is missing required functions/constants: {missing}")
    return module


def load_frozen_data(base, args):
    meta, groups, catalog, cats, profiles = base.load_metadata(
        args.catalog_pickle, args.category_profiles
    )

    k = pd.read_csv(args.kbrd)
    manifest = pd.read_csv(args.manifest)
    kg = pd.read_csv(
        args.kg,
        usecols=["event_id", "method", "candidate_movie_ids", "candidate_kg_ranks"],
    )
    kg = kg[kg.method == "A1_frequency_3"].copy()

    for frame in (k, manifest, kg):
        frame["event_id"] = frame.event_id.astype(int)
        if frame.event_id.duplicated().any():
            raise ValueError("Duplicate event IDs found in a frozen input.")

    if set(k.event_id) != set(manifest.event_id) or set(k.event_id) != set(kg.event_id):
        raise ValueError("KBRD, manifest, and KG event IDs do not match exactly.")

    manifest = manifest[["event_id", "conv_id", "target_movie_id"]]
    k = k.merge(
        manifest, on="event_id", how="inner", validate="one_to_one",
        suffixes=("", "_manifest"),
    )
    for col in ("conv_id", "target_movie_id"):
        other = f"{col}_manifest"
        if other in k and not np.array_equal(k[col].to_numpy(), k[other].to_numpy()):
            raise ValueError(f"KBRD/manifest {col} mismatch.")

    kg_map = {}
    for row in kg.itertuples(index=False):
        movie_ids = np.asarray(
            base.as_list(row.candidate_movie_ids, "KG candidate IDs"), dtype=np.int64
        )
        ranks = np.asarray(
            base.as_list(row.candidate_kg_ranks, "KG ranks"), dtype=np.int64
        )
        if (
            len(movie_ids) != len(ranks)
            or len(np.unique(movie_ids)) != len(movie_ids)
            or np.any(ranks < 0)
        ):
            raise ValueError(f"Invalid KG candidate/rank arrays at event {row.event_id}.")
        kg_map[int(row.event_id)] = {
            int(movie_id): int(rank) for movie_id, rank in zip(movie_ids, ranks)
        }

    events, candidates = {}, {}
    for row in k.sort_values("event_id").itertuples(index=False):
        event_id = int(row.event_id)
        ids = np.asarray(base.as_list(row.top_movie_ids, "KBRD IDs"), dtype=np.int64)
        scores = np.asarray(base.as_list(row.top_scores, "KBRD scores"), dtype=float)
        if (
            len(ids) != base.CANDIDATE_K
            or len(scores) != base.CANDIDATE_K
            or len(np.unique(ids)) != base.CANDIDATE_K
        ):
            raise ValueError(f"Invalid KBRD Top-500 at event {event_id}.")
        if not np.isfinite(scores).all() or not set(map(int, ids)).issubset(catalog):
            raise ValueError(f"Invalid KBRD scores/catalog IDs at event {event_id}.")

        transferred = np.asarray(
            [kg_map[event_id].get(int(movie_id), 0) for movie_id in ids],
            dtype=np.int64,
        )
        events[event_id] = {
            "conv": int(row.conv_id),
            "target": int(row.target_movie_id),
        }
        candidates[event_id] = {"ids": ids, "scores": scores, "kg": transferred}

    if len(events) != 3010 or len({v["conv"] for v in events.values()}) != 883:
        raise ValueError("Expected exactly 3,010 events and 883 conversations.")

    return {
        "events": events,
        "cand": candidates,
        "groups": groups,
        "cats": cats,
        "catalog": catalog,
        "category_profiles": profiles,
        "metadata": meta,
    }


def paired_cluster_bootstrap(
    base, frames, event_conversation, held_conversations, categories,
    comparisons, n_boot, data,
):
    """Direct paired differences using identical conversation resamples."""
    category_index = {category: i for i, category in enumerate(categories)}
    n_categories = len(categories)
    events_by_conversation = {int(c): [] for c in held_conversations}
    for event_id, conversation_id in event_conversation.items():
        if int(conversation_id) in events_by_conversation:
            events_by_conversation[int(conversation_id)].append(int(event_id))

    conversations = np.asarray(sorted(events_by_conversation), dtype=np.int64)
    rng = np.random.default_rng(BOOT_SEED)
    samples = rng.choice(conversations, size=(n_boot, len(conversations)), replace=True)

    indexed = {name: frame.set_index("event_id") for name, frame in frames.items()}
    contributions = {
        name: base.contribution_vectors(
            frame, sorted(event_conversation_event_ids(frame)),
            data, category_index, n_categories,
        )
        for name, frame in frames.items()
    }

    output = []
    for pair_name, control_name, gated_name, lam in comparisons:
        control = indexed[control_name]
        gated = indexed[gated_name]
        control_exp, control_opp = contributions[control_name]
        gated_exp, gated_opp = contributions[gated_name]

        fair_deltas, hr_deltas, ndcg_deltas = [], [], []
        for sample in samples:
            sampled_events = []
            for conversation_id in sample:
                sampled_events.extend(events_by_conversation[int(conversation_id)])

            control_util = control.loc[sampled_events][["hit10", "ndcg10"]].to_numpy(float).mean(axis=0)
            gated_util = gated.loc[sampled_events][["hit10", "ndcg10"]].to_numpy(float).mean(axis=0)

            c_exp = np.zeros(n_categories * 5)
            c_opp = np.zeros(n_categories * 5)
            g_exp = np.zeros(n_categories * 5)
            g_opp = np.zeros(n_categories * 5)
            for event_id in sampled_events:
                c_exp += control_exp[event_id]
                c_opp += control_opp[event_id]
                g_exp += gated_exp[event_id]
                g_opp += gated_opp[event_id]

            c_fair = base.disparity_vectors(c_exp, c_opp, n_categories)
            g_fair = base.disparity_vectors(g_exp, g_opp, n_categories)
            fair_deltas.append(g_fair - c_fair)
            hr_deltas.append(gated_util[0] - control_util[0])
            ndcg_deltas.append(gated_util[1] - control_util[1])

        # Full held-out point differences, not bootstrap means.
        control_fair = fairness_from_contributions(
            base, control_exp, control_opp, n_categories
        )
        gated_fair = fairness_from_contributions(
            base, gated_exp, gated_opp, n_categories
        )
        control_hr = float(control.hit10.mean())
        gated_hr = float(gated.hit10.mean())
        control_ndcg = float(control.ndcg10.mean())
        gated_ndcg = float(gated.ndcg10.mean())

        fair_arr = np.asarray(fair_deltas, dtype=float)
        hr_arr = np.asarray(hr_deltas, dtype=float)
        ndcg_arr = np.asarray(ndcg_deltas, dtype=float)

        def ci(values):
            values = values[np.isfinite(values)]
            if not len(values):
                return float("nan"), float("nan")
            return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))

        fair_low, fair_high = ci(fair_arr)
        hr_low, hr_high = ci(hr_arr)
        ndcg_low, ndcg_high = ci(ndcg_arr)

        output.append({
            "comparison": pair_name,
            "lambda": lam,
            "control_method": control_name,
            "kg_gated_method": gated_name,
            "delta_fairness_gated_minus_control": gated_fair - control_fair,
            "fairness_delta_ci95_low": fair_low,
            "fairness_delta_ci95_high": fair_high,
            "bootstrap_probability_fairness_improves": float(np.mean(fair_arr < 0)),
            "delta_HR@10_gated_minus_control": gated_hr - control_hr,
            "delta_HR@10_ci95_low": hr_low,
            "delta_HR@10_ci95_high": hr_high,
            "delta_NDCG@10_gated_minus_control": gated_ndcg - control_ndcg,
            "delta_NDCG@10_ci95_low": ndcg_low,
            "delta_NDCG@10_ci95_high": ndcg_high,
            "bootstrap_replicates": n_boot,
            "conversation_clusters": len(conversations),
        })
    return pd.DataFrame(output)


def event_conversation_event_ids(frame):
    return frame.event_id.astype(int).tolist()


def fairness_from_contributions(base, exposure_map, opportunity_map, n_categories):
    exposure = np.zeros(n_categories * 5)
    opportunity = np.zeros(n_categories * 5)
    for event_id in exposure_map:
        exposure += exposure_map[event_id]
        opportunity += opportunity_map[event_id]
    return base.disparity_vectors(exposure, opportunity, n_categories)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-script", type=Path,
        default=Path("22_09_kbrd_cross_backbone_validation.py"))
    parser.add_argument("--kbrd", type=Path,
        default=Path("results_corrected/experiment_22_04B/kbrd_validation_top500.csv"))
    parser.add_argument("--manifest", type=Path,
        default=Path("results_corrected/experiment_22_04B/kbrd_validation_event_manifest.csv"))
    parser.add_argument("--kg", type=Path,
        default=Path("results_corrected/adaptive_m2_validation_export22_candidates.csv"))
    parser.add_argument("--catalog-pickle", type=Path, default=Path("catalog_audit_corrected.pkl"))
    parser.add_argument("--category-profiles", type=Path,
        default=Path("results_category_audit/experiment_21_22A/catalog_item_category_profiles.csv"))
    parser.add_argument("--out", type=Path,
        default=Path("results_corrected/experiment_22_10"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    if args.bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates; final run should use 10,000.")

    base = load_base_module(args.base_script)
    data = load_frozen_data(base, args)

    event_ids = sorted(data["events"])
    event_conversation = {eid: data["events"][eid]["conv"] for eid in event_ids}
    warm, tune, held = base.split_conversations(event_conversation.values())
    events_by_conversation = defaultdict(list)
    for eid, cid in event_conversation.items():
        events_by_conversation[int(cid)].append(int(eid))
    warm_events = sorted(e for c in warm for e in events_by_conversation[c])
    tune_events = sorted(e for c in tune for e in events_by_conversation[c])
    held_events = sorted(e for c in held for e in events_by_conversation[c])

    # Validate the frozen split by the event counts already reported by 22.09.
    # Do not hard-code conversation counts: the authoritative split function
    # determines those counts, and 22.09's published run reported event counts.
    print(
        "Conversation split diagnostics: "
        f"warm={len(warm)} ({len(warm_events)} events), "
        f"tune={len(tune)} ({len(tune_events)} events), "
        f"heldout={len(held)} ({len(held_events)} events)"
    )
    if (warm & tune) or (warm & held) or (tune & held):
        raise RuntimeError("Conversation partitions overlap.")
    if set(warm) | set(tune) | set(held) != set(event_conversation.values()):
        raise RuntimeError("Conversation partitions do not cover all validation conversations.")
    if (
        len(warm_events) != 1817
        or len(tune_events) != 569
        or len(held_events) != 624
        or set(warm_events) | set(tune_events) | set(held_events) != set(event_ids)
    ):
        raise RuntimeError(
            "Event split differs from the frozen 22.09 split: "
            f"warm={len(warm_events)}, tune={len(tune_events)}, "
            f"heldout={len(held_events)}. Do not proceed until the split "
            "implementation/seed is reconciled with Experiment 22.09."
        )

    # Reconstruct the same KBRD-only warm-up ledger as Experiment 22.09.
    warm_ledger = base.Ledger()
    for eid in warm_events:
        candidate = data["cand"][eid]
        groups = np.asarray([data["groups"][int(m)] for m in candidate["ids"]], dtype=int)
        warm_ledger.update(candidate["ids"], data["cats"], groups)

    # B0 is the unchanged KBRD ranking. The four matched arms use fixed lambda.
    frames = {"B0_KBRD": base.baseline_frame(data, held_events)}
    arm_specs = []
    comparisons = []
    for lam in MATCHED_LAMBDAS:
        token = str(lam).replace(".", "p")
        no_gate = f"B_category_only_lambda_{token}"
        kg_gate = f"B_KG_gated_lambda_{token}"
        arm_specs.extend([
            (no_gate, float(lam), False),
            (kg_gate, float(lam), True),
        ])
        comparisons.append((f"KG_gating_at_lambda_{lam:.2f}", no_gate, kg_gate, float(lam)))

    tuning_summaries = []
    for method, lam, gate in arm_specs:
        # Each policy is replayed independently through the same tune events.
        tune_frame, ledger_after_tune = base.run_policy(
            data, tune_events, warm_ledger, lam, gate
        )
        held_frame, _ = base.run_policy(
            data, held_events, ledger_after_tune, lam, gate
        )
        frames[method] = held_frame

        tune_fairness, _ = base.evaluate_fairness(tune_frame, tune_events, data)
        tuning_summaries.append({
            "method": method,
            "lambda": lam,
            "kg_gate": gate,
            "tuning_HR@10": float(tune_frame.hit10.mean()),
            "tuning_NDCG@10": float(tune_frame.ndcg10.mean()),
            "tuning_category_group_log_rate_MAD": tune_fairness,
            "lambda_selected_on_test": False,
        })

    args.out.mkdir(parents=True, exist_ok=True)

    # Held-out point estimates and fairness details.
    point_rows, fairness_rows, detail_rows = [], [], []
    for method, frame in frames.items():
        summary = base.summarize(frame)
        summary["method"] = method
        if method == "B0_KBRD":
            summary["lambda"] = 0.0
            summary["kg_gate"] = False
        else:
            spec = next(x for x in arm_specs if x[0] == method)
            summary["lambda"] = spec[1]
            summary["kg_gate"] = spec[2]
        point_rows.append(summary)

        fairness, detail = base.evaluate_fairness(frame, held_events, data)
        fairness_rows.append({
            "method": method,
            "lambda": summary["lambda"],
            "kg_gate": summary["kg_gate"],
            "category_group_log_rate_MAD": fairness,
            "categories_with_supported_groups": len({x["category"] for x in detail}),
        })
        for row in detail:
            detail_rows.append({"method": method, **row})

    point_df = pd.DataFrame(point_rows)
    fairness_df = pd.DataFrame(fairness_rows)
    tuning_df = pd.DataFrame(tuning_summaries)
    point_df.to_csv(args.out / "matched_lambda_point_estimates.csv", index=False)
    fairness_df.to_csv(args.out / "category_fairness_results.csv", index=False)
    tuning_df.to_csv(args.out / "tuning_replay_diagnostics.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(args.out / "category_fairness_detail.csv", index=False)
    pd.concat([frame.assign(method=name) for name, frame in frames.items()],
              ignore_index=True).to_csv(args.out / "event_level_results.csv", index=False)

    categories = sorted({category for values in data["cats"].values() for category in values})
    category_index = {category: i for i, category in enumerate(categories)}
    contributions = {}
    for method, frame in frames.items():
        contributions[method] = base.contribution_vectors(
            frame, held_events, data, category_index, len(categories)
        )

    # Direct paired inference: same conversation resamples for both methods
    # in each matched-lambda comparison.
    bootstrap_df = paired_cluster_bootstrap(
        base, frames, event_conversation, set(event_conversation[e] for e in held_events),
        categories, comparisons, args.bootstrap, data,
    )
    bootstrap_df.to_csv(args.out / "direct_paired_kg_gating_effects.csv", index=False)

    # Save a compact, explicit record of the matched contrasts.
    comparison_df = pd.DataFrame([
        {"comparison": name, "lambda": lam, "control_method": control,
         "kg_gated_method": gated, "only_intended_difference": "KG support gate"}
        for name, control, gated, lam in comparisons
    ])
    comparison_df.to_csv(args.out / "matched_comparisons.csv", index=False)

    kg_counts = np.asarray([int(np.sum(data["cand"][eid]["kg"] > 0)) for eid in event_ids])
    checks = {
        "experiment": "22.10_KBRD_matched_lambda_KG_gating_attribution",
        "validation_only": True,
        "test_split_evaluated": False,
        "events": len(event_ids),
        "conversations": len(set(event_conversation.values())),
        "warm_events": len(warm_events),
        "tuning_events": len(tune_events),
        "heldout_events": len(held_events),
        "warm_conversations": len(warm),
        "tuning_conversations": len(tune),
        "heldout_conversations": len(held),
        "split_seed": getattr(base, "SEED", SEED),
        "matched_lambdas": list(MATCHED_LAMBDAS),
        "candidate_pool": "KBRD Top-500",
        "KG transfer key": "(event_id, movie_id)",
        "missing_KG_evidence_default": 0,
        "KG_support_definition": "KG rank > 0; retrieval support, not semantic relevance",
        "events_with_KG_supported_candidates": int(np.sum(kg_counts > 0)),
        "events_without_KG_supported_candidates": int(np.sum(kg_counts == 0)),
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": BOOT_SEED,
        "bootstrap_sampling_unit": "held-out conversation_id",
        "same_resamples_for_all_methods": True,
        "direct_paired_contrasts": [
            {"lambda": lam, "control": control, "KG_gated": gated}
            for _, control, gated, lam in comparisons
        ],
        "test_data_loaded": False,
    }
    with open(args.out / "validation_checks.json", "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2)

    manifest = {
        **checks,
        "inputs": {
            "base_script": str(args.base_script),
            "kbrd": str(args.kbrd),
            "manifest": str(args.manifest),
            "kg": str(args.kg),
            "catalog_pickle": str(args.catalog_pickle),
            "category_profiles": str(args.category_profiles),
        },
        "outputs": [
            "matched_lambda_point_estimates.csv",
            "category_fairness_results.csv",
            "category_fairness_detail.csv",
            "event_level_results.csv",
            "tuning_replay_diagnostics.csv",
            "direct_paired_kg_gating_effects.csv",
            "matched_comparisons.csv",
            "validation_checks.json",
        ],
        "interpretation_notes": [
            "Matched lambda controls reranking strength when attributing the incremental KG-gating effect.",
            "The policy is sequential: each arm updates its own exposure ledger during tuning and held-out replay.",
            "KG rank > 0 indicates retrieval support, not semantic relevance.",
            "Conversation-cluster bootstrap intervals quantify held-out sampling uncertainty, not category-label uncertainty.",
            "This is validation-only evidence and is not an independent test-set result.",
        ],
    }
    with open(args.out / "experiment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nEXPERIMENT 22.10 COMPLETE")
    print("\nMatched-lambda held-out point estimates:")
    print(point_df.to_string(index=False))
    print("\nDirect paired KG-gating effects:")
    print(bootstrap_df.to_string(index=False))
    print("\nOutputs:", args.out.resolve())
    print("Test split evaluated: NO")


if __name__ == "__main__":
    main()
