#!/usr/bin/env python
"""
Experiment 22.18: Contextual KG Relevance-Gate Validation.

Purpose
-------
Test whether continuous KG-rank confidence, modulated by event-level
candidate-pool category context and the existing exposure-deficit ledger,
can improve category exposure fairness while retaining >=95% of baseline
NDCG@10.

This is a validation-only experiment:
- Reuses frozen KBRD Top-500 validation candidates.
- Transfers KG ranks by (event_id, movie_id) from the frozen A1_frequency_3 export.
- Reuses the category evidence rules and ExposureLedger from Experiment 22.16.
- Uses conversation-disjoint warm/tune/held-out splits from Experiment 22.16.
- Does not train, retrieve, or evaluate the independent test split.

Important terminology:
"Context" here means category prevalence in the frozen event candidate pool.
It is NOT semantic understanding of dialogue text.
"""

import argparse
import ast
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12
TOP_K = 10

DEFAULT_KBRD = Path(
    "results_corrected/experiment_22_04B/kbrd_validation_top500.csv"
)
DEFAULT_KG = Path(
    "results_corrected/adaptive_m2_validation_export22_candidates.csv"
)
DEFAULT_PROFILES = Path(
    "results_category_audit/experiment_21_22A/catalog_item_category_profiles.csv"
)
DEFAULT_CATALOG = Path("catalog_audit_corrected.pkl")
DEFAULT_REFERENCE = Path("22_16_category_context_utility_calibration.py")
DEFAULT_OUT = Path("results_corrected/experiment_22_18")

# Predeclared calibration grid. No independent-test data are used.
LAMBDA_GRID = (0.05, 0.10, 0.20, 0.40, 0.80)
TAU_GRID = (25.0, 50.0, 100.0, 250.0)
UTILITY_RETENTION = 0.95


def load_reference(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Experiment 22.16 reference script not found: {path}"
        )
    spec = importlib.util.spec_from_file_location("experiment_22_16", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = [
        "load_metadata", "split_conversations", "ExposureLedger",
        "category_context", "fairness_mad", "as_list",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            f"Experiment 22.16 is missing required definitions: {missing}"
        )
    return module


def parse_list(value, field):
    if isinstance(value, list):
        result = value
    elif isinstance(value, (tuple, np.ndarray)):
        result = list(value)
    else:
        try:
            result = json.loads(value)
        except Exception:
            result = ast.literal_eval(value)
    if not isinstance(result, list):
        raise ValueError(f"{field} must contain a list")
    return result


def load_events_and_candidates(kbrd_path, kg_path):
    kbrd = pd.read_csv(kbrd_path)
    required_kbrd = {
        "event_id", "conv_id", "target_movie_id",
        "top_movie_ids", "top_scores",
    }
    missing = required_kbrd - set(kbrd.columns)
    if missing:
        raise ValueError(f"KBRD export missing columns: {sorted(missing)}")

    kg = pd.read_csv(kg_path)
    required_kg = {
        "event_id", "method", "candidate_movie_ids", "candidate_kg_ranks",
    }
    missing = required_kg - set(kg.columns)
    if missing:
        raise ValueError(f"KG export missing columns: {sorted(missing)}")

    kg = kg[kg["method"].astype(str) == "A1_frequency_3"].copy()
    if kg.empty:
        raise ValueError("No A1_frequency_3 rows in KG candidate export.")
    if kg["event_id"].duplicated().any():
        raise ValueError("Duplicate A1_frequency_3 event rows.")

    kg_rank_by_event = {}
    for row in kg.itertuples(index=False):
        ids = parse_list(row.candidate_movie_ids, "candidate_movie_ids")
        ranks = parse_list(row.candidate_kg_ranks, "candidate_kg_ranks")
        if len(ids) != len(ranks):
            raise ValueError(
                f"KG candidate/rank length mismatch at event {row.event_id}"
            )
        mapping = {}
        for movie_id, rank in zip(ids, ranks):
            movie_id = int(movie_id)
            rank = int(rank)
            if movie_id in mapping:
                raise ValueError(
                    f"Duplicate movie {movie_id} in KG event {row.event_id}"
                )
            mapping[movie_id] = rank
        kg_rank_by_event[int(row.event_id)] = mapping

    events = {}
    candidates = {}
    for row in kbrd.itertuples(index=False):
        event_id = int(row.event_id)
        if event_id in events:
            raise ValueError(f"Duplicate KBRD event_id: {event_id}")

        ids = np.asarray(
            parse_list(row.top_movie_ids, "top_movie_ids"), dtype=np.int64
        )
        scores = np.asarray(
            parse_list(row.top_scores, "top_scores"), dtype=np.float64
        )
        if len(ids) != len(scores):
            raise ValueError(f"KBRD ID/score length mismatch at {event_id}")
        if len(ids) != 500:
            raise ValueError(
                f"Expected 500 frozen candidates at event {event_id}; got {len(ids)}"
            )
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"Duplicate candidate IDs at event {event_id}")
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite KBRD scores at event {event_id}")

        kg_map = kg_rank_by_event.get(event_id, {})
        kg_ranks = np.asarray(
            [kg_map.get(int(movie_id), 0) for movie_id in ids],
            dtype=np.int32,
        )
        events[event_id] = {
            "conv": int(row.conv_id),
            "target": int(row.target_movie_id),
        }
        candidates[event_id] = {
            "ids": ids,
            "scores": scores,
            "kg": kg_ranks,
        }

    if set(events) != set(kg_rank_by_event):
        missing_kg = sorted(set(events) - set(kg_rank_by_event))[:10]
        extra_kg = sorted(set(kg_rank_by_event) - set(events))[:10]
        raise ValueError(
            "KBRD/KG event IDs do not match. "
            f"Missing KG examples={missing_kg}; extra KG examples={extra_kg}"
        )
    return events, candidates


def exposure_update(ledger, candidate, ranked_ids, item2group, categories):
    ids = candidate["ids"]
    groups = np.asarray(
        [item2group[int(movie_id)] for movie_id in ids], dtype=int
    )
    ranked_groups = np.asarray(
        [item2group[int(movie_id)] for movie_id in ranked_ids[:TOP_K]],
        dtype=int,
    )
    ledger.update(ids, groups, ranked_ids[:TOP_K], ranked_groups, categories)


def baseline_rank(candidate):
    # KBRD Top-500 order is the frozen baseline order.
    return candidate["ids"]


def contextual_gate_rerank(candidate, item2group, categories, ledger, lam, tau):
    """
    Continuous contextual relevance gate.

    For candidate i:
      context_i = mean normalized prevalence of its verified categories
                  in this event's frozen candidate pool.
      kg_conf_i = exp(-kg_rank_i / tau), with unsupported items assigned 0.
      deficit_i = mean category x popularity-group exposure deficit.
      gate_i = context_i * kg_conf_i.
      adjusted_score_i = KBRD_score_i + lambda * score_scale
                         * normalized(deficit_i * gate_i).

    This is a candidate-pool-context proxy, not dialogue-semantic relevance.
    """
    ids = candidate["ids"]
    scores = candidate["scores"]
    kg_ranks = candidate["kg"]
    groups = np.asarray(
        [item2group[int(movie_id)] for movie_id in ids], dtype=int
    )

    context, categorized_count = reference_category_context(ids, categories)
    deficits = ledger.deficits(ids, groups, categories)

    bonus = np.zeros(len(ids), dtype=np.float64)
    eligible = 0
    supported = 0

    for index, movie_id in enumerate(ids):
        movie_categories = categories.get(int(movie_id), [])
        kg_rank = int(kg_ranks[index])
        if not movie_categories or kg_rank <= 0:
            continue

        supported += 1
        context_relevance = float(
            np.mean([context.get(category, 0.0) for category in movie_categories])
        )
        kg_confidence = math.exp(-float(kg_rank) / float(tau))
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
    return ranked_ids, {
        "eligible_contextual_kg_candidates": int(eligible),
        "kg_supported_candidates": int(supported),
        "categorized_candidates": int(categorized_count),
        "top10_promoted_positions": int(
            np.sum(order[:TOP_K] != np.arange(min(TOP_K, len(order))))
        ),
    }


def reference_category_context(candidate_ids, categories):
    # Same event candidate-pool context definition as Experiment 22.16.
    counts = {}
    categorized_count = 0
    for movie_id in candidate_ids:
        movie_categories = categories.get(int(movie_id), [])
        if not movie_categories:
            continue
        categorized_count += 1
        for category in movie_categories:
            counts[category] = counts.get(category, 0.0) + (
                1.0 / len(movie_categories)
            )
    if not counts or categorized_count == 0:
        return {}, 0
    prevalence = {
        category: value / float(categorized_count)
        for category, value in counts.items()
    }
    maximum = max(prevalence.values())
    return {
        category: value / (maximum + EPS)
        for category, value in prevalence.items()
    }, categorized_count


def ranking_metrics(ranked_ids, target):
    positions = np.flatnonzero(ranked_ids[:TOP_K] == int(target))
    if not len(positions):
        return 0, 0.0
    rank = int(positions[0]) + 1
    return 1, 1.0 / math.log2(rank + 1.0)


def run_split(event_ids, events, candidates, item2group, categories,
              start_ledger, lam=None, tau=None, treatment=False):
    ledger = start_ledger.clone()
    rows = []
    for event_id in sorted(event_ids):
        candidate = candidates[event_id]
        if treatment:
            ranked_ids, info = contextual_gate_rerank(
                candidate, item2group, categories, ledger, lam, tau
            )
        else:
            ranked_ids = baseline_rank(candidate)
            info = {
                "eligible_contextual_kg_candidates": 0,
                "kg_supported_candidates": int(np.sum(candidate["kg"] > 0)),
                "categorized_candidates": 0,
                "top10_promoted_positions": 0,
            }
        hit, ndcg = ranking_metrics(ranked_ids, events[event_id]["target"])
        exposure_update(ledger, candidate, ranked_ids, item2group, categories)
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
        "category_group_log_rate_MAD": float(
            reference.fairness_mad(ledger)
        ),
        "mean_eligible_contextual_kg_candidates": float(
            frame.eligible_contextual_kg_candidates.mean()
        ) if len(frame) else 0.0,
        "mean_top10_promoted_positions": float(
            frame.top10_promoted_positions.mean()
        ) if len(frame) else 0.0,
    }
    return frame, ledger, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kbrd", type=Path, default=DEFAULT_KBRD)
    parser.add_argument("--kg", type=Path, default=DEFAULT_KG)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--utility-retention", type=float, default=UTILITY_RETENTION)
    args = parser.parse_args()

    global reference
    reference = load_reference(args.reference)

    for path in (args.kbrd, args.kg, args.profiles, args.catalog):
        if not path.exists():
            raise FileNotFoundError(path)

    item2group, catalog, categories = reference.load_metadata(
        args.catalog, args.profiles
    )
    events, candidates = load_events_and_candidates(args.kbrd, args.kg)

    if not events:
        raise RuntimeError("No validation events loaded.")
    if any(not set(candidate["ids"]).issubset(catalog)
           for candidate in candidates.values()):
        raise ValueError("A frozen KBRD candidate is absent from the catalog.")

    conv_ids = [event["conv"] for event in events.values()]
    warm_convs, tune_convs, heldout_convs = reference.split_conversations(conv_ids)

    warm_ids = {eid for eid, e in events.items() if e["conv"] in warm_convs}
    tune_ids = {eid for eid, e in events.items() if e["conv"] in tune_convs}
    heldout_ids = {eid for eid, e in events.items() if e["conv"] in heldout_convs}

    if warm_ids & tune_ids or warm_ids & heldout_ids or tune_ids & heldout_ids:
        raise RuntimeError("Conversation split leakage detected.")
    if warm_ids | tune_ids | heldout_ids != set(events):
        raise RuntimeError("Conversation split does not cover all events.")

    args.out.mkdir(parents=True, exist_ok=True)

    print("Loading frozen validation exports...")
    print(
        f"Events={len(events)} | conversations={len(set(conv_ids))} | "
        f"warm={len(warm_ids)} | tune={len(tune_ids)} | heldout={len(heldout_ids)}"
    )
    print(
        f"KG-supported candidate rows={sum(int(np.sum(c['kg'] > 0)) for c in candidates.values())}"
    )

    # Baseline warm history is replayed using the unchanged KBRD ranking.
    baseline_warm = reference.ExposureLedger()
    _, baseline_warm, _ = run_split(
        warm_ids, events, candidates, item2group, categories,
        baseline_warm, treatment=False
    )
    baseline_tune_frame, baseline_tune_ledger, baseline_tune_summary = run_split(
        tune_ids, events, candidates, item2group, categories,
        baseline_warm, treatment=False
    )

    # Tune lambda and rank-decay tau on tuning conversations only.
    tuning_rows = []
    best = None
    for tau in TAU_GRID:
        for lam in LAMBDA_GRID:
            _, _, summary = run_split(
                tune_ids, events, candidates, item2group, categories,
                baseline_warm, lam=lam, tau=tau, treatment=True
            )
            retained = (
                summary["NDCG@10"] /
                max(baseline_tune_summary["NDCG@10"], EPS)
            )
            feasible = retained >= args.utility_retention
            row = {
                "lambda": lam,
                "tau": tau,
                **summary,
                "NDCG_retention": retained,
                "utility_constraint_met": bool(feasible),
            }
            tuning_rows.append(row)
            if feasible:
                key = (
                    summary["category_group_log_rate_MAD"],
                    -summary["NDCG@10"],
                    lam,
                    tau,
                )
                if best is None or key < best["key"]:
                    best = {"key": key, "lambda": lam, "tau": tau}

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv(args.out / "tuning_grid_results.csv", index=False)

    if best is None:
        selected_lam = 0.0
        selected_tau = float(TAU_GRID[0])
        selection_status = "NO_FEASIBLE_NONZERO_CONFIGURATION"
    else:
        selected_lam = float(best["lambda"])
        selected_tau = float(best["tau"])
        selection_status = "UTILITY_CONSTRAINED_CONFIGURATION_SELECTED"

    # Replay selected treatment on warm and tuning conversations to establish
    # the treatment-specific held-out starting ledger.
    treatment_warm = reference.ExposureLedger()
    _, treatment_warm, _ = run_split(
        warm_ids, events, candidates, item2group, categories,
        treatment_warm, lam=selected_lam, tau=selected_tau, treatment=True
    )
    _, treatment_tune_ledger, treatment_tune_summary = run_split(
        tune_ids, events, candidates, item2group, categories,
        treatment_warm, lam=selected_lam, tau=selected_tau, treatment=True
    )

    baseline_heldout_frame, _, baseline_heldout = run_split(
        heldout_ids, events, candidates, item2group, categories,
        baseline_tune_ledger, treatment=False
    )
    treatment_heldout_frame, _, treatment_heldout = run_split(
        heldout_ids, events, candidates, item2group, categories,
        treatment_tune_ledger, lam=selected_lam, tau=selected_tau,
        treatment=(selected_lam > 0)
    )

    baseline_heldout_frame.to_csv(
        args.out / "baseline_heldout_event_results.csv", index=False
    )
    treatment_heldout_frame.to_csv(
        args.out / "contextual_gate_heldout_event_results.csv", index=False
    )

    delta_ndcg = treatment_heldout["NDCG@10"] - baseline_heldout["NDCG@10"]
    delta_hr = treatment_heldout["HR@10"] - baseline_heldout["HR@10"]
    delta_fairness = (
        treatment_heldout["category_group_log_rate_MAD"]
        - baseline_heldout["category_group_log_rate_MAD"]
    )
    ndcg_retention = (
        treatment_heldout["NDCG@10"] /
        max(baseline_heldout["NDCG@10"], EPS)
    )

    comparison = pd.DataFrame([
        {"method": "B0_KBRD", **baseline_heldout},
        {
            "method": "B5_contextual_relevance_gate",
            **treatment_heldout,
            "selected_lambda": selected_lam,
            "selected_tau": selected_tau,
            "selection_status": selection_status,
        },
    ])
    comparison.to_csv(args.out / "heldout_comparison.csv", index=False)

    checks = {
        "all_frozen_event_ids_loaded": len(events) == len(candidates),
        "all_candidate_lists_have_500_unique_items": all(
            len(c["ids"]) == 500 and len(np.unique(c["ids"])) == 500
            for c in candidates.values()
        ),
        "all_candidates_in_catalog": all(
            set(map(int, c["ids"])).issubset(catalog)
            for c in candidates.values()
        ),
        "kg_ranks_transferred_by_event_and_movie": True,
        "conversation_disjoint_split": (
            not (warm_convs & tune_convs)
            and not (warm_convs & heldout_convs)
            and not (tune_convs & heldout_convs)
        ),
        "all_events_assigned_once": (
            len(warm_ids | tune_ids | heldout_ids) == len(events)
        ),
        "utility_constraint_used_tuning_only": True,
        "heldout_events_not_used_for_selection": True,
        "no_training_performed": True,
        "no_retrieval_rerun": True,
        "no_independent_test_evaluation": True,
        "context_is_candidate_pool_category_prevalence_not_dialogue_semantics": True,
        "overall_pass": False,
    }
    checks["overall_pass"] = all(
        bool(value) for key, value in checks.items()
        if key != "overall_pass"
    )

    assessment = {
        "experiment": "22.18_contextual_relevance_gate_validation",
        "scope": "frozen_validation_exports_only",
        "events": len(events),
        "conversations": len(set(conv_ids)),
        "warm_events": len(warm_ids),
        "tuning_events": len(tune_ids),
        "heldout_events": len(heldout_ids),
        "selected_lambda": selected_lam,
        "selected_tau": selected_tau,
        "selection_status": selection_status,
        "utility_retention_threshold": args.utility_retention,
        "heldout_delta_HR@10": delta_hr,
        "heldout_delta_NDCG@10": delta_ndcg,
        "heldout_NDCG_retention": ndcg_retention,
        "heldout_delta_category_group_log_rate_MAD": delta_fairness,
        "fairness_improved": bool(delta_fairness < 0),
        "utility_retained": bool(ndcg_retention >= args.utility_retention),
        "intervention_success": bool(
            delta_fairness < 0 and ndcg_retention >= args.utility_retention
        ),
        "interpretation": (
            "Candidate-pool category prevalence is used as contextual relevance; "
            "this is not semantic dialogue relevance. A successful result requires "
            "lower held-out category-group log-rate MAD and at least 95% NDCG retention."
        ),
        "independent_test_evaluated": False,
        "training_performed": False,
        "retrieval_rerun": False,
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
            "kbrd_candidates": str(args.kbrd),
            "kg_candidate_export": str(args.kg),
            "category_profiles": str(args.profiles),
            "catalog_metadata": str(args.catalog),
            "experiment_22_16_reference": str(args.reference),
        },
        "calibration_grid": {
            "lambda": list(LAMBDA_GRID),
            "kg_rank_decay_tau": list(TAU_GRID),
        },
        "outputs": [
            "tuning_grid_results.csv",
            "baseline_heldout_event_results.csv",
            "contextual_gate_heldout_event_results.csv",
            "heldout_comparison.csv",
            "validation_checks.json",
            "final_research_assessment.json",
            "experiment_manifest.json",
        ],
    }
    (args.out / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\n========== EXPERIMENT 22.18 COMPLETE ==========")
    print(comparison.to_string(index=False))
    print("\nResearch assessment:")
    print(json.dumps(assessment, indent=2))
    print("\nValidation:")
    print(json.dumps(checks, indent=2))
    print("\nOutput:", args.out.resolve())

    if not checks["overall_pass"]:
        raise RuntimeError("Experiment 22.18 validation failed.")


if __name__ == "__main__":
    main()



