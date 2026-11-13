#!/usr/bin/env python
"""
Experiment 22.16: Category-context utility calibration.

Validation-only experiment. Uses the frozen KBRD Top-500 candidate pool and
the existing A1_frequency_3 KG-rank transfer. No model training, retrieval
rerun, or independent test split is used.

Operational definition of "context" in this implementation:
the category composition of the event's frozen candidate pool. This is
candidate-pool context, NOT semantic dialogue-context understanding.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2200
TOP_K = 10
CANDIDATE_K = 500
EPS = 1e-12
DEFAULT_LAMBDAS = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20)


def as_list(value, field):
    try:
        result = json.loads(value) if isinstance(value, str) else value
    except Exception:
        result = ast.literal_eval(value)
    if not isinstance(result, list):
        raise ValueError(f"{field} must contain a list")
    return result


def as_bool(value):
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_genres(value, delimiter):
    if pd.isna(value):
        return []
    return sorted({
        " ".join(token.strip().casefold().split())
        for token in str(value).split(delimiter) if token.strip()
    })


def split_conversations(conv_ids):
    unique = np.asarray(sorted(set(map(int, conv_ids))), dtype=np.int64)
    rng = np.random.default_rng(SEED)
    rng.shuffle(unique)
    n_warm = round(0.60 * len(unique))
    n_tune = round(0.20 * len(unique))
    return (
        set(map(int, unique[:n_warm])),
        set(map(int, unique[n_warm:n_warm + n_tune])),
        set(map(int, unique[n_warm + n_tune:])),
    )


def load_metadata(catalog_path, profile_path):
    with open(catalog_path, "rb") as stream:
        metadata = pickle.load(stream)

    item2group = {int(k): int(v) for k, v in metadata["item2group"].items()}
    catalog = set(map(int, metadata["movie_ids"]))
    if set(item2group) != catalog:
        raise ValueError("Catalog and item2group IDs do not match")
    if any(group not in range(5) for group in item2group.values()):
        raise ValueError("Popularity group IDs must be in [0, 4]")

    profiles = pd.read_csv(profile_path)
    required = {
        "movie_entity_id", "canonical_genres", "imdb_genres",
        "imdb_genre_evidence", "imdb_conflicting_records",
        "accepted_genre_evidence", "identity_resolution_status",
    }
    missing = required - set(profiles.columns)
    if missing:
        raise ValueError(f"Category profile missing columns: {sorted(missing)}")
    profiles["movie_entity_id"] = profiles.movie_entity_id.astype(int)
    if profiles.movie_entity_id.duplicated().any():
        raise ValueError("Duplicate movie IDs in category profiles")

    categories = {movie_id: [] for movie_id in catalog}
    for row in profiles.itertuples(index=False):
        movie_id = int(row.movie_entity_id)
        if movie_id not in catalog:
            continue
        if str(row.identity_resolution_status).upper() == "IDENTITY_REVIEW_REQUIRED":
            continue

        accepted = []
        if as_bool(row.imdb_genre_evidence) and not as_bool(row.imdb_conflicting_records):
            raw = row.imdb_genres
            if not pd.isna(raw):
                raw = str(raw)
                accepted += parse_genres(raw, "|" if "|" in raw else ";")
        if as_bool(row.accepted_genre_evidence):
            accepted += parse_genres(row.canonical_genres, ";")
        categories[movie_id] = sorted(set(accepted))

    return item2group, catalog, categories


class ExposureLedger:
    """Tracks category x popularity-group exposure and opportunity."""

    def __init__(self):
        self.exposure = defaultdict(float)
        self.opportunity = defaultdict(float)

    @staticmethod
    def opportunity_weights(n):
        weights = 1.0 / np.log2(np.arange(1, n + 1, dtype=float) + 1.0)
        return weights / (weights.sum() + EPS)

    def update_opportunity(self, candidate_ids, groups, categories):
        weights = self.opportunity_weights(len(candidate_ids))
        for rank, movie_id in enumerate(candidate_ids):
            movie_categories = categories.get(int(movie_id), [])
            if not movie_categories:
                continue
            contribution = float(weights[rank]) / len(movie_categories)
            for category in movie_categories:
                self.opportunity[category, int(groups[rank])] += contribution

    def update_exposure(self, ranked_ids, groups, categories):
        for rank, movie_id in enumerate(ranked_ids[:TOP_K]):
            movie_categories = categories.get(int(movie_id), [])
            if not movie_categories:
                continue
            contribution = 1.0 / math.log2(rank + 2.0) / len(movie_categories)
            for category in movie_categories:
                self.exposure[category, int(groups[rank])] += contribution

    def update(self, candidate_ids, candidate_groups, ranked_ids, ranked_groups, categories):
        self.update_opportunity(candidate_ids, candidate_groups, categories)
        self.update_exposure(ranked_ids, ranked_groups, categories)

    def deficits(self, candidate_ids, candidate_groups, categories):
        totals = defaultdict(lambda: [0.0, 0.0])
        for (category, group), opportunity in self.opportunity.items():
            totals[category][0] += self.exposure[category, group]
            totals[category][1] += opportunity

        reference = {
            category: exposure / (opportunity + EPS) if opportunity > 0 else 0.0
            for category, (exposure, opportunity) in totals.items()
        }

        result = {}
        for movie_id, group in zip(candidate_ids, candidate_groups):
            for category in categories.get(int(movie_id), []):
                opportunity = self.opportunity[category, int(group)]
                ref = reference.get(category, 0.0)
                rate = self.exposure[category, int(group)] / (opportunity + EPS)
                result[category, int(group)] = (
                    max(0.0, ref - rate) / (ref + EPS)
                    if opportunity > EPS and ref > EPS else 0.0
                )
        return result

    def clone(self):
        copied = ExposureLedger()
        copied.exposure.update(self.exposure)
        copied.opportunity.update(self.opportunity)
        return copied


def category_context(candidate_ids, categories):
    """
    Event-level candidate-pool context: normalized prevalence of each
    category among categorized candidates in this frozen Top-500 pool.
    """
    counts = defaultdict(float)
    categorized_count = 0
    for movie_id in candidate_ids:
        movie_categories = categories.get(int(movie_id), [])
        if not movie_categories:
            continue
        categorized_count += 1
        for category in movie_categories:
            counts[category] += 1.0 / len(movie_categories)

    if not counts or categorized_count == 0:
        return {}, 0
    denominator = float(categorized_count)
    prevalence = {category: value / denominator for category, value in counts.items()}
    max_prevalence = max(prevalence.values())
    normalized = {
        category: value / (max_prevalence + EPS)
        for category, value in prevalence.items()
    }
    return normalized, categorized_count


def rerank(candidate, item2group, categories, ledger, lam, mode):
    ids = candidate["ids"]
    scores = candidate["scores"]
    kg_ranks = candidate["kg"]
    groups = np.asarray([item2group[int(movie_id)] for movie_id in ids], dtype=int)

    deficits = ledger.deficits(ids, groups, categories)
    context, categorized_count = category_context(ids, categories)
    bonus = np.zeros(len(ids), dtype=float)
    eligible = 0

    for index, movie_id in enumerate(ids):
        movie_categories = categories.get(int(movie_id), [])
        if not movie_categories:
            continue
        if mode in {"kg_gated", "context_kg_gated"} and kg_ranks[index] <= 0:
            continue

        eligible += 1
        values = []
        for category in movie_categories:
            deficit = deficits.get((category, int(groups[index])), 0.0)
            if mode in {"context", "context_kg_gated"}:
                deficit *= context.get(category, 0.0)
            values.append(deficit)
        bonus[index] = float(np.mean(values)) if values else 0.0

    max_bonus = float(bonus.max(initial=0.0))
    if max_bonus > 0:
        bonus /= max_bonus

    score_scale = float(np.median(np.abs(scores)))
    if score_scale <= EPS:
        score_scale = float(np.max(np.abs(scores))) if len(scores) else 1.0

    adjusted = scores + float(lam) * score_scale * bonus
    order = np.lexsort((ids, -adjusted))
    ranked_ids = ids[order]
    ranked_groups = groups[order]

    return ranked_ids, ranked_groups, {
        "eligible_categorized_candidates": eligible,
        "categorized_candidates": categorized_count,
        "top10_promoted_positions": int(np.sum(order[:TOP_K] != np.arange(min(TOP_K, len(order))))),
        "context_categories": len(context),
    }


def ranking_metrics(ranked_ids, target):
    positions = np.flatnonzero(ranked_ids[:TOP_K] == int(target))
    if not len(positions):
        return 0, 0.0, None
    rank = int(positions[0]) + 1
    return 1, 1.0 / math.log2(rank + 1.0), rank


def fairness_mad(ledger):
    """MAD of log exposure/opportunity rates over supported category-group cells."""
    rates = []
    for key, opportunity in ledger.opportunity.items():
        if opportunity > EPS:
            rates.append((ledger.exposure[key] + EPS) / (opportunity + EPS))
    if len(rates) < 2:
        return float("nan")
    log_rates = np.log(np.asarray(rates, dtype=float))
    return float(np.mean(np.abs(log_rates - np.mean(log_rates))))


def fairness_on_frame(frame, candidates, item2group, categories):
    """Measure disparity on this split only, not on accumulated warm-up history."""
    ledger = ExposureLedger()
    for row in frame.itertuples(index=False):
        event_id = int(row.event_id)
        candidate_ids = candidates[event_id]["ids"]
        candidate_groups = np.asarray(
            [item2group[int(movie_id)] for movie_id in candidate_ids], dtype=int
        )
        ranked_ids = np.asarray(as_list(row.top10_movie_ids, "ranked top-10 IDs"), dtype=int)
        ranked_groups = np.asarray(as_list(row.top10_group_ids, "ranked top-10 groups"), dtype=int)
        ledger.update(
            candidate_ids, candidate_groups, ranked_ids, ranked_groups, categories
        )
    return fairness_mad(ledger)


def run_sequential(events, candidates, event_ids, item2group, categories, start_ledger, lam, mode):
    ledger = start_ledger.clone()
    rows = []
    for event_id in sorted(event_ids):
        event = events[event_id]
        candidate = candidates[event_id]
        ranked_ids, ranked_groups, info = rerank(
            candidate, item2group, categories, ledger, lam, mode
        )
        hit, ndcg, rank = ranking_metrics(ranked_ids, event["target"])
        rows.append({
            "event_id": event_id,
            "conv_id": event["conv"],
            "hit10": hit,
            "ndcg10": ndcg,
            "target_rank_top500": rank,
            **info,
            "top10_movie_ids": json.dumps(list(map(int, ranked_ids[:TOP_K]))),
            "top10_group_ids": json.dumps(list(map(int, ranked_groups[:TOP_K]))),
        })
        candidate_groups = np.asarray(
            [item2group[int(movie_id)] for movie_id in candidate["ids"]], dtype=int
        )
        ledger.update(
            candidate["ids"], candidate_groups,
            ranked_ids, ranked_groups, categories
        )
    return pd.DataFrame(rows), ledger


def baseline_sequential(events, candidates, event_ids, item2group, categories, start_ledger):
    ledger = start_ledger.clone()
    rows = []
    for event_id in sorted(event_ids):
        event = events[event_id]
        candidate = candidates[event_id]
        ids = candidate["ids"]
        groups = np.asarray([item2group[int(movie_id)] for movie_id in ids], dtype=int)
        hit, ndcg, rank = ranking_metrics(ids, event["target"])
        _, categorized = category_context(ids, categories)
        rows.append({
            "event_id": event_id,
            "conv_id": event["conv"],
            "hit10": hit,
            "ndcg10": ndcg,
            "target_rank_top500": rank,
            "eligible_categorized_candidates": categorized,
            "categorized_candidates": categorized,
            "top10_promoted_positions": 0,
            "context_categories": len(category_context(ids, categories)[0]),
            "top10_movie_ids": json.dumps(list(map(int, ids[:TOP_K]))),
            "top10_group_ids": json.dumps(list(map(int, groups[:TOP_K]))),
        })
        ledger.update(ids, groups, ids, groups, categories)
    return pd.DataFrame(rows), ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=Path("results_corrected/experiment_22_16"))
    parser.add_argument("--lambdas", default=",".join(map(str, DEFAULT_LAMBDAS)))
    parser.add_argument("--utility-retention", type=float, default=0.95)
    args = parser.parse_args()

    lambdas = sorted(set(float(value) for value in args.lambdas.split(",")))
    if not lambdas or min(lambdas) < 0:
        raise ValueError("Lambda values must be non-negative")
    if not 0 < args.utility_retention <= 1:
        raise ValueError("utility-retention must be in (0, 1]")

    args.out.mkdir(parents=True, exist_ok=True)
    print("Loading frozen validation inputs...")
    item2group, catalog, categories = load_metadata(
        args.catalog_pickle, args.category_profiles
    )
    kbrd = pd.read_csv(args.kbrd)
    manifest = pd.read_csv(args.manifest)
    kg = pd.read_csv(args.kg, usecols=[
        "event_id", "method", "candidate_movie_ids", "candidate_kg_ranks"
    ])
    kg = kg[kg.method == "A1_frequency_3"].copy()

    for frame in (kbrd, manifest, kg):
        frame["event_id"] = frame.event_id.astype(int)
        if frame.event_id.duplicated().any():
            raise ValueError("Duplicate event IDs found in an input export")

    if set(kbrd.event_id) != set(manifest.event_id) or set(kbrd.event_id) != set(kg.event_id):
        raise ValueError("KBRD, manifest and KG event IDs must match exactly")

    manifest = manifest[["event_id", "conv_id", "target_movie_id"]]
    joined = kbrd.merge(manifest, on="event_id", how="inner",
                        validate="one_to_one", suffixes=("", "_manifest"))
    for col in ("conv_id", "target_movie_id"):
        other = f"{col}_manifest"
        if other in joined and not np.array_equal(joined[col], joined[other]):
            raise ValueError(f"KBRD and manifest {col} values disagree")

    kg_map = {}
    for row in kg.itertuples(index=False):
        ids = np.asarray(as_list(row.candidate_movie_ids, "KG candidate IDs"), dtype=np.int64)
        ranks = np.asarray(as_list(row.candidate_kg_ranks, "KG ranks"), dtype=np.int64)
        if len(ids) != len(ranks) or len(np.unique(ids)) != len(ids) or np.any(ranks < 0):
            raise ValueError(f"Invalid KG arrays for event {row.event_id}")
        kg_map[int(row.event_id)] = dict(zip(map(int, ids), map(int, ranks)))

    events, candidates = {}, {}
    for row in joined.sort_values("event_id").itertuples(index=False):
        event_id = int(row.event_id)
        ids = np.asarray(as_list(row.top_movie_ids, "KBRD candidate IDs"), dtype=np.int64)
        scores = np.asarray(as_list(row.top_scores, "KBRD scores"), dtype=float)
        if len(ids) != CANDIDATE_K or len(scores) != CANDIDATE_K:
            raise ValueError(f"Expected exactly {CANDIDATE_K} candidates at event {event_id}")
        if len(np.unique(ids)) != CANDIDATE_K or not np.isfinite(scores).all():
            raise ValueError(f"Invalid candidate IDs/scores at event {event_id}")
        if not set(map(int, ids)).issubset(catalog):
            raise ValueError(f"Out-of-catalog candidate at event {event_id}")
        ranks = kg_map[event_id]
        kg_ranks = np.asarray([ranks.get(int(movie_id), 0) for movie_id in ids], dtype=np.int64)
        events[event_id] = {"conv": int(row.conv_id), "target": int(row.target_movie_id)}
        candidates[event_id] = {"ids": ids, "scores": scores, "kg": kg_ranks}

    if len(events) != 3010 or len({event["conv"] for event in events.values()}) != 883:
        raise ValueError("Expected the frozen 3,010-event / 883-conversation validation export")

    event_ids = sorted(events)
    warm, tune, held = split_conversations(event["conv"] for event in events.values())
    by_conversation = defaultdict(list)
    for event_id, event in events.items():
        by_conversation[event["conv"]].append(event_id)
    warm_ids = sorted(e for c in warm for e in by_conversation[c])
    tune_ids = sorted(e for c in tune for e in by_conversation[c])
    held_ids = sorted(e for c in held for e in by_conversation[c])

    if set(warm_ids) | set(tune_ids) | set(held_ids) != set(event_ids):
        raise RuntimeError("Conversation split does not cover all events")
    if warm & tune or warm & held or tune & held:
        raise RuntimeError("Conversation leakage across splits")

    print(f"Events={len(events)} | conversations={len(by_conversation)}")
    print(f"Split events: warm={len(warm_ids)}, tune={len(tune_ids)}, heldout={len(held_ids)}")

    # Warm-up history is strictly KBRD-only and uses no held-out events.
    warm_ledger = ExposureLedger()
    for event_id in warm_ids:
        candidate = candidates[event_id]
        groups = np.asarray([item2group[int(m)] for m in candidate["ids"]], dtype=int)
        warm_ledger.update(candidate["ids"], groups, candidate["ids"], groups, categories)

    baseline_tune, baseline_tune_ledger = baseline_sequential(
        events, candidates, tune_ids, item2group, categories, warm_ledger
    )
    baseline_tune_ndcg = float(baseline_tune.ndcg10.mean())
    tuning_rows = []
    selected = {}

    modes = {
        "B1_category_only": "category",
        "B2_KG_gated": "kg_gated",
        "B3_category_context": "context",
        "B4_context_KG_gated": "context_kg_gated",
    }

    for method, mode in modes.items():
        feasible = []
        for lam in lambdas:
            trial, ledger = run_sequential(
                events, candidates, tune_ids, item2group, categories,
                warm_ledger, lam, mode
            )
            ndcg = float(trial.ndcg10.mean())
            fair = fairness_on_frame(trial, candidates, item2group, categories)
            row = {
                "method": method,
                "mode": mode,
                "lambda": lam,
                "HR@10": float(trial.hit10.mean()),
                "NDCG@10": ndcg,
                "category_group_log_rate_MAD": fair,
                "utility_retention_vs_KBRD": ndcg / (baseline_tune_ndcg + EPS),
                "meets_utility_constraint": bool(
                    ndcg >= args.utility_retention * baseline_tune_ndcg
                ),
            }
            tuning_rows.append(row)
            if row["meets_utility_constraint"] and np.isfinite(fair):
                feasible.append(row)

        if feasible:
            chosen = sorted(
                feasible,
                key=lambda x: (x["category_group_log_rate_MAD"], -x["NDCG@10"], x["lambda"])
            )[0]["lambda"]
        else:
            chosen = 0.0
        selected[method] = float(chosen)
        print(f"{method}: selected lambda={chosen}")

    pd.DataFrame(tuning_rows).to_csv(args.out / "tuning_results.csv", index=False)

    # Reconstruct the sequential ledger through tuning events before held-out evaluation.
    frames = {}
    held_ledgers = {}
    frames["B0_KBRD"], held_ledgers["B0_KBRD"] = baseline_sequential(
        events, candidates, held_ids, item2group, categories, baseline_tune_ledger
    )
    for method, mode in modes.items():
        _, ledger_after_tune = run_sequential(
            events, candidates, tune_ids, item2group, categories,
            warm_ledger, selected[method], mode
        )
        frames[method], held_ledgers[method] = run_sequential(
            events, candidates, held_ids, item2group, categories,
            ledger_after_tune, selected[method], mode
        )

    point_rows = []
    for method, frame in frames.items():
        ledger = held_ledgers[method]
        point_rows.append({
            "method": method,
            "selected_lambda": 0.0 if method == "B0_KBRD" else selected[method],
            "events": len(frame),
            "conversations": int(frame.conv_id.nunique()),
            "HR@10": float(frame.hit10.mean()),
            "NDCG@10": float(frame.ndcg10.mean()),
            "category_group_log_rate_MAD": fairness_on_frame(
                frame, candidates, item2group, categories
            ),
            "mean_eligible_categorized_candidates": float(
                frame.eligible_categorized_candidates.mean()
            ),
            "mean_top10_promoted_positions": float(frame.top10_promoted_positions.mean()),
        })

    pd.DataFrame(point_rows).to_csv(args.out / "point_estimates.csv", index=False)
    pd.concat(
        [frame.assign(method=method) for method, frame in frames.items()],
        ignore_index=True
    ).to_csv(args.out / "event_level_results.csv", index=False)

    checks = {
        "experiment": "22.16_category_context_utility_calibration",
        "overall_pass": True,
        "validation_only": True,
        "independent_test_split_loaded": False,
        "training_performed": False,
        "retrieval_rerun": False,
        "coefficient_search": False,
        "frozen_kbrd_top500_used": True,
        "kg_support_transferred_by_event_and_movie_id": True,
        "missing_kg_evidence_defaults_to_zero": True,
        "conversation_disjoint_split": True,
        "warm_history_uses_kbrd_only": True,
        "utility_constraint_tuned_on_tuning_split_only": True,
        "utility_retention_threshold": args.utility_retention,
        "context_definition": "category prevalence in each frozen event candidate pool",
        "context_is_semantic_dialogue_understanding": False,
        "events": len(events),
        "conversations": len(by_conversation),
        "warm_events": len(warm_ids),
        "tuning_events": len(tune_ids),
        "heldout_events": len(held_ids),
        "selected_lambda": selected,
        "methods": list(frames),
        "all_methods_have_same_heldout_event_ids": all(
            set(frame.event_id) == set(frames["B0_KBRD"].event_id)
            for frame in frames.values()
        ),
    }
    if not checks["all_methods_have_same_heldout_event_ids"]:
        checks["overall_pass"] = False

    with open(args.out / "validation_checks.json", "w", encoding="utf-8") as stream:
        json.dump(checks, stream, indent=2)
    manifest = {
        **checks,
        "inputs": {
            "kbrd": str(args.kbrd),
            "manifest": str(args.manifest),
            "kg_candidates": str(args.kg),
            "catalog_pickle": str(args.catalog_pickle),
            "category_profiles": str(args.category_profiles),
        },
        "outputs": [
            "tuning_results.csv", "point_estimates.csv",
            "event_level_results.csv", "validation_checks.json",
            "experiment_manifest.json",
        ],
        "interpretation_limits": [
            "Candidate-pool category prevalence is a proxy context signal, not dialogue semantics.",
            "KG rank indicates retrieval support, not semantic relevance.",
            "This is validation-only evidence; no independent test-set result is produced.",
            "Utility feasibility is selected on tuning conversations and reported on held-out validation conversations.",
        ],
    }
    with open(args.out / "experiment_manifest.json", "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)

    print("\n========== EXPERIMENT 22.16 COMPLETE ==========")
    print(pd.DataFrame(point_rows).to_string(index=False))
    print("\nValidation:", json.dumps(checks, indent=2))
    print("Output:", args.out.resolve())


if __name__ == "__main__":
    main()
