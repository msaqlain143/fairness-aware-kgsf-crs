#!/usr/bin/env python
"""
Experiment 22.0: KG-FairCRS validation.

Uses the existing validation-only adaptive-M2 candidate export.
No model inference, checkpoint modification, or test-set access is performed.

Design:
- A1_frequency_3 is the primary M2 backbone.
- Category assignments reproduce the acceptance logic in Experiment 21.23B.
- Popularity groups come directly from catalog_audit_corrected.pkl["item2group"].
- KG rank > 0 is used only as a contextual support signal (not as proof of relevance).
- A conversation-level 60/20/20 split separates exposure warm-up, lambda selection,
  and final held-out validation.
- The exposure/relevance ledger is chronological and is updated after each event.
- The current event's candidates are never included in its own historical deficit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_BASE = "A0_fixed_025"
METHOD_M2 = "A1_frequency_3"
DEFAULT_LAMBDAS = [0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.20]
TOP_K = 10
EPS = 1e-12
SEED = 2200


def parse_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_genres(value, delimiter: str) -> list[str]:
    if pd.isna(value):
        return []
    return sorted({
        " ".join(str(x).strip().casefold().split())
        for x in str(value).split(delimiter)
        if str(x).strip()
    })


def gini(values) -> float:
    x = np.asarray(list(values), dtype=float)
    if x.size == 0 or np.any(x < 0):
        return 0.0
    total = x.sum()
    if total <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    return float((2 * np.dot(np.arange(1, n + 1), x) /
                  (n * total)) - (n + 1) / n)


def read_json_list(value, field: str) -> list:
    try:
        result = json.loads(value)
    except Exception as exc:
        raise ValueError(f"Cannot parse JSON list in {field}: {exc}") from exc
    if not isinstance(result, list):
        raise ValueError(f"{field} must contain a JSON list.")
    return result


def conversation_split(conv_ids: list[int], seed: int = SEED):
    """Deterministic split by conversation, never by individual event."""
    unique = sorted(set(map(int, conv_ids)))
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(unique, dtype=np.int64)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_warm = int(round(0.60 * n))
    n_tune = int(round(0.20 * n))
    warm = set(map(int, shuffled[:n_warm]))
    tune = set(map(int, shuffled[n_warm:n_warm + n_tune]))
    held = set(map(int, shuffled[n_warm + n_tune:]))
    return warm, tune, held


def load_categories(profile_path: Path, catalog_ids: set[int]):
    """
    Reproduce Experiment 21.23B:
      * ambiguous identity => withhold all category assignment
      * IMDb genres accepted only when imdb_genre_evidence and no conflict
      * DBpedia canonical_genres accepted only when accepted_genre_evidence
      * hybrid-only/unvalidated-only evidence is not independently accepted
    """
    profiles = pd.read_csv(profile_path)
    profiles["movie_entity_id"] = profiles["movie_entity_id"].astype(int)
    if profiles["movie_entity_id"].duplicated().any():
        raise ValueError("Duplicate movie_entity_id in category profiles.")
    if set(profiles["movie_entity_id"]) != catalog_ids:
        missing = len(catalog_ids - set(profiles["movie_entity_id"]))
        extra = len(set(profiles["movie_entity_id"]) - catalog_ids)
        raise ValueError(
            f"Category profile/catalog mismatch: missing={missing}, extra={extra}"
        )

    item_categories: dict[int, list[str]] = {}
    for row in profiles.to_dict("records"):
        movie_id = int(row["movie_entity_id"])
        if str(row.get("identity_resolution_status", "")) == "AMBIGUOUS_REQUIRES_REVIEW":
            item_categories[movie_id] = []
            continue

        cats = set()
        imdb_conflict = parse_bool(row.get("imdb_conflicting_records", False))
        imdb_valid = parse_bool(row.get("imdb_genre_evidence", False)) and not imdb_conflict
        dbpedia_valid = parse_bool(row.get("accepted_genre_evidence", False))

        if imdb_valid:
            cats.update(parse_genres(row.get("imdb_genres"), "|"))
        if dbpedia_valid:
            cats.update(parse_genres(row.get("canonical_genres"), ";"))

        item_categories[movie_id] = sorted(cats)

    return item_categories, profiles


def load_group_mapping(pickle_path: Path):
    with pickle_path.open("rb") as f:
        meta = pickle.load(f)
    required = {"movie_ids", "freq_dict", "item2group", "group_meta"}
    if not isinstance(meta, dict) or not required.issubset(meta):
        raise ValueError("Catalog pickle is missing required metadata keys.")
    item2group = {int(k): int(v) for k, v in meta["item2group"].items()}
    freq_dict = {int(k): int(v) for k, v in meta["freq_dict"].items()}
    movie_ids = set(map(int, meta["movie_ids"]))
    if set(item2group) != movie_ids or set(freq_dict) != movie_ids:
        raise ValueError("Popularity mapping does not cover the exact catalog.")
    groups = set(item2group.values())
    if groups != {0, 1, 2, 3, 4}:
        raise ValueError(f"Expected five groups 0..4, found {groups}.")
    return meta, item2group, freq_dict, movie_ids


def load_inputs(args):
    meta, item2group, freq_dict, catalog_ids = load_group_mapping(args.catalog_pickle)
    categories, profiles = load_categories(args.category_profiles, catalog_ids)

    events = pd.read_csv(args.events)
    mapping = pd.read_csv(args.mapping)
    candidate_columns = [
        "event_id", "method", "candidate_k", "candidate_movie_ids",
        "candidate_scores", "candidate_kg_ranks"
    ]
    # Stream the large 343 MB export and retain only the primary M2 method.
    candidate_parts = []
    for chunk in pd.read_csv(
        args.candidates, usecols=candidate_columns, chunksize=256
    ):
        primary = chunk[chunk["method"] == METHOD_M2]
        if not primary.empty:
            candidate_parts.append(primary)
    if not candidate_parts:
        raise ValueError(f"No rows found for primary method {METHOD_M2}.")
    c = pd.concat(candidate_parts, ignore_index=True)

    required_event = {
        "event_id", "method", "target_movie_id", "target_frequency",
        "frequency_group", "context_nonempty", "rank_top50", "hit10",
        "hit50", "ndcg10", "top10_movie_ids"
    }
    if not required_event.issubset(events.columns):
        raise ValueError(f"Event CSV missing: {required_event - set(events.columns)}")
    if not {"event_id", "conv_id", "target_movie_id"}.issubset(mapping.columns):
        raise ValueError("Conversation mapping lacks required columns.")

    mapping["event_id"] = mapping["event_id"].astype(int)
    mapping["conv_id"] = mapping["conv_id"].astype(int)
    if mapping["event_id"].duplicated().any():
        raise ValueError("Duplicate event_id in conversation mapping.")

    # Use only the primary M2 method for candidate-level reranking.
    e = events[events["method"] == METHOD_M2].copy()
    if c["event_id"].duplicated().any() or e["event_id"].duplicated().any():
        raise ValueError("Primary M2 data has duplicate event IDs.")
    if set(c.event_id.astype(int)) != set(e.event_id.astype(int)):
        raise ValueError("Candidate/event IDs do not match.")

    e = e.merge(
        mapping[["event_id", "conv_id", "target_movie_id"]].rename(
            columns={"target_movie_id": "mapping_target_movie_id"}
        ),
        on="event_id", how="left", validate="one_to_one"
    )
    if e["conv_id"].isna().any():
        raise ValueError("Some validation events lack conversation IDs.")
    if not np.array_equal(
        e["target_movie_id"].astype(int).to_numpy(),
        e["mapping_target_movie_id"].astype(int).to_numpy()
    ):
        raise ValueError("Target movie mismatch in conversation mapping.")
    e = e.drop(columns=["mapping_target_movie_id"])
    e = e.sort_values("event_id").reset_index(drop=True)
    c = c.sort_values("event_id").reset_index(drop=True)

    records = []
    for row in c.itertuples(index=False):
        ids = np.asarray(read_json_list(row.candidate_movie_ids, "candidate_movie_ids"), dtype=np.int64)
        scores = np.asarray(read_json_list(row.candidate_scores, "candidate_scores"), dtype=float)
        kg = np.asarray(read_json_list(row.candidate_kg_ranks, "candidate_kg_ranks"), dtype=np.int64)
        if not (len(ids) == len(scores) == len(kg) == int(row.candidate_k)):
            raise ValueError(f"Candidate arrays misaligned at event {row.event_id}.")
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"Duplicate candidate IDs at event {row.event_id}.")
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite candidate score at event {row.event_id}.")
        unknown = set(map(int, ids)) - catalog_ids
        if unknown:
            raise ValueError(f"Candidate IDs outside catalog: {list(unknown)[:5]}")
        records.append({
            "event_id": int(row.event_id),
            "candidate_ids": ids,
            "scores": scores,
            "kg_ranks": kg,
        })

    cand_by_event = {r["event_id"]: r for r in records}
    e_by_event = {int(r.event_id): r._asdict() for r in e.itertuples(index=False)}
    if set(cand_by_event) != set(e_by_event):
        raise ValueError("Candidate/event alignment failure.")

    # Ensure candidate export top ten exactly reproduces the existing M2 Top-10.
    for event_id, cr in cand_by_event.items():
        er = e_by_event[event_id]
        original_top10 = list(map(int, read_json_list(er["top10_movie_ids"], "top10_movie_ids")))
        if list(map(int, cr["candidate_ids"][:10])) != original_top10:
            raise ValueError(f"Top-10 integrity mismatch at event {event_id}.")

    return {
        "events": e,
        "all_events": events,
        "event_by_id": e_by_event,
        "candidate_by_event": cand_by_event,
        "mapping": mapping,
        "item_categories": categories,
        "profiles": profiles,
        "item2group": item2group,
        "freq_dict": freq_dict,
        "catalog_ids": catalog_ids,
        "meta": meta,
    }


class ExposureLedger:
    """Historical category x popularity-group exposure/opportunity ledger."""

    def __init__(self):
        self.exposure = defaultdict(float)
        self.opportunity = defaultdict(float)

    def deficits(self, candidate_ids, groups, item_categories):
        # Reference ratio for each category is its overall exposure/opportunity.
        cat_totals = defaultdict(lambda: [0.0, 0.0])
        for (cat, group), opp in self.opportunity.items():
            cat_totals[cat][0] += self.exposure[(cat, group)]
            cat_totals[cat][1] += opp

        ref = {}
        for cat, (exp, opp) in cat_totals.items():
            ref[cat] = exp / (opp + EPS) if opp > 0 else 0.0

        result = {}
        candidate_cells = set()
        for movie_id, group in zip(candidate_ids, groups):
            for cat in item_categories.get(int(movie_id), []):
                candidate_cells.add((cat, int(group)))
        for cat, group in candidate_cells:
            opp = self.opportunity[(cat, group)]
            if opp <= EPS or ref.get(cat, 0.0) <= EPS:
                result[(cat, group)] = 0.0
                continue
            rate = self.exposure[(cat, group)] / (opp + EPS)
            result[(cat, group)] = max(0.0, ref[cat] - rate) / (ref[cat] + EPS)
        return result

    @staticmethod
    def opportunity_weights(n: int):
        # Rank-biased predicted relevance mass over the existing M2 candidate list.
        ranks = np.arange(1, n + 1, dtype=float)
        w = 1.0 / np.log2(ranks + 1.0)
        return w / (w.sum() + EPS)

    @staticmethod
    def position_weights(k: int):
        ranks = np.arange(1, k + 1, dtype=float)
        return 1.0 / np.log2(ranks + 1.0)

    def update(self, ids, categories, groups, top_k=TOP_K):
        n = len(ids)
        opp_w = self.opportunity_weights(n)
        for idx, movie_id in enumerate(ids):
            cats = categories.get(int(movie_id), [])
            if not cats:
                continue
            frac = 1.0 / len(cats)
            for cat in cats:
                self.opportunity[(cat, int(groups[idx]))] += float(opp_w[idx]) * frac

        exp_w = self.position_weights(min(top_k, n))
        for pos, movie_id in enumerate(ids[:top_k]):
            cats = categories.get(int(movie_id), [])
            if not cats:
                continue
            frac = 1.0 / len(cats)
            for cat in cats:
                self.exposure[(cat, int(groups[pos]))] += float(exp_w[pos]) * frac


def rank_event(candidate, data, ledger, lam, use_kg_gate):
    ids = candidate["candidate_ids"]
    scores = candidate["scores"]
    kg = candidate["kg_ranks"]
    groups = np.asarray([data["item2group"][int(i)] for i in ids], dtype=int)
    cats_map = data["item_categories"]

    # Historical deficits are computed before the current event is added.
    deficits = ledger.deficits(ids, groups, cats_map)
    category_bonus = np.zeros(len(ids), dtype=float)
    eligible_count = 0
    categorized_count = 0

    for j, movie_id in enumerate(ids):
        cats = cats_map.get(int(movie_id), [])
        if not cats:
            continue
        categorized_count += 1
        if use_kg_gate and kg[j] <= 0:
            continue
        eligible_count += 1
        frac = 1.0 / len(cats)
        category_bonus[j] = sum(
            frac * deficits.get((cat, int(groups[j])), 0.0)
            for cat in cats
        )

    if category_bonus.max(initial=0.0) > 0:
        category_bonus /= category_bonus.max()

    # Scale adjustment relative to the event's score magnitude.
    score_scale = float(np.median(np.abs(scores)))
    if score_scale <= EPS:
        score_scale = float(np.max(np.abs(scores))) if len(scores) else 1.0
    adjusted = scores + (lam * score_scale * category_bonus)

    # Deterministic tie-breaking by movie entity ID.
    order = np.lexsort((ids, -adjusted))
    ranked_ids = ids[order]
    ranked_groups = groups[order]
    ranked_scores = adjusted[order]
    ranked_kg = kg[order]

    return {
        "movie_ids": ranked_ids,
        "groups": ranked_groups,
        "scores": ranked_scores,
        "kg_ranks": ranked_kg,
        "eligible_count": eligible_count,
        "categorized_count": categorized_count,
        "promoted_count": int(np.sum(order[:TOP_K] != np.arange(TOP_K))),
        "bonus": category_bonus,
    }


def event_relevance(ranked_ids, target):
    pos = np.flatnonzero(ranked_ids[:TOP_K] == int(target))
    if len(pos) == 0:
        return 0, 0.0, None
    rank = int(pos[0]) + 1
    return 1, 1.0 / math.log2(rank + 1.0), rank


def category_disparity(exposure, opportunity):
    """
    Mean within-category absolute deviation of log exposure/opportunity rates
    across popularity groups with opportunity support.
    Lower is more balanced. Categories with fewer than two supported groups
    are excluded from this disparity statistic.
    """
    cats = sorted({c for c, g in opportunity if opportunity[(c, g)] > EPS})
    values = []
    rows = []
    for cat in cats:
        rates = []
        for group in range(5):
            opp = opportunity.get((cat, group), 0.0)
            if opp > EPS:
                rate = exposure.get((cat, group), 0.0) / (opp + EPS)
                rates.append((group, rate, opp, exposure.get((cat, group), 0.0)))
        if len(rates) < 2:
            continue
        logs = np.log(np.asarray([max(r[1], EPS) for r in rates]))
        center = float(np.mean(logs))
        mad = float(np.mean(np.abs(logs - center)))
        values.append(mad)
        for group, rate, opp, exp in rates:
            rows.append({
                "category": cat,
                "popularity_group": group,
                "exposure": exp,
                "opportunity_mass": opp,
                "exposure_per_opportunity": rate,
                "category_log_rate_mad": mad,
            })
    return (float(np.mean(values)) if values else float("nan")), rows


def evaluate_static_baselines(data, event_ids):
    rows = []
    for event_id in event_ids:
        er = data["event_by_id"][event_id]
        for method in [METHOD_BASE, METHOD_M2]:
            subset = data["all_events"]
            match = subset[(subset.event_id == event_id) & (subset.method == method)]
            if match.empty:
                continue
            r = match.iloc[0]
            rows.append({
                "event_id": event_id,
                "conv_id": int(er["conv_id"]),
                "method": method,
                "hit10": float(r["hit10"]),
                "ndcg10": float(r["ndcg10"]),
                "rank_top50": None if pd.isna(r["rank_top50"]) else int(r["rank_top50"]),
            })
    return pd.DataFrame(rows)


def run_policy(data, event_ids, initial_ledger=None, lam=0.0, use_kg_gate=True,
               update_ledger=True, collect_rows=True):
    ledger = initial_ledger if initial_ledger is not None else ExposureLedger()
    event_rows = []
    for event_id in sorted(event_ids):
        er = data["event_by_id"][event_id]
        cr = data["candidate_by_event"][event_id]
        ranked = rank_event(cr, data, ledger, lam, use_kg_gate)
        target = int(er["target_movie_id"])
        hit, ndcg, rank = event_relevance(ranked["movie_ids"], target)
        event_rows.append({
            "event_id": event_id,
            "conv_id": int(er["conv_id"]),
            "hit10": hit,
            "ndcg10": ndcg,
            "target_rank_top500": (
                int(np.flatnonzero(ranked["movie_ids"] == target)[0]) + 1
                if np.any(ranked["movie_ids"] == target) else None
            ),
            "eligible_categorized_candidates": ranked["eligible_count"],
            "categorized_candidates": ranked["categorized_count"],
            "top10_promoted_positions": ranked["promoted_count"],
            "top10_movie_ids": json.dumps(list(map(int, ranked["movie_ids"][:TOP_K]))),
            "top10_group_ids": json.dumps(list(map(int, ranked["groups"][:TOP_K]))),
        })
        if update_ledger:
            ledger.update(
                ranked["movie_ids"], data["item_categories"], ranked["groups"], TOP_K
            )
    return pd.DataFrame(event_rows), ledger


def seed_ledger_with_m2(data, event_ids):
    ledger = ExposureLedger()
    for event_id in sorted(event_ids):
        cr = data["candidate_by_event"][event_id]
        ids = cr["candidate_ids"]
        groups = np.asarray([data["item2group"][int(i)] for i in ids], dtype=int)
        ledger.update(ids, data["item_categories"], groups, TOP_K)
    return ledger


def measure_partition_fairness(df, event_ids, data):
    """Measure exposure/opportunity disparity for one evaluated event partition."""
    exposure = defaultdict(float)
    opportunity = defaultdict(float)

    for row in df.sort_values("event_id").itertuples(index=False):
        ids = list(map(int, json.loads(row.top10_movie_ids)))
        groups = list(map(int, json.loads(row.top10_group_ids)))
        for pos, movie_id in enumerate(ids):
            cats = data["item_categories"].get(movie_id, [])
            if not cats:
                continue
            weight = 1.0 / math.log2(pos + 2.0)
            for cat in cats:
                exposure[(cat, groups[pos])] += weight / len(cats)

    for event_id in event_ids:
        cr = data["candidate_by_event"][event_id]
        weights = ExposureLedger.opportunity_weights(len(cr["candidate_ids"]))
        for j, movie_id in enumerate(cr["candidate_ids"]):
            cats = data["item_categories"].get(int(movie_id), [])
            group = data["item2group"][int(movie_id)]
            if cats:
                for cat in cats:
                    opportunity[(cat, group)] += float(weights[j]) / len(cats)

    disparity, rows = category_disparity(exposure, opportunity)
    return disparity, rows, exposure, opportunity


def summarize_events(df):
    if df.empty:
        return {"events": 0, "HR@10": None, "NDCG@10": None}
    return {
        "events": int(len(df)),
        "HR@10": float(df.hit10.mean()),
        "NDCG@10": float(df.ndcg10.mean()),
        "mean_eligible_categorized_candidates": float(
            df.eligible_categorized_candidates.mean()
        ) if "eligible_categorized_candidates" in df else None,
        "mean_top10_promoted_positions": float(
            df.top10_promoted_positions.mean()
        ) if "top10_promoted_positions" in df else None,
    }


def cluster_bootstrap(df, n_boot=1000, seed=SEED):
    """Conversation-cluster bootstrap for event-level HR@10 and NDCG@10."""
    convs = np.asarray(sorted(df.conv_id.unique()), dtype=np.int64)
    grouped = {
        int(c): df[df.conv_id == c][["hit10", "ndcg10"]].to_numpy(dtype=float)
        for c in convs
    }
    rng = np.random.default_rng(seed)
    hr, ndcg = [], []
    for _ in range(n_boot):
        sampled = rng.choice(convs, size=len(convs), replace=True)
        vals = np.vstack([grouped[int(c)] for c in sampled])
        hr.append(float(vals[:, 0].mean()))
        ndcg.append(float(vals[:, 1].mean()))
    return {
        "conversations": int(len(convs)),
        "bootstrap_replicates": int(n_boot),
        "HR@10_ci95": [float(np.quantile(hr, 0.025)), float(np.quantile(hr, 0.975))],
        "NDCG@10_ci95": [float(np.quantile(ndcg, 0.025)), float(np.quantile(ndcg, 0.975))],
    }


def clone_ledger(source: ExposureLedger):
    out = ExposureLedger()
    out.exposure.update(source.exposure)
    out.opportunity.update(source.opportunity)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=Path("results_corrected/experiment_22_00"))
    parser.add_argument("--lambdas", type=str,
        default=",".join(map(str, DEFAULT_LAMBDAS)))
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    lambdas = sorted(set(float(x) for x in args.lambdas.split(",")))
    if not lambdas or any(x < 0 for x in lambdas):
        raise ValueError("Lambda grid must contain nonnegative values.")

    args.out.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("EXPERIMENT 22.0 — KG-FairCRS VALIDATION")
    print("=" * 78)
    print("Loading inputs...")
    data = load_inputs(args)

    events = data["events"]
    event_ids = sorted(map(int, events.event_id.unique()))
    conv_for_event = {
        int(r.event_id): int(r.conv_id)
        for r in events[["event_id", "conv_id"]].drop_duplicates().itertuples(index=False)
    }
    if len(event_ids) != 3010:
        raise ValueError(f"Expected 3,010 events, found {len(event_ids)}.")
    unique_conversations = len(set(conv_for_event.values()))
    if unique_conversations != 883:
        raise ValueError(f"Expected 883 unique conversations, found {unique_conversations}.")

    warm, tune, held = conversation_split(list(conv_for_event.values()))
    ids_by_conv = defaultdict(list)
    for event_id, conv_id in conv_for_event.items():
        ids_by_conv[conv_id].append(event_id)
    warm_events = sorted(eid for conv in warm for eid in ids_by_conv[conv])
    tune_events = sorted(eid for conv in tune for eid in ids_by_conv[conv])
    held_events = sorted(eid for conv in held for eid in ids_by_conv[conv])

    print(f"Events: {len(event_ids)} | conversations: {unique_conversations}")
    print(f"Split: warm={len(warm_events)}, tune={len(tune_events)}, heldout={len(held_events)}")

    # Warm-start historical state using only A1/M2 recommendations from warm-up conversations.
    warm_ledger = seed_ledger_with_m2(data, warm_events)

    # Tune lambda on a conversation-disjoint tuning partition.
    tuning_results = []
    for lam in lambdas:
        for label, gate in [("A2_category_only", False), ("A3_KG_FairCRS", True)]:
            ledger = clone_ledger(warm_ledger)
            result, ledger = run_policy(
                data, tune_events, initial_ledger=ledger, lam=lam,
                use_kg_gate=gate, update_ledger=True
            )
            result["method"] = label
            result["lambda"] = lam
            tuning_results.append(result)
    tune_df = pd.concat(tuning_results, ignore_index=True)
    tune_df.to_csv(args.out / "lambda_tuning_event_results.csv", index=False)

    # Select lambda for full KG-FairCRS only, subject to 95% of A1 NDCG@10.
    baseline_tune = evaluate_static_baselines(data, tune_events)
    baseline_a1 = baseline_tune[baseline_tune.method == METHOD_M2]
    baseline_ndcg = float(baseline_a1.ndcg10.mean())
    eligible = []
    for lam in lambdas:
        subset = tune_df[
            (tune_df.method == "A3_KG_FairCRS") & (tune_df["lambda"] == lam)
        ]
        ndcg = float(subset.ndcg10.mean())
        disparity, _, _, _ = measure_partition_fairness(
            subset, tune_events, data
        )
        eligible.append({
            "lambda": lam,
            "NDCG@10": ndcg,
            "HR@10": float(subset.hit10.mean()),
            "category_group_log_rate_MAD": disparity,
            "retention_vs_A1": ndcg / (baseline_ndcg + EPS),
            "meets_95pct_utility": ndcg >= 0.95 * baseline_ndcg,
        })
    eligible_df = pd.DataFrame(eligible)
    eligible_df.to_csv(args.out / "lambda_selection.csv", index=False)
    feasible = eligible_df[
        eligible_df.meets_95pct_utility
        & eligible_df.category_group_log_rate_MAD.notna()
    ]
    if feasible.empty:
        selected_lambda = 0.0
        selection_reason = (
            "No lambda met both the 95% NDCG constraint and supported "
            "category-group fairness calculation; selected no-promotion fallback."
        )
    else:
        # Primary objective: minimize category-group exposure disparity;
        # tie-break by higher NDCG, then smaller lambda.
        chosen = feasible.sort_values(
            ["category_group_log_rate_MAD", "NDCG@10", "lambda"],
            ascending=[True, False, True]
        ).iloc[0]
        selected_lambda = float(chosen["lambda"])
        selection_reason = (
            "Minimum category-group log exposure/opportunity disparity "
            "among settings satisfying the 95% NDCG@10 constraint."
        )

    # Rebuild state by replaying warm-up baseline, then tuning conversations using
    # the selected KG-FairCRS policy. Then evaluate held-out conversations.
    final_ledger = clone_ledger(warm_ledger)
    tune_replay, final_ledger = run_policy(
        data, tune_events, initial_ledger=final_ledger,
        lam=selected_lambda, use_kg_gate=True, update_ledger=True
    )
    held_kg, held_ledger = run_policy(
        data, held_events, initial_ledger=final_ledger,
        lam=selected_lambda, use_kg_gate=True, update_ledger=True
    )
    held_cat, _ = run_policy(
        data, held_events, initial_ledger=clone_ledger(final_ledger),
        lam=selected_lambda, use_kg_gate=False, update_ledger=True
    )

    held_kg["method"] = "A3_KG_FairCRS"
    held_cat["method"] = "A2_category_only"
    held_policy = pd.concat([held_cat, held_kg], ignore_index=True)
    held_policy.to_csv(args.out / "event_recommendations.csv", index=False)

    # Static baseline metrics on the same held-out conversations.
    base_df = evaluate_static_baselines(data, held_events)
    baseline_rows = []
    for method in [METHOD_BASE, METHOD_M2]:
        subset = base_df[base_df.method == method]
        baseline_rows.append({
            "method": method,
            "events": int(len(subset)),
            "HR@10": float(subset.hit10.mean()),
            "NDCG@10": float(subset.ndcg10.mean()),
        })
    baseline_summary = pd.DataFrame(baseline_rows)

    policy_summary = []
    for method, df in [("A2_category_only", held_cat), ("A3_KG_FairCRS", held_kg)]:
        s = summarize_events(df)
        s["method"] = method
        policy_summary.append(s)
    policy_summary = pd.DataFrame(policy_summary)

    # Compute held-out exposure/opportunity metrics from the actual policy rankings.
    metric_rows = []
    group_rows = []

    # Original A1 recommendations are the fairness reference.
    a1_rows = data["all_events"][
        (data["all_events"]["method"] == METHOD_M2)
        & (data["all_events"]["event_id"].isin(held_events))
    ][["event_id", "top10_movie_ids", "top10_group_ids"]].copy()
    a1_rows = a1_rows.merge(
        data["mapping"][["event_id", "conv_id"]],
        on="event_id", validate="one_to_one"
    )
    a1_disparity, a1_group_rows, a1_exp, a1_opp = measure_partition_fairness(
        a1_rows, held_events, data
    )
    metric_rows.append({
        "method": METHOD_M2,
        "category_group_log_rate_MAD": a1_disparity,
        "verified_categories": len({c for c, g in a1_opp}),
        "category_group_cells_with_opportunity": sum(v > EPS for v in a1_opp.values()),
        "heldout_verified_category_exposure": float(sum(a1_exp.values())),
        "heldout_verified_category_opportunity_mass": float(sum(a1_opp.values())),
    })
    for r in a1_group_rows:
        group_rows.append({"method": METHOD_M2, **r})

    for method, df in [("A2_category_only", held_cat), ("A3_KG_FairCRS", held_kg)]:
        disparity, rows, inc_exp, inc_opp = measure_partition_fairness(
            df, held_events, data
        )
        metric_rows.append({
            "method": method,
            "category_group_log_rate_MAD": disparity,
            "verified_categories": len({c for c, g in inc_opp}),
            "category_group_cells_with_opportunity": sum(v > EPS for v in inc_opp.values()),
            "heldout_verified_category_exposure": float(sum(inc_exp.values())),
            "heldout_verified_category_opportunity_mass": float(sum(inc_opp.values())),
        })
        for r in rows:
            group_rows.append({"method": method, **r})

    fairness_summary = pd.DataFrame(metric_rows)
    group_df = pd.DataFrame(group_rows)

    # Combine core utility and fairness summaries.
    comparison = pd.concat([
        baseline_summary.assign(
            mean_eligible_categorized_candidates=np.nan,
            mean_top10_promoted_positions=np.nan
        ),
        policy_summary[[
            "method", "events", "HR@10", "NDCG@10",
            "mean_eligible_categorized_candidates",
            "mean_top10_promoted_positions"
        ]]
    ], ignore_index=True)
    comparison = comparison.merge(fairness_summary, on="method", how="left")
    comparison.to_csv(args.out / "relevance_fairness_comparison.csv", index=False)
    group_df.to_csv(args.out / "category_group_exposure.csv", index=False)

    # Conversation-cluster bootstrap for held-out policy utility.
    ci = []
    for method, df in [("A2_category_only", held_cat), ("A3_KG_FairCRS", held_kg)]:
        result = cluster_bootstrap(df, n_boot=args.bootstrap)
        ci.append({"method": method, **result})
    pd.DataFrame(ci).to_csv(args.out / "conversation_bootstrap_ci.csv", index=False)

    # Validation checks and explicit limitations.
    checks = {
        "split": "validation_only",
        "test_split_evaluated": False,
        "checkpoint_modified": False,
        "events": len(event_ids),
        "conversations": unique_conversations,
        "warmup_conversations": len(warm),
        "tuning_conversations": len(tune),
        "heldout_conversations": len(held),
        "warmup_events": len(warm_events),
        "tuning_events": len(tune_events),
        "heldout_events": len(held_events),
        "primary_m2_method": METHOD_M2,
        "baseline_method": METHOD_BASE,
        "candidate_pool_size": 500,
        "top10_export_integrity": True,
        "all_candidate_ids_in_catalog": True,
        "all_popularity_groups_valid": True,
        "category_acceptance_logic": "Experiment 21.23B",
        "category_assignment_source": "Experiment 21.22A profile",
        "unknown_category_items_retained_without_category_bonus": True,
        "contextual_eligibility": "candidate KG rank > 0",
        "contextual_eligibility_is_not_semantic_relevance_claim": True,
        "chronological_order": "ascending event_id within processed split",
        "current_event_excluded_from_its_own_deficit": True,
        "lambda_grid": lambdas,
        "selected_lambda": selected_lambda,
        "selection_reason": selection_reason,
        "utility_constraint": "NDCG@10 >= 95% of A1_frequency_3 on tuning conversations",
        "limitations": [
            "Candidate pool is restricted to the exported M2 Top-500.",
            "KG rank > 0 is a KG-support signal, not a guarantee of semantic relevance.",
            "Category fairness is measurable only for validated category evidence.",
            "The 60/20/20 conversation split is one deterministic validation partition.",
            "Opportunity mass uses normalized rank-discount weights over the M2 Top-500; it is a proxy for predicted relevance, not observed user relevance."
        ],
    }
    (args.out / "validation_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )

    summary = {
        "experiment": "22.00_KG_FairCRS",
        "split": "validation_only",
        "selected_lambda": selected_lambda,
        "selection_reason": selection_reason,
        "tuning_baseline_A1_NDCG10": baseline_ndcg,
        "tuning_lambda_results": eligible,
        "heldout_baselines": baseline_summary.to_dict("records"),
        "heldout_policy_results": policy_summary.to_dict("records"),
        "heldout_fairness": fairness_summary.to_dict("records"),
        "category_assignment_counts": {
            "catalog_items": len(data["catalog_ids"]),
            "items_with_verified_categories": int(sum(bool(v) for v in data["item_categories"].values())),
            "items_without_verified_categories": int(sum(not bool(v) for v in data["item_categories"].values())),
            "distinct_categories": int(len({c for cats in data["item_categories"].values() for c in cats})),
        },
        "group_distribution": {
            str(g): int(sum(1 for x in data["item2group"].values() if x == g))
            for g in range(5)
        },
        "checks": checks,
    }
    (args.out / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print("EXPERIMENT 22.0 COMPLETE")
    print("=" * 78)
    print("Selected lambda:", selected_lambda)
    print("\nHeld-out relevance comparison:")
    print(comparison.to_string(index=False))
    print("\nHeld-out category fairness:")
    print(fairness_summary.to_string(index=False))
    print("\nOutputs:", args.out.resolve())
    print("Test split evaluated: NO")



# ---------------------------------------------------------------------------
# EXPERIMENT 22.1: KG relevance-gate ablation and graded support
# ---------------------------------------------------------------------------

KG_POLICIES = [
    ("B0_no_gate", "none", None),
    ("B1_rank_le_50", "threshold", 50),
    ("B2_rank_le_100", "threshold", 100),
    ("B3_rank_le_250", "threshold", 250),
    ("B4_rank_le_500", "threshold", 500),
    ("B5_graded_support", "graded", None),
]


def rank_event_22_1(candidate, data, ledger, lam, policy_mode, threshold=None):
    """Rerank one event under a controlled KG-support policy.

    KG ranks are contextual support signals only. They are not treated as
    semantic relevance labels. The candidate pool and base scores remain fixed.
    """
    ids = candidate["candidate_ids"]
    scores = candidate["scores"]
    kg = candidate["kg_ranks"]
    groups = np.asarray([data["item2group"][int(i)] for i in ids], dtype=int)
    cats_map = data["item_categories"]

    deficits = ledger.deficits(ids, groups, cats_map)
    category_bonus = np.zeros(len(ids), dtype=float)
    eligible_count = 0
    categorized_count = 0

    positive_kg_support = np.zeros(len(ids), dtype=float)
    positive = kg > 0
    if positive.any():
        # Monotonic, bounded support: rank 1 -> 1; lower-ranked evidence
        # contributes progressively less. Non-retrieved items remain 0.
        positive_kg_support[positive] = 1.0 / np.log2(kg[positive].astype(float) + 1.0)

    for j, movie_id in enumerate(ids):
        cats = cats_map.get(int(movie_id), [])
        if not cats:
            continue
        categorized_count += 1

        if policy_mode == "none":
            support = 1.0
        elif policy_mode == "threshold":
            support = float(kg[j] > 0 and kg[j] <= int(threshold))
        elif policy_mode == "graded":
            support = float(positive_kg_support[j])
        else:
            raise ValueError(f"Unknown KG policy mode: {policy_mode}")

        if support <= 0:
            continue

        eligible_count += 1
        frac = 1.0 / len(cats)
        historical_deficit = sum(
            frac * deficits.get((cat, int(groups[j])), 0.0)
            for cat in cats
        )
        category_bonus[j] = historical_deficit * support

    if category_bonus.max(initial=0.0) > 0:
        category_bonus /= category_bonus.max()

    score_scale = float(np.median(np.abs(scores)))
    if score_scale <= EPS:
        score_scale = float(np.max(np.abs(scores))) if len(scores) else 1.0

    adjusted = scores + (lam * score_scale * category_bonus)
    order = np.lexsort((ids, -adjusted))
    ranked_ids = ids[order]
    ranked_groups = groups[order]

    return {
        "movie_ids": ranked_ids,
        "groups": ranked_groups,
        "scores": adjusted[order],
        "kg_ranks": kg[order],
        "eligible_count": eligible_count,
        "categorized_count": categorized_count,
        "promoted_count": int(np.sum(order[:TOP_K] != np.arange(TOP_K))),
        "bonus": category_bonus,
    }


def run_policy_22_1(
    data, event_ids, initial_ledger=None, lam=0.0,
    policy_mode="none", threshold=None, update_ledger=True
):
    ledger = initial_ledger if initial_ledger is not None else ExposureLedger()
    rows = []
    for event_id in sorted(event_ids):
        er = data["event_by_id"][event_id]
        cr = data["candidate_by_event"][event_id]
        ranked = rank_event_22_1(
            cr, data, ledger, lam, policy_mode, threshold
        )
        target = int(er["target_movie_id"])
        hit, ndcg, rank = event_relevance(ranked["movie_ids"], target)
        rows.append({
            "event_id": int(event_id),
            "conv_id": int(er["conv_id"]),
            "hit10": int(hit),
            "ndcg10": float(ndcg),
            "target_rank_top500": (
                int(np.flatnonzero(ranked["movie_ids"] == target)[0]) + 1
                if np.any(ranked["movie_ids"] == target) else None
            ),
            "eligible_categorized_candidates": int(ranked["eligible_count"]),
            "categorized_candidates": int(ranked["categorized_count"]),
            "top10_promoted_positions": int(ranked["promoted_count"]),
            "top10_movie_ids": json.dumps(list(map(int, ranked["movie_ids"][:TOP_K]))),
            "top10_group_ids": json.dumps(list(map(int, ranked["groups"][:TOP_K]))),
        })
        if update_ledger:
            ledger.update(
                ranked["movie_ids"], data["item_categories"],
                ranked["groups"], TOP_K
            )
    return pd.DataFrame(rows), ledger


def paired_cluster_bootstrap_22_1(
    baseline_df, method_df, n_boot=1000, seed=SEED
):
    """Paired conversation-cluster bootstrap for HR@10 and NDCG@10 deltas."""
    b = baseline_df[["event_id", "conv_id", "hit10", "ndcg10"]].rename(
        columns={"hit10": "base_hit10", "ndcg10": "base_ndcg10"}
    )
    m = method_df[["event_id", "conv_id", "hit10", "ndcg10"]].rename(
        columns={"hit10": "method_hit10", "ndcg10": "method_ndcg10"}
    )
    paired = b.merge(m, on=["event_id", "conv_id"], validate="one_to_one")
    if len(paired) != len(b) or len(paired) != len(m):
        raise ValueError("Paired bootstrap requires identical event coverage.")

    convs = np.asarray(sorted(paired.conv_id.unique()), dtype=np.int64)
    grouped = {
        int(c): paired.loc[paired.conv_id == c, [
            "base_hit10", "method_hit10", "base_ndcg10", "method_ndcg10"
        ]].to_numpy(dtype=float)
        for c in convs
    }
    rng = np.random.default_rng(seed)
    dh, dn = [], []
    for _ in range(n_boot):
        sampled = rng.choice(convs, size=len(convs), replace=True)
        values = np.vstack([grouped[int(c)] for c in sampled])
        dh.append(float((values[:, 1] - values[:, 0]).mean()))
        dn.append(float((values[:, 3] - values[:, 2]).mean()))
    return {
        "conversations": int(len(convs)),
        "bootstrap_replicates": int(n_boot),
        "delta_HR@10": float(method_df.hit10.mean() - baseline_df.hit10.mean()),
        "delta_HR@10_ci95": [
            float(np.quantile(dh, 0.025)), float(np.quantile(dh, 0.975))
        ],
        "delta_NDCG@10": float(method_df.ndcg10.mean() - baseline_df.ndcg10.mean()),
        "delta_NDCG@10_ci95": [
            float(np.quantile(dn, 0.025)), float(np.quantile(dn, 0.975))
        ],
    }


def main_22_1():
    parser = argparse.ArgumentParser(
        description="Experiment 22.1: KG relevance-gate ablation and graded support."
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
        default=Path("results_corrected/experiment_22_01"))
    parser.add_argument("--lambdas", type=str,
        default=",".join(map(str, DEFAULT_LAMBDAS)))
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    lambdas = sorted(set(float(x) for x in args.lambdas.split(",")))
    if not lambdas or any(x < 0 or not np.isfinite(x) for x in lambdas):
        raise ValueError("Lambda grid must contain finite nonnegative values.")
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates.")

    args.out.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("EXPERIMENT 22.1 — KG RELEVANCE-GATE ABLATION")
    print("=" * 78)
    print("Loading the existing frozen candidate export...")
    data = load_inputs(args)

    events = data["events"]
    event_ids = sorted(map(int, events.event_id.unique()))
    conv_for_event = {
        int(r.event_id): int(r.conv_id)
        for r in events[["event_id", "conv_id"]].drop_duplicates().itertuples(index=False)
    }
    if len(event_ids) != 3010:
        raise ValueError(f"Expected 3,010 events, found {len(event_ids)}.")
    unique_conversations = len(set(conv_for_event.values()))
    if unique_conversations != 883:
        raise ValueError(
            f"Expected 883 unique conversations, found {unique_conversations}."
        )

    warm, tune, held = conversation_split(list(conv_for_event.values()))
    ids_by_conv = defaultdict(list)
    for event_id, conv_id in conv_for_event.items():
        ids_by_conv[conv_id].append(event_id)
    warm_events = sorted(eid for conv in warm for eid in ids_by_conv[conv])
    tune_events = sorted(eid for conv in tune for eid in ids_by_conv[conv])
    held_events = sorted(eid for conv in held for eid in ids_by_conv[conv])

    print(
        f"Events={len(event_ids)} conversations={unique_conversations} | "
        f"warm={len(warm_events)} tune={len(tune_events)} heldout={len(held_events)}"
    )

    warm_ledger = seed_ledger_with_m2(data, warm_events)
    baseline_tune = evaluate_static_baselines(data, tune_events)
    baseline_a1_tune = baseline_tune[baseline_tune.method == METHOD_M2]
    baseline_tune_ndcg = float(baseline_a1_tune.ndcg10.mean())
    utility_floor = 0.95 * baseline_tune_ndcg

    # Tune each gate and lambda on the same conversation-disjoint tuning set.
    tuning_rows = []
    tuning_event_frames = []
    selected_by_policy = []
    for label, mode, threshold in KG_POLICIES:
        for lam in lambdas:
            ledger = clone_ledger(warm_ledger)
            result, _ = run_policy_22_1(
                data, tune_events, initial_ledger=ledger, lam=lam,
                policy_mode=mode, threshold=threshold, update_ledger=True
            )
            result["policy"] = label
            result["lambda"] = lam
            tuning_event_frames.append(result)
            disparity, _, _, _ = measure_partition_fairness(
                result, tune_events, data
            )
            ndcg = float(result.ndcg10.mean())
            tuning_rows.append({
                "policy": label,
                "policy_mode": mode,
                "kg_rank_threshold": threshold,
                "lambda": lam,
                "events": int(len(result)),
                "HR@10": float(result.hit10.mean()),
                "NDCG@10": ndcg,
                "category_group_log_rate_MAD": disparity,
                "retention_vs_A1": ndcg / (baseline_tune_ndcg + EPS),
                "meets_95pct_utility": bool(ndcg >= utility_floor),
                "mean_eligible_categorized_candidates": float(
                    result.eligible_categorized_candidates.mean()
                ),
            })

        policy_tuning = pd.DataFrame(
            [r for r in tuning_rows if r["policy"] == label]
        )
        feasible = policy_tuning[
            policy_tuning.meets_95pct_utility
            & policy_tuning.category_group_log_rate_MAD.notna()
        ]
        if feasible.empty:
            chosen = {
                "policy": label,
                "policy_mode": mode,
                "kg_rank_threshold": threshold,
                "lambda": 0.0,
                "selection_status": "no_feasible_lambda_fallback",
                "tuning_fairness": None,
                "tuning_ndcg10": None,
            }
        else:
            best = feasible.sort_values(
                ["category_group_log_rate_MAD", "NDCG@10", "lambda"],
                ascending=[True, False, True]
            ).iloc[0]
            chosen = {
                "policy": label,
                "policy_mode": mode,
                "kg_rank_threshold": threshold,
                "lambda": float(best["lambda"]),
                "selection_status": "selected_under_95pct_utility_constraint",
                "tuning_fairness": float(best["category_group_log_rate_MAD"]),
                "tuning_ndcg10": float(best["NDCG@10"]),
            }
        selected_by_policy.append(chosen)

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv(args.out / "gate_lambda_tuning.csv", index=False)
    pd.concat(tuning_event_frames, ignore_index=True).to_csv(
        args.out / "tuning_event_results.csv", index=False
    )
    selection_df = pd.DataFrame(selected_by_policy)
    selection_df.to_csv(args.out / "selected_policy_parameters.csv", index=False)

    feasible_selected = selection_df[
        selection_df.selection_status == "selected_under_95pct_utility_constraint"
    ]
    if feasible_selected.empty:
        raise RuntimeError(
            "No KG gate policy met the tuning NDCG constraint. "
            "Do not proceed to held-out validation with a fabricated selection."
        )
    best_overall_row = feasible_selected.sort_values(
        ["tuning_fairness", "tuning_ndcg10", "lambda"],
        ascending=[True, False, True]
    ).iloc[0]
    # Keep native Python scalar types for reliable JSON serialization.
    overall_choice = next(
        item for item in selected_by_policy
        if item["policy"] == str(best_overall_row["policy"])
    )

    # For each gate, replay tuning events from the same warm state using its
    # tuning-selected lambda; then evaluate the held-out conversation partition.
    held_frames = []
    replay_ledgers = {}
    for chosen in selected_by_policy:
        label = chosen["policy"]
        mode = chosen["policy_mode"]
        threshold = chosen["kg_rank_threshold"]
        lam = float(chosen["lambda"])
        ledger = clone_ledger(warm_ledger)
        tune_replay, ledger = run_policy_22_1(
            data, tune_events, initial_ledger=ledger, lam=lam,
            policy_mode=mode, threshold=threshold, update_ledger=True
        )
        held_result, _ = run_policy_22_1(
            data, held_events, initial_ledger=clone_ledger(ledger), lam=lam,
            policy_mode=mode, threshold=threshold, update_ledger=True
        )
        held_result["method"] = label
        held_result["lambda"] = lam
        held_frames.append(held_result)
        replay_ledgers[label] = ledger

    held_all = pd.concat(held_frames, ignore_index=True)
    held_all.to_csv(args.out / "heldout_event_recommendations.csv", index=False)

    # Static A0/A1 references on exactly the same held-out events.
    baseline_held = evaluate_static_baselines(data, held_events)
    baseline_held.to_csv(args.out / "heldout_static_baselines.csv", index=False)
    a1_held = baseline_held[baseline_held.method == METHOD_M2].copy()

    # Fairness, opportunity, utility, and coverage for every pre-tuned policy.
    summary_rows = []
    group_rows = []
    bootstrap_rows = []
    paired_rows = []
    for label, mode, threshold in KG_POLICIES:
        result = held_all[held_all.method == label].copy()
        disparity, group_detail, exp, opp = measure_partition_fairness(
            result, held_events, data
        )
        chosen = next(x for x in selected_by_policy if x["policy"] == label)
        summary_rows.append({
            "method": label,
            "policy_mode": mode,
            "kg_rank_threshold": threshold,
            "selected_lambda": chosen["lambda"],
            "selection_status": chosen["selection_status"],
            "events": int(len(result)),
            "HR@10": float(result.hit10.mean()),
            "NDCG@10": float(result.ndcg10.mean()),
            "retention_vs_A1": float(result.ndcg10.mean() / (a1_held.ndcg10.mean() + EPS)),
            "category_group_log_rate_MAD": disparity,
            "mean_eligible_categorized_candidates": float(
                result.eligible_categorized_candidates.mean()
            ),
            "mean_top10_promoted_positions": float(
                result.top10_promoted_positions.mean()
            ),
            "verified_categories": int(len({c for c, g in opp})),
            "supported_category_group_cells": int(sum(v > EPS for v in opp.values())),
            "verified_category_exposure": float(sum(exp.values())),
            "verified_category_opportunity_mass": float(sum(opp.values())),
        })
        for row in group_detail:
            group_rows.append({"method": label, **row})
        bootstrap_rows.append({
            "method": label,
            **cluster_bootstrap(result, n_boot=args.bootstrap),
        })
        paired_rows.append({
            "method": label,
            **paired_cluster_bootstrap_22_1(
                a1_held, result, n_boot=args.bootstrap,
                seed=SEED + len(paired_rows)
            ),
        })

    # Include A1's own held-out fairness as an unmodified reference.
    a1_rows = data["all_events"][
        (data["all_events"].method == METHOD_M2)
        & (data["all_events"].event_id.isin(held_events))
    ][["event_id", "top10_movie_ids", "top10_group_ids"]].copy()
    a1_rows = a1_rows.merge(
        data["mapping"][["event_id", "conv_id"]],
        on="event_id", validate="one_to_one"
    )
    a1_disparity, a1_detail, a1_exp, a1_opp = measure_partition_fairness(
        a1_rows, held_events, data
    )
    summary_rows.insert(0, {
        "method": METHOD_M2,
        "policy_mode": "unchanged_baseline",
        "kg_rank_threshold": None,
        "selected_lambda": 0.0,
        "selection_status": "reference",
        "events": int(len(a1_held)),
        "HR@10": float(a1_held.hit10.mean()),
        "NDCG@10": float(a1_held.ndcg10.mean()),
        "retention_vs_A1": 1.0,
        "category_group_log_rate_MAD": a1_disparity,
        "mean_eligible_categorized_candidates": np.nan,
        "mean_top10_promoted_positions": 0.0,
        "verified_categories": int(len({c for c, g in a1_opp})),
        "supported_category_group_cells": int(sum(v > EPS for v in a1_opp.values())),
        "verified_category_exposure": float(sum(a1_exp.values())),
        "verified_category_opportunity_mass": float(sum(a1_opp.values())),
    })
    for row in a1_detail:
        group_rows.append({"method": METHOD_M2, **row})

    comparison = pd.DataFrame(summary_rows)
    comparison.to_csv(args.out / "heldout_comparison.csv", index=False)
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
        "experiment": "22.1_KG_relevance_gate_ablation",
        "split": "validation_only",
        "test_split_evaluated": False,
        "checkpoint_modified": False,
        "events": len(event_ids),
        "conversations": unique_conversations,
        "warmup_events": len(warm_events),
        "tuning_events": len(tune_events),
        "heldout_events": len(held_events),
        "candidate_pool": "same exported M2 Top-500",
        "primary_backbone": METHOD_M2,
        "same_conversation_split_as_22_0": True,
        "category_acceptance_logic": "Experiment 21.23B",
        "popularity_groups": "catalog_audit_corrected.pkl item2group",
        "kg_rank_interpretation": "contextual support signal, not semantic relevance label",
        "policies": [
            {"name": n, "mode": m, "threshold": t} for n, m, t in KG_POLICIES
        ],
        "graded_support_formula": "1/log2(KG_rank + 1) for KG rank > 0; zero otherwise",
        "utility_constraint": "tuning NDCG@10 >= 95% of A1_frequency_3",
        "selection": "per-policy minimum tuning disparity subject to utility constraint; overall policy selected on tuning partition",
        "event_order": "ascending event_id, matching Experiment 22.0",
        "current_event_excluded_from_own_historical_deficit": True,
        "paired_bootstrap_unit": "conversation",
        "bootstrap_replicates": args.bootstrap,
        "overall_tuning_selected_policy": overall_choice["policy"],
        "overall_tuning_selected_lambda": overall_choice["lambda"],
        "limitations": [
            "Only the frozen exported M2 Top-500 candidate pool is evaluated.",
            "KG rank is a retrieval-support proxy and does not establish semantic relevance.",
            "The graded support function is a prespecified monotonic rank weighting, not a learned relevance model.",
            "Category fairness is evaluated only for validated category assignments.",
            "The opportunity denominator remains the normalized rank-discounted candidate-pool proxy used in Experiment 22.0.",
            "This is a single deterministic conversation-disjoint validation split; no test set is accessed."
        ],
    }
    (args.out / "validation_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    summary = {
        "experiment": "22.1_KG_relevance_gate_ablation",
        "selected_on_tuning_partition": overall_choice,
        "tuning_baseline_A1_NDCG10": baseline_tune_ndcg,
        "utility_floor": utility_floor,
        "selected_parameters_by_policy": selected_by_policy,
        "heldout_comparison": comparison.to_dict("records"),
        "checks": checks,
    }
    (args.out / "validation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print("EXPERIMENT 22.1 COMPLETE")
    print("=" * 78)
    print("Policy selected on tuning:", overall_choice["policy"])
    print("Selected lambda:", overall_choice["lambda"])
    print("\nHeld-out validation comparison:")
    print(comparison[[
        "method", "HR@10", "NDCG@10", "retention_vs_A1",
        "category_group_log_rate_MAD",
        "mean_eligible_categorized_candidates"
    ]].to_string(index=False))
    print("\nOutputs:", args.out.resolve())
    print("Test split evaluated: NO")


if __name__ == "__main__":
    try:
        main_22_1()
    except Exception as exc:
        print(f"\nEXPERIMENT 22.1 FAILED: {exc}", file=sys.stderr)
        raise
