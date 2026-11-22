#!/usr/bin/env python
"""Experiment 22.11E — Relevance-Constrained Category Exposure Adequacy Audit.

Validation-only audit over frozen KBRD validation candidates, KG alignment,
existing Top-10 event results, reconciled 22.11C category evidence and the
22.11D catalog popularity mapping. No training, retrieval, reranking, tuning,
or independent-test access is performed.

Important methodological boundary:
The available KBRD evidence supports a KG-support indicator and the established
22.16 candidate-pool category-context proxy. It does NOT provide a validated
conversation-semantic relevance score q[e,i]. Therefore this script reports
KG-supported/category-context candidate availability separately and does not
rename that proxy as semantic relevance or invent a relevance threshold.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, pickle, platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_CATALOG = 6924
TOP_K = 10
SEED = 22115
DEFAULTS = {
    "alignment": "results_corrected/experiment_22_11/candidate_level_kbrd_kg_alignment.csv",
    "events": "results_corrected/experiment_22_10/event_level_results.csv",
    "profiles": "results_corrected/experiment_22_11C/reconciled_catalog_category_profiles.csv",
    "evidence": "results_corrected/experiment_22_11C/reconciled_item_category_evidence.csv",
    "catalog": "catalog_audit_corrected.pkl",
    "out": "results_corrected/experiment_22_11E",
}
BASE = "B0_KBRD"
METHODS = ["B_category_only_lambda_0p2", "B_KG_gated_lambda_0p2",
           "B_category_only_lambda_0p4", "B_KG_gated_lambda_0p4"]
SCENARIOS = ["strict_dbpedia_primary", "strict_plus_hybrid_sensitivity",
             "strict_plus_imdb_sensitivity", "imdb_external_only_report"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_list(value, field):
    if isinstance(value, (list, tuple, np.ndarray)):
        result = list(value)
    else:
        try:
            result = json.loads(str(value))
        except Exception:
            result = ast.literal_eval(str(value))
    if not isinstance(result, (list, tuple)):
        raise ValueError(f"{field} must be a list")
    return result


def split_categories(value):
    if pd.isna(value):
        return []
    return sorted({" ".join(x.strip().casefold().split())
                   for x in str(value).replace("|", ";").split(";") if x.strip()})


def require(df, cols, label):
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def evidence_populations(ev: pd.DataFrame):
    x = ev.copy()
    x["movie_entity_id"] = pd.to_numeric(x.movie_entity_id, errors="raise").astype(int)
    x["source_norm"] = x.source.astype(str).str.strip().str.casefold()
    x["type_norm"] = x.evidence_type.astype(str).str.strip().str.upper()
    x["identity_status"] = x.identity_status.fillna("").astype(str)
    x = x[x.identity_status.ne("AMBIGUOUS_REQUIRES_REVIEW")]
    rows = []
    for r in x.itertuples(index=False):
        for cat in split_categories(r.category):
            rows.append({"movie_id": int(r.movie_entity_id), "category": cat,
                         "source": r.source_norm, "evidence_type": r.type_norm})
    norm = pd.DataFrame(rows, columns=["movie_id", "category", "source", "evidence_type"])
    norm = norm.drop_duplicates(["movie_id", "category", "source", "evidence_type"])
    strict = norm[(norm.source == "dbpedia") & (norm.evidence_type == "ACCEPTED_GENRE")]
    hybrid = norm[(norm.source == "dbpedia") & (norm.evidence_type == "ACCEPTED_HYBRID")]
    imdb = norm[norm.source == "imdb"]
    return {
        "strict_dbpedia_primary": strict.drop_duplicates(["movie_id", "category"]).copy(),
        "strict_plus_hybrid_sensitivity": pd.concat([strict, hybrid]).drop_duplicates(["movie_id", "category"]).copy(),
        "strict_plus_imdb_sensitivity": pd.concat([strict, imdb]).drop_duplicates(["movie_id", "category"]).copy(),
        "imdb_external_only_report": imdb.drop_duplicates(["movie_id", "category"]).copy(),
    }


def context_prevalence(candidate_ids, item_categories):
    """Exact 22.16 candidate-pool category-context prevalence normalization."""
    counts = defaultdict(float)
    categorized = 0
    for mid in candidate_ids:
        cats = item_categories.get(int(mid), [])
        if not cats:
            continue
        categorized += 1
        for cat in cats:
            counts[cat] += 1.0 / len(cats)
    if not counts or not categorized:
        return {}, 0
    prev = {c: v / float(categorized) for c, v in counts.items()}
    maximum = max(prev.values())
    return {c: v / (maximum + 1e-12) for c, v in prev.items()}, categorized


def bootstrap_mean_ci(event_frame, value_col, reps, rng):
    """Conversation-cluster percentile CI for mean event-level outcome."""
    if event_frame.empty:
        return np.nan, np.nan, np.nan, 0
    per = event_frame.groupby("conv_id")[value_col].agg(["sum", "count"])
    convs = per.index.to_numpy()
    if len(convs) < 2:
        point = float(event_frame[value_col].mean())
        return point, np.nan, np.nan, len(convs)
    vals = np.empty(reps, dtype=float)
    sums = per["sum"].to_numpy(float)
    counts = per["count"].to_numpy(float)
    for b in range(reps):
        ix = rng.integers(0, len(convs), size=len(convs))
        vals[b] = sums[ix].sum() / counts[ix].sum() if counts[ix].sum() else np.nan
    vals = vals[np.isfinite(vals)]
    return (float(event_frame[value_col].mean()), float(np.quantile(vals, .025)),
            float(np.quantile(vals, .975)), len(convs))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    for key, default in DEFAULTS.items():
        ap.add_argument(f"--{key}", type=Path, default=Path(default))
    ap.add_argument("--bootstrap", type=int, default=2000)
    a = ap.parse_args()
    if a.bootstrap < 100:
        raise ValueError("--bootstrap must be at least 100")
    root = a.root.resolve()
    paths = {k: (getattr(a, k) if getattr(a, k).is_absolute() else root / getattr(a, k))
             for k in DEFAULTS}
    for key in ["alignment", "events", "profiles", "evidence", "catalog"]:
        if not paths[key].is_file():
            raise FileNotFoundError(f"Required frozen input not found ({key}): {paths[key]}")
    out = paths["out"]
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load frozen artifacts and enforce schemas.
    align = pd.read_csv(paths["alignment"])
    events = pd.read_csv(paths["events"])
    profiles = pd.read_csv(paths["profiles"])
    evidence = pd.read_csv(paths["evidence"])
    require(align, {"event_id", "conv_id", "movie_id", "kbrd_rank", "kg_rank", "kg_supported"}, "22.11B candidate alignment")
    require(events, {"event_id", "conv_id", "method", "top10_movie_ids"}, "22.10 event results")
    require(profiles, {"movie_entity_id", "identity_resolution_status"}, "22.11C profiles")
    require(evidence, {"movie_entity_id", "category", "source", "evidence_type", "identity_status"}, "22.11C evidence")
    for col in ["event_id", "conv_id", "movie_id", "kbrd_rank", "kg_rank", "kg_supported"]:
        align[col] = pd.to_numeric(align[col], errors="raise")
    align[["event_id", "conv_id", "movie_id", "kbrd_rank", "kg_rank", "kg_supported"]] = align[["event_id", "conv_id", "movie_id", "kbrd_rank", "kg_rank", "kg_supported"]].astype(int)
    for col in ["event_id", "conv_id"]:
        events[col] = pd.to_numeric(events[col], errors="raise").astype(int)
    profiles["movie_entity_id"] = pd.to_numeric(profiles.movie_entity_id, errors="raise").astype(int)
    if profiles.movie_entity_id.duplicated().any():
        raise RuntimeError("Duplicate movie IDs in 22.11C reconciled profiles")
    if align.duplicated(["event_id", "movie_id"]).any():
        raise RuntimeError("Duplicate (event_id,movie_id) rows in frozen candidate alignment")
    if (align.kg_rank < 0).any() or not set(align.kg_supported.unique()).issubset({0, 1}):
        raise RuntimeError("Invalid KG rank/support values")
    if events.duplicated(["event_id", "method"]).any():
        raise RuntimeError("Duplicate event-method rows in frozen Top-10 results")
    exposure_event_ids = set(events.event_id.astype(int))
    candidate_event_ids_full = set(align.event_id.astype(int))
    if not exposure_event_ids.issubset(candidate_event_ids_full):
        raise RuntimeError("Some observed Top-10 events are absent from candidate alignment")
    candidate_events_full_count = len(candidate_event_ids_full)
    # Use the exact 624-event exposure evaluation cohort for candidate-vs-
    # exposure comparisons; do not mix it with the larger 3,010-event export.
    align = align[align.event_id.isin(exposure_event_ids)].copy()
    expected_methods = {BASE, *METHODS}
    if set(events.method.astype(str)) != expected_methods:
        raise RuntimeError(f"Unexpected method set: {sorted(set(events.method.astype(str)))}")

    # 2. Load training-only global popularity groups.
    with paths["catalog"].open("rb") as f:
        catalog = pickle.load(f)
    if not isinstance(catalog, dict) or not {"movie_ids", "freq_dict", "item2group"}.issubset(catalog):
        raise RuntimeError("Catalog pickle lacks required movie_ids/freq_dict/item2group")
    movie_ids = set(map(int, catalog["movie_ids"]))
    freq = {int(k): int(v) for k, v in catalog["freq_dict"].items()}
    item2group = {int(k): int(v) for k, v in catalog["item2group"].items()}
    if len(movie_ids) != EXPECTED_CATALOG or set(freq) != movie_ids or set(item2group) != movie_ids:
        raise RuntimeError("Catalog/frequency/popularity mapping alignment failed")
    if set(profiles.movie_entity_id) != movie_ids:
        raise RuntimeError("22.11C profile IDs do not match the 6,924-item catalog")
    if set(item2group.values()) != {0, 1, 2, 3, 4}:
        raise RuntimeError("Global popularity groups must be 0..4")
    align["popularity_group"] = align.movie_id.map(item2group)
    if align.popularity_group.isna().any():
        raise RuntimeError("Candidate item missing global popularity group")
    align["popularity_group"] = align.popularity_group.astype(int)
    align["training_frequency"] = align.movie_id.map(freq).astype(int)
    align["interaction_class"] = np.select(
        [align.training_frequency.eq(0), align.training_frequency.eq(1), align.training_frequency.le(4)],
        ["Zero-shot", "One-shot", "Low-interaction"], default="Established")

    # 3. Build four provenance-preserving category populations.
    populations = evidence_populations(evidence)
    cat_maps = {}
    population_rows = []
    for scenario, frame in populations.items():
        frame = frame[frame.movie_id.isin(movie_ids)].drop_duplicates(["movie_id", "category"])
        cat_maps[scenario] = frame.groupby("movie_id").category.apply(list).to_dict()
        population_rows.append({"scenario": scenario, "evidence_role":
            "primary" if scenario == "strict_dbpedia_primary" else
            ("separate_external_source_report" if scenario == "imdb_external_only_report" else "sensitivity_only"),
            "categorized_catalog_items": int(frame.movie_id.nunique()),
            "category_assignments": int(len(frame)), "categories": int(frame.category.nunique()),
            "catalog_coverage": float(frame.movie_id.nunique()/EXPECTED_CATALOG)})
    population_summary = pd.DataFrame(population_rows)

    # 4. Candidate-level evidence/context metrics per scenario.
    candidate_frames, category_candidate_rows, opportunity_rows = [], [], []
    event_ids = sorted(set(align.event_id.astype(int)))
    by_event = {int(e): g.sort_values("kbrd_rank") for e, g in align.groupby("event_id", sort=False)}
    bad_candidate_sizes = {eid: len(g) for eid, g in by_event.items() if len(g) != 500}
    if bad_candidate_sizes:
        raise RuntimeError(f"Expected exactly 500 candidates per audited event; examples={dict(list(bad_candidate_sizes.items())[:5])}")
    bad_rank_sequences = [eid for eid,g in by_event.items() if g.kbrd_rank.duplicated().any()]
    if bad_rank_sequences:
        raise RuntimeError(f"Duplicate KBRD ranks within events: {bad_rank_sequences[:5]}")
    event_context = {}
    for scenario, item_cats in cat_maps.items():
        for eid in event_ids:
            g = by_event[eid]
            ids = g.movie_id.to_numpy(int)
            ctx, categorized_n = context_prevalence(ids, item_cats)
            event_context[(scenario, eid)] = ctx
            cats_per_item = [item_cats.get(int(mid), []) for mid in ids]
            # Match ExposureLedger.update_opportunity: normalized log-discount
            # weights over the complete frozen candidate pool.
            opp_weights = 1.0 / np.log2(np.arange(1, len(ids) + 1, dtype=float) + 1.0)
            opp_weights = opp_weights / (opp_weights.sum() + 1e-12)
            for row, cats, opp_w in zip(g.itertuples(index=False), cats_per_item, opp_weights):
                if cats:
                    for cat in cats:
                        opportunity_rows.append({"scenario": scenario, "event_id": eid,
                            "conv_id": int(row.conv_id), "movie_id": int(row.movie_id),
                            "category": cat, "popularity_group": int(row.popularity_group),
                            "candidate_opportunity_mass": float(opp_w / len(cats))})
            for row, cats in zip(g.itertuples(index=False), cats_per_item):
                cat_ctx = float(np.mean([ctx.get(c, 0.0) for c in cats])) if cats else 0.0
                candidate_frames.append({
                    "scenario": scenario, "event_id": eid, "conv_id": int(row.conv_id),
                    "movie_id": int(row.movie_id), "kbrd_rank": int(row.kbrd_rank),
                    "popularity_group": int(row.popularity_group),
                    "training_frequency": int(row.training_frequency),
                    "interaction_class": row.interaction_class,
                    "categories": ";".join(cats), "category_count": len(cats),
                    "categorized_candidate": int(bool(cats)),
                    "kg_rank": int(row.kg_rank), "kg_supported": int(row.kg_supported),
                    "candidate_pool_category_context": cat_ctx,
                    "kg_supported_and_categorized": int(bool(cats) and int(row.kg_supported) == 1),
                })
            category_items = defaultdict(set)
            for mid, cats in zip(ids, cats_per_item):
                for cat in cats:
                    category_items[cat].add(int(mid))
            for cat, mids in category_items.items():
                sub = g[g.movie_id.isin(mids)]
                category_candidate_rows.append({
                    "scenario": scenario, "event_id": eid, "conv_id": int(g.conv_id.iloc[0]),
                    "category": cat, "candidate_unique_items": int(len(mids)),
                    "candidate_rows": int(len(sub)),
                    "kg_supported_candidate_items": int(sub.loc[sub.kg_supported.eq(1), "movie_id"].nunique()),
                    "zero_shot_items": int(sub.loc[sub.interaction_class.eq("Zero-shot"), "movie_id"].nunique()),
                    "one_shot_items": int(sub.loc[sub.interaction_class.eq("One-shot"), "movie_id"].nunique()),
                    "low_interaction_items": int(sub.loc[sub.interaction_class.eq("Low-interaction"), "movie_id"].nunique()),
                    "established_items": int(sub.loc[sub.interaction_class.eq("Established"), "movie_id"].nunique()),
                    "mean_candidate_pool_context": float(np.mean([event_context[(scenario,eid)].get(cat,0.0)])),
                })
    candidate_detail = pd.DataFrame(candidate_frames)
    opportunity_detail = pd.DataFrame(opportunity_rows)
    item_opportunity = opportunity_detail.groupby(["scenario", "movie_id", "category"]).agg(
        candidate_appearances=("event_id", "nunique"),
        candidate_opportunity_mass=("candidate_opportunity_mass", "sum"),
        candidate_conversations=("conv_id", "nunique")).reset_index()
    event_opportunity = opportunity_detail.groupby(["scenario", "event_id", "conv_id", "category"]).agg(
        category_opportunity_mass=("candidate_opportunity_mass", "sum"),
        category_candidate_items=("movie_id", "nunique")).reset_index()
    event_category_candidates = pd.DataFrame(category_candidate_rows)
    candidate_summary = candidate_detail.groupby("scenario").agg(
        candidate_rows=("movie_id", "size"), unique_candidate_items=("movie_id", "nunique"),
        events=("event_id", "nunique"), conversations=("conv_id", "nunique"),
        categorized_candidate_rows=("categorized_candidate", "sum"),
        kg_supported_candidate_rows=("kg_supported", "sum"),
        kg_supported_categorized_rows=("kg_supported_and_categorized", "sum"),
        mean_category_context=("candidate_pool_category_context", "mean"),
    ).reset_index()
    candidate_summary["categorized_candidate_row_rate"] = candidate_summary.categorized_candidate_rows / candidate_summary.candidate_rows
    candidate_summary["kg_supported_categorized_row_rate"] = candidate_summary.kg_supported_categorized_rows / candidate_summary.candidate_rows

    category_candidate_summary = event_category_candidates.groupby(["scenario", "category"]).agg(
        candidate_events=("event_id", "nunique"), candidate_conversations=("conv_id", "nunique"),
        mean_candidate_items_per_event=("candidate_unique_items", "mean"),
        median_candidate_items_per_event=("candidate_unique_items", "median"),
        total_candidate_rows=("candidate_rows", "sum"),
        mean_kg_supported_items_per_event=("kg_supported_candidate_items", "mean"),
        events_with_zero_shot=("zero_shot_items", lambda x: int((x > 0).sum())),
        events_with_one_shot=("one_shot_items", lambda x: int((x > 0).sum())),
        events_with_low_interaction=("low_interaction_items", lambda x: int((x > 0).sum())),
        events_with_established=("established_items", lambda x: int((x > 0).sum())),
    ).reset_index()

    # 5. Reconstruct per-event/method Top-10 items and map them to evidence.
    top_rows = []
    cand_lookup = align.set_index(["event_id", "movie_id"])
    for r in events.itertuples(index=False):
        eid, conv, method = int(r.event_id), int(r.conv_id), str(r.method)
        ids = [int(x) for x in parse_list(r.top10_movie_ids, "top10_movie_ids")]
        if len(ids) != TOP_K or len(set(ids)) != TOP_K:
            raise RuntimeError(f"Invalid Top-10 list at event={eid}, method={method}")
        for scenario, item_cats in cat_maps.items():
            ctx = event_context.get((scenario, eid), {})
            for rank, mid in enumerate(ids, start=1):
                if (eid, mid) not in cand_lookup.index:
                    raise RuntimeError(f"Top-10 item absent from candidate alignment: {eid}/{mid}")
                z = cand_lookup.loc[(eid, mid)]
                cats = item_cats.get(mid, [])
                top_rows.append({
                    "scenario": scenario, "event_id": eid, "conv_id": conv,
                    "method": method, "rank": rank, "movie_id": mid,
                    "popularity_group": int(z.popularity_group),
                    "training_frequency": int(z.training_frequency),
                    "interaction_class": z.interaction_class,
                    "categories": ";".join(cats), "categorized": int(bool(cats)),
                    "kg_rank": int(z.kg_rank), "kg_supported": int(z.kg_supported),
                    "kg_supported_and_categorized": int(bool(cats) and int(z.kg_supported)==1),
                    "discounted_exposure": 1.0 / np.log2(rank + 1.0),
                    "candidate_pool_category_context": float(np.mean([ctx.get(c,0.0) for c in cats])) if cats else 0.0,
                })
    top_detail = pd.DataFrame(top_rows)

    # 6. Event-level category exposure; zeroes are retained for observed events.
    exp_rows = []
    event_methods = events[["event_id", "conv_id", "method"]].drop_duplicates()
    for scenario, item_cats in cat_maps.items():
        cats_all = sorted({c for cs in item_cats.values() for c in cs})
        for er in event_methods.itertuples(index=False):
            eid, conv, method = int(er.event_id), int(er.conv_id), str(er.method)
            sub = top_detail[(top_detail.scenario == scenario) & (top_detail.event_id == eid) & (top_detail.method == method)]
            for cat in cats_all:
                x = sub[sub.categories.map(lambda s: cat in s.split(";") if s else False)].copy()
                # Match ExposureLedger attribution: each item's exposure is
                # divided equally across its verified categories.
                if not x.empty:
                    x["category_fraction"] = x.categories.map(lambda s: 1.0 / max(1, len(s.split(";"))))
                else:
                    x["category_fraction"] = pd.Series(dtype=float)
                exp_rows.append({"scenario": scenario, "event_id": eid, "conv_id": conv,
                                 "method": method, "category": cat,
                                 "top10_item_category_incidences": int(len(x)),
                                 "fractional_category_exposure_mass": float(x.category_fraction.sum()),
                                 "discounted_category_exposure": float((x.discounted_exposure * x.category_fraction).sum()),
                                 "distinct_exposed_items": int(x.movie_id.nunique()),
                                 "kg_supported_category_exposure_mass": float((x.category_fraction * x.kg_supported).sum()),
                                 "eligible_proxy_category_exposure_mass": float((x.category_fraction * x.kg_supported_and_categorized).sum())})
    event_exposure = pd.DataFrame(exp_rows)
    # Candidate opportunity is method-independent: every reranking method for
    # the same event shares the identical frozen candidate pool. Therefore the
    # opportunity table is unique on (scenario,event,conversation,category),
    # while event_exposure intentionally has one row per method and repeats
    # those keys. This is a many-to-one join, not a one-to-one join.
    opportunity_key = ["scenario", "event_id", "conv_id", "category"]
    if event_opportunity.duplicated(opportunity_key).any():
        raise RuntimeError("Duplicate event-category keys in candidate opportunity ledger")
    event_exposure = event_exposure.merge(
        event_opportunity, on=opportunity_key, how="left", validate="many_to_one")
    event_exposure["category_opportunity_mass"] = event_exposure.category_opportunity_mass.fillna(0.0)
    event_exposure["category_candidate_items"] = event_exposure.category_candidate_items.fillna(0).astype(int)
    event_exposure["category_exposure_opportunity_ratio"] = (
        event_exposure.discounted_category_exposure / (event_exposure.category_opportunity_mass + 1e-12))
    exposure_summary = event_exposure.groupby(["scenario", "method", "category"]).agg(
        events=("event_id", "nunique"), conversations=("conv_id", "nunique"),
        total_top10_item_category_incidences=("top10_item_category_incidences", "sum"),
        total_fractional_category_exposure_mass=("fractional_category_exposure_mass", "sum"),
        mean_fractional_category_exposure_per_event=("fractional_category_exposure_mass", "mean"),
        total_category_opportunity_mass=("category_opportunity_mass", "sum"),
        total_discounted_category_exposure=("discounted_category_exposure", "sum"),
        mean_discounted_exposure_per_event=("discounted_category_exposure", "mean"),
        mean_exposure_opportunity_ratio=("category_exposure_opportunity_ratio", "mean"),
        events_with_exposure=("fractional_category_exposure_mass", lambda x: int((x > 0).sum())),
        events_with_zero_exposure=("fractional_category_exposure_mass", lambda x: int((x == 0).sum())),
        total_kg_supported_exposure_mass=("kg_supported_category_exposure_mass", "sum"),
        total_eligible_proxy_exposure_mass=("eligible_proxy_category_exposure_mass", "sum"),
    ).reset_index()
    exposure_summary["exposure_event_rate"] = exposure_summary.events_with_exposure / exposure_summary.events

    # 7. Item-level concentration and zero-exposure accounting, conditional on
    # items that actually appeared in the frozen candidate pools.
    item_rows = []
    for scenario, item_cats in cat_maps.items():
        candidate_ids = set(candidate_detail.loc[candidate_detail.scenario.eq(scenario), "movie_id"].astype(int))
        for method in sorted(expected_methods):
            sub = top_detail[(top_detail.scenario == scenario) & (top_detail.method == method)]
            exp = sub.groupby("movie_id").agg(
                top10_slots=("rank", "size"), discounted_exposure=("discounted_exposure", "sum"),
                exposed_events=("event_id", "nunique"), kg_supported=("kg_supported", "max"),
                popularity_group=("popularity_group", "first"), interaction_class=("interaction_class", "first"),
            )
            for mid in sorted(candidate_ids):
                cats = item_cats.get(mid, [])
                if not cats:
                    continue
                if mid in exp.index:
                    z = exp.loc[mid]
                    slots, de, ee, kg = int(z.top10_slots), float(z.discounted_exposure), int(z.exposed_events), int(z.kg_supported)
                    pg, ic = int(z.popularity_group), str(z.interaction_class)
                else:
                    slots, de, ee, kg = 0, 0.0, 0, int(align.loc[align.movie_id.eq(mid), "kg_supported"].max())
                    pg, ic = item2group[mid], ("Zero-shot" if freq[mid]==0 else "One-shot" if freq[mid]==1 else "Low-interaction" if freq[mid]<=4 else "Established")
                opp_sub = item_opportunity[(item_opportunity.scenario==scenario) & (item_opportunity.movie_id==mid)]
                candidate_appearances = int(opp_sub.candidate_appearances.sum()) if not opp_sub.empty else 0
                candidate_opportunity_mass = float(opp_sub.candidate_opportunity_mass.sum()) if not opp_sub.empty else 0.0
                item_rows.append({"scenario": scenario, "method": method, "movie_id": mid,
                                  "categories": ";".join(cats), "popularity_group": pg,
                                  "interaction_class": ic, "kg_supported_in_any_candidate_event": kg,
                                  "candidate_appearances": candidate_appearances,
                                  "candidate_opportunity_mass": candidate_opportunity_mass,
                                  "top10_slots": slots, "discounted_exposure": de,
                                  "category_attributed_discounted_exposure": de / len(cats),
                                  "exposure_opportunity_ratio": de / (candidate_opportunity_mass + 1e-12),
                                  "exposed_events": ee, "zero_exposure_in_observed_events": int(slots==0)})
    item_exposure = pd.DataFrame(item_rows)
    concentration_rows = []
    for (scenario, method), g in item_exposure.groupby(["scenario", "method"]):
        for cat, sub in g.assign(_cats=g.categories.str.split(";")).explode("_cats").groupby("_cats"):
            vals = sub.category_attributed_discounted_exposure.to_numpy(float)
            total = vals.sum()
            shares = vals / total if total > 0 else np.zeros_like(vals)
            concentration_rows.append({"scenario": scenario, "method": method, "category": cat,
                "candidate_items": int(sub.movie_id.nunique()),
                "items_with_zero_exposure": int(sub.zero_exposure_in_observed_events.sum()),
                "zero_exposure_item_rate": float(sub.zero_exposure_in_observed_events.mean()),
                "distinct_exposed_items": int((sub.top10_slots > 0).sum()),
                "total_discounted_exposure": float(total),
                "item_exposure_hhi": float(np.sum(shares**2)) if total>0 else np.nan,
                "largest_item_exposure_share": float(shares.max()) if total>0 else np.nan})
    concentration = pd.DataFrame(concentration_rows)

    # 8. Conversation-cluster bootstrap for event-level category exposure rate.
    rng = np.random.default_rng(SEED)
    boot_rows = []
    for (scenario, method, cat), g in event_exposure.groupby(["scenario", "method", "category"], sort=True):
        point, lo, hi, nconv = bootstrap_mean_ci(g, "fractional_category_exposure_mass", a.bootstrap, rng)
        # Also bootstrap probability an event has >=1 slot.
        g2 = g.assign(has_exposure=(g.fractional_category_exposure_mass > 0).astype(float))
        p, plo, phi, _ = bootstrap_mean_ci(g2, "has_exposure", a.bootstrap, rng)
        boot_rows.append({"scenario": scenario, "method": method, "category": cat,
                          "mean_fractional_category_exposure_per_event": point, "ci95_low_exposure": lo,
                          "ci95_high_exposure": hi, "event_exposure_rate": p,
                          "ci95_low_event_rate": plo, "ci95_high_event_rate": phi,
                          "conversation_clusters": nconv, "bootstrap_reps": a.bootstrap,
                          "sampling_unit": "conv_id", "seed": SEED})
    bootstrap = pd.DataFrame(boot_rows)

    # 9. Evidence-status labels. No arbitrary minimum-size or power threshold.
    feasibility = []
    for scenario in SCENARIOS:
        cc = category_candidate_summary[category_candidate_summary.scenario.eq(scenario)]
        for r in cc.itertuples(index=False):
            ee = exposure_summary[(exposure_summary.scenario==scenario) &
                                  (exposure_summary.category==r.category) & (exposure_summary.method==BASE)]
            exp_events = int(ee.events_with_exposure.iloc[0]) if not ee.empty else 0
            n_items = int(item_exposure[(item_exposure.scenario==scenario) &
                                         (item_exposure.method==BASE) &
                                         (item_exposure.categories.map(lambda s:r.category in s.split(";") if s else False))].movie_id.nunique())
            n_convs = int(r.candidate_conversations)
            # No arbitrary minimum-size threshold is imposed here. This is an
            # evidence-status classification, not a formal power determination.
            if n_items > 0 and n_convs > 0 and exp_events > 0:
                label = "candidate_and_baseline_exposure_observed"
            elif n_items > 0 and n_convs > 0:
                label = "candidate_supported_but_no_baseline_exposure_observed"
            else:
                label = "not_represented_in_candidate_population"
            feasibility.append({"scenario":scenario,"category":r.category,
                "candidate_conversations":n_convs,"mean_candidate_items_per_event":float(r.mean_candidate_items_per_event),
                "candidate_unique_items_in_catalog":n_items,"baseline_events_with_exposure":exp_events,
                "classification":label,
                "thresholds_are_descriptive_not_formal_power":True})
    feasibility = pd.DataFrame(feasibility)

    # 10. Outputs.
    outputs = {
        "evidence_population_summary.csv": population_summary,
        "candidate_level_category_kg_context.csv": candidate_detail,
        "candidate_population_summary.csv": candidate_summary,
        "category_candidate_support_by_event.csv": event_category_candidates,
        "candidate_opportunity_item_ledger.csv": item_opportunity,
        "event_category_opportunity_ledger.csv": event_opportunity,
        "category_candidate_support_summary.csv": category_candidate_summary,
        "event_method_top10_item_evidence.csv": top_detail,
        "event_category_exposure_ledger.csv": event_exposure,
        "category_exposure_summary.csv": exposure_summary,
        "item_level_exposure_ledger.csv": item_exposure,
        "category_item_exposure_concentration.csv": concentration,
        "category_exposure_cluster_bootstrap.csv": bootstrap,
        "category_feasibility_classification.csv": feasibility,
    }
    for name, df in outputs.items():
        df.to_csv(out/name, index=False)

    # 11. Validation: integrity keys must pass; safety guardrails must remain false.
    event_sets = {m:set(events.loc[events.method.eq(m),"event_id"].astype(int)) for m in expected_methods}
    checks = {
        "experiment":"22.11E_relevance_constrained_category_exposure_adequacy",
        "candidate_event_movie_unique": not align.duplicated(["event_id","movie_id"]).any(),
        "catalog_size_6924": len(movie_ids)==EXPECTED_CATALOG,
        "profile_catalog_id_alignment": set(profiles.movie_entity_id)==movie_ids,
        "exactly_500_unique_candidates_per_audited_event": all(len(g)==500 and g.movie_id.nunique()==500 for g in by_event.values()),
        "popularity_groups_0_to_4": set(item2group.values())=={0,1,2,3,4},
        "all_five_methods_present": set(events.method.astype(str))==expected_methods,
        "all_methods_share_event_ids": all(event_sets[m]==event_sets[BASE] for m in METHODS),
        "top10_lists_have_ten_unique_items": bool((top_detail.groupby(["scenario","event_id","method"]).movie_id.nunique()==TOP_K).all()),
        "top10_items_exist_in_frozen_candidate_alignment": True,
        "all_four_evidence_populations_present": set(populations)==set(SCENARIOS),
        "strict_primary_uses_dbpedia_accepted_genre_only": set(populations["strict_dbpedia_primary"].source)=={"dbpedia"} and set(populations["strict_dbpedia_primary"].evidence_type)=={"ACCEPTED_GENRE"},
        "hybrid_and_imdb_remain_separate_sensitivity_populations": True,
        "ambiguous_identity_excluded": not (set(candidate_detail.loc[candidate_detail.categorized_candidate.eq(1),"movie_id"]) & set(evidence.loc[evidence.identity_status.eq("AMBIGUOUS_REQUIRES_REVIEW"),"movie_entity_id"].astype(int))),
        "no_training": False, "no_retrieval_rerun": False, "no_reranking": False,
        "no_parameter_tuning": False, "independent_test_not_read": False,
        "relevance_semantics_limited_to_existing_context_and_kg_support_proxy": True,
    }
    integrity = [k for k,v in checks.items() if isinstance(v,bool) and k not in
                 {"no_training","no_retrieval_rerun","no_reranking","no_parameter_tuning","independent_test_not_read"}]
    guardrails = ["no_training","no_retrieval_rerun","no_reranking","no_parameter_tuning","independent_test_not_read"]
    # These flags state whether the prohibited operation occurred: all must be False.
    checks["overall_pass"] = all(checks[k] for k in integrity) and all(not checks[k] for k in guardrails)
    with (out/"validation_checks.json").open("w",encoding="utf-8") as f:
        json.dump(checks,f,indent=2,allow_nan=False)

    inputs = {k:{"path":str(p.resolve()),"sha256":sha256(p),"bytes":p.stat().st_size}
              for k,p in paths.items() if k!="out"}
    manifest = {
        "experiment":"22.11E_relevance_constrained_category_exposure_adequacy",
        "created_utc":datetime.now(timezone.utc).isoformat(),"python":platform.python_version(),
        "numpy":np.__version__,"pandas":pd.__version__,"scope":"frozen KBRD validation candidates and existing Top-10 event outputs",
        "candidate_events_in_full_frozen_export":candidate_events_full_count,
        "candidate_events_audited_matched_to_exposure_cohort":int(align.event_id.nunique()),
        "candidate_conversations_in_audited_cohort":int(align.conv_id.nunique()),
        "candidate_rows_in_audited_cohort":int(len(align)),"exposure_events":int(events.event_id.nunique()),
        "exposure_conversations":int(events.conv_id.nunique()),"top10_methods":sorted(expected_methods),
        "bootstrap_reps":a.bootstrap,"bootstrap_seed":SEED,"bootstrap_unit":"conversation_id",
        "context_definition":"22.16 normalized category prevalence among categorized items in frozen event Top-500 candidate pool",
        "opportunity_definition":"22.16 normalized log-discount candidate-pool opportunity weights, divided equally across each item's categories",
        "kg_support_definition":"existing kg_supported indicator from frozen 22.11B candidate alignment",
        "relevance_limitation":"No validated dialogue-semantic q[e,i] is present in these KBRD artifacts; established 22.16 candidate-pool context and KG support are reported separately, not called semantic relevance.",
        "exposure_attribution":"Discounted Top-10 exposure is divided equally across each item's verified categories, matching the ExposureLedger multi-category attribution rule.",
        "feasibility_classification":"Evidence-status labels only; no invented minimum sample-size threshold or formal power claim.",
        "input_files":inputs,"outputs":list(outputs)+["validation_checks.json"]}
    with (out/"experiment_manifest.json").open("w",encoding="utf-8") as f:
        json.dump(manifest,f,indent=2,allow_nan=False)

    print("\n"+"="*78+"\n22.11E COMPLETE\n"+"="*78)
    print("\nEVIDENCE POPULATIONS\n",population_summary.to_string(index=False))
    print("\nCANDIDATE SUPPORT\n",candidate_summary.to_string(index=False))
    print("\nBASELINE CATEGORY EXPOSURE (top 20 by fractional exposure mass)\n",
          exposure_summary[exposure_summary.method.eq(BASE)].sort_values("total_fractional_category_exposure_mass",ascending=False).head(20).to_string(index=False))
    print("\nFEASIBILITY CLASSIFICATIONS\n",feasibility.groupby(["scenario","classification"]).size().rename("categories").reset_index().to_string(index=False))
    print("\nValidation overall_pass:",checks["overall_pass"])
    print("Training: NO | Retrieval rerun: NO | Reranking: NO | Parameter tuning: NO | Independent test: NOT READ")
    print("Output:",out.resolve())
    if not checks["overall_pass"]:
        raise RuntimeError("22.11E validation failed; inspect validation_checks.json")


if __name__ == "__main__":
    main()





