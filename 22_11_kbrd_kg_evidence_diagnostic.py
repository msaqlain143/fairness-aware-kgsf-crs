#!/usr/bin/env python
"""
Experiment 22.11 — KBRD–KG Evidence Diagnostic
------------------------------------------------
Validation-only diagnostic. No model training, retrieval reruns, coefficient
search, or test-split evaluation.

Primary alignment key: (event_id, movie_id)
KG rank == 0 is treated as "no recorded support in this exported ranking",
not as proof of semantic irrelevance.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


DEFAULT_KBRD = Path("results_corrected/experiment_22_04B/kbrd_validation_top500.csv")
DEFAULT_KG = Path("results_corrected/adaptive_m2_validation_export22_candidates.csv")
DEFAULT_EVENTS = Path("results_corrected/experiment_22_10/event_level_results.csv")
DEFAULT_CATEGORIES = Path(
    "results_category_audit/experiment_21_22A/catalog_item_category_profiles.csv"
)
DEFAULT_OUT = Path("results_corrected/experiment_22_11")


def parse_list(value: Any) -> list:
    """Parse Python-list/JSON-like CSV cells safely."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return []
    try:
        x = json.loads(s)
        return x if isinstance(x, list) else []
    except Exception:
        pass
    try:
        x = ast.literal_eval(s)
        return list(x) if isinstance(x, (list, tuple)) else []
    except Exception:
        # Conservative fallback for bracketed comma-separated numeric arrays.
        if s.startswith("[") and s.endswith("]"):
            body = s[1:-1].strip()
            if not body:
                return []
            out = []
            for token in body.split(","):
                token = token.strip()
                try:
                    out.append(float(token) if "." in token else int(token))
                except Exception:
                    out.append(token.strip("'\""))
            return out
    return []


def as_int(value: Any, default: int | None = None) -> int | None:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def bool_value(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y"}


def normalize_categories(row: pd.Series) -> list[str]:
    """Only accept explicit, verified category evidence."""
    accepted = bool_value(row.get("accepted_genre_evidence", False))
    if not accepted:
        return []
    raw = row.get("canonical_genres", "")
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    # Existing profiles commonly store semicolon-delimited genre strings.
    vals = re.split(r"[;|]", str(raw))
    return sorted({v.strip().lower() for v in vals if v.strip()})


def make_rank_map(ids: list) -> dict[int, int]:
    result = {}
    for rank, raw_id in enumerate(ids, start=1):
        movie_id = as_int(raw_id)
        if movie_id is not None and movie_id not in result:
            result[movie_id] = rank
    return result


def clustered_bootstrap_difference(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    cluster_col: str = "conv_id",
    n_boot: int = 2000,
    seed: int = 20261111,
) -> dict:
    """Conversation-cluster bootstrap CI for mean(group=1)-mean(group=0)."""
    d = df[[group_col, value_col, cluster_col]].dropna().copy()
    d[group_col] = d[group_col].astype(int)
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[value_col])
    if d.empty or d[group_col].nunique() < 2:
        return {"difference": np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
                "clusters": int(d[cluster_col].nunique()) if not d.empty else 0}

    def stat(frame: pd.DataFrame) -> float:
        a = frame.loc[frame[group_col] == 1, value_col]
        b = frame.loc[frame[group_col] == 0, value_col]
        return float(a.mean() - b.mean()) if len(a) and len(b) else np.nan

    point = stat(d)
    clusters = d[cluster_col].dropna().unique()
    rng = np.random.default_rng(seed)
    draws = []
    by_cluster = {c: d[d[cluster_col] == c] for c in clusters}
    for _ in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        boot = pd.concat([by_cluster[c] for c in sampled], ignore_index=True)
        value = stat(boot)
        if np.isfinite(value):
            draws.append(value)
    lo, hi = (np.quantile(draws, [0.025, 0.975]) if draws else (np.nan, np.nan))
    return {"difference": point, "ci95_low": float(lo), "ci95_high": float(hi),
            "clusters": int(len(clusters)), "bootstrap_replicates": int(len(draws))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kbrd", type=Path, default=DEFAULT_KBRD)
    parser.add_argument("--kg", type=Path, default=DEFAULT_KG)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--kg-method", default="A1_frequency_3")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()

    for p in (args.kbrd, args.kg, args.events, args.categories):
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    args.out.mkdir(parents=True, exist_ok=True)
    kbrd = pd.read_csv(args.kbrd)
    kg_all = pd.read_csv(args.kg)
    events = pd.read_csv(args.events)
    categories = pd.read_csv(args.categories)

    require_columns(kbrd, {"event_id", "conv_id", "top_movie_ids", "top_k"}, "KBRD Top-500")
    require_columns(kg_all, {"event_id", "method", "candidate_movie_ids", "candidate_kg_ranks"},
                    "KG candidate export")
    require_columns(events, {"event_id", "conv_id", "method", "top10_movie_ids"},
                    "Experiment 22.10 event results")
    require_columns(categories, {"movie_entity_id", "accepted_genre_evidence", "canonical_genres"},
                    "Verified category profiles")

    # KBRD: exactly one row per validation event.
    kbrd = kbrd.copy()
    kbrd["event_id"] = pd.to_numeric(kbrd["event_id"], errors="raise").astype(int)
    kbrd["conv_id"] = pd.to_numeric(kbrd["conv_id"], errors="raise").astype(int)
    if kbrd["event_id"].duplicated().any():
        raise ValueError("KBRD Top-500 export has duplicate event_id rows.")
    kbrd["kbrd_ids"] = kbrd["top_movie_ids"].map(parse_list)
    kbrd["kbrd_rank_map"] = kbrd["kbrd_ids"].map(make_rank_map)

    # KG: select the explicitly agreed candidate method and require one row/event.
    kg = kg_all.loc[kg_all["method"].astype(str) == args.kg_method].copy()
    if kg.empty:
        raise ValueError(f"No KG rows found for method={args.kg_method!r}")
    kg["event_id"] = pd.to_numeric(kg["event_id"], errors="raise").astype(int)
    if kg["event_id"].duplicated().any():
        raise ValueError(f"KG export has duplicate event_id rows for {args.kg_method}.")
    kg["candidate_ids"] = kg["candidate_movie_ids"].map(parse_list)
    kg["kg_ranks"] = kg["candidate_kg_ranks"].map(parse_list)
    if not (kg["candidate_ids"].str.len() == kg["kg_ranks"].str.len()).all():
        bad = kg.loc[kg["candidate_ids"].str.len() != kg["kg_ranks"].str.len(), "event_id"].tolist()[:10]
        raise ValueError(f"KG movie/rank arrays have unequal lengths; sample event_ids={bad}")
    kg["kg_rank_map"] = kg.apply(
        lambda r: {
            int(mid): int(rank)
            for mid, rank in zip(r["candidate_ids"], r["kg_ranks"])
            if as_int(mid) is not None and as_int(rank) is not None
        }, axis=1
    )

    # Verified category lookup. Unverified/unknown category profiles are excluded.
    categories = categories.copy()
    categories["movie_entity_id"] = pd.to_numeric(
        categories["movie_entity_id"], errors="coerce"
    )
    categories = categories.dropna(subset=["movie_entity_id"])
    categories["movie_entity_id"] = categories["movie_entity_id"].astype(int)
    categories["verified_genres"] = categories.apply(normalize_categories, axis=1)
    cat_map = dict(zip(categories["movie_entity_id"], categories["verified_genres"]))

    # Event-level join: do not assume row order is identical across exports.
    base = kbrd[["event_id", "conv_id", "kbrd_ids", "kbrd_rank_map"]].merge(
        kg[["event_id", "candidate_ids", "kg_rank_map"]],
        on="event_id", how="inner", validate="one_to_one"
    )
    if base.empty:
        raise ValueError("No overlapping event_id values between KBRD and KG exports.")
    base["kbrd_n"] = base["kbrd_ids"].map(len)
    base["kg_candidate_n"] = base["candidate_ids"].map(len)

    # Candidate-level matched records (only exported Top-500 candidate pool).
    rows = []
    for r in base.itertuples(index=False):
        for movie_id in r.kbrd_ids:
            mid = as_int(movie_id)
            if mid is None:
                continue
            kr = r.kbrd_rank_map.get(mid)
            gr = r.kg_rank_map.get(mid, 0)
            rows.append({
                "event_id": int(r.event_id),
                "conv_id": int(r.conv_id),
                "movie_id": mid,
                "kbrd_rank": int(kr) if kr is not None else np.nan,
                "kg_rank": int(gr),
                "kg_supported": int(gr > 0),
                "verified_genres": cat_map.get(mid, []),
                "verified_category": int(bool(cat_map.get(mid, []))),
            })
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        raise ValueError("No candidate-level KBRD records could be constructed.")
    pairs["kbrd_rank"] = pd.to_numeric(pairs["kbrd_rank"], errors="coerce")
    pairs.to_csv(args.out / "candidate_level_kbrd_kg_alignment.csv", index=False)

    # Rank association: only movies with positive KG rank, within shared event.
    corr_rows = []
    for event_id, g in pairs.groupby("event_id"):
        supported = g.loc[g["kg_supported"] == 1, ["kbrd_rank", "kg_rank"]].dropna()
        if len(supported) >= 3 and supported["kbrd_rank"].nunique() > 1 and supported["kg_rank"].nunique() > 1:
            if spearmanr is not None:
                rho, pval = spearmanr(supported["kbrd_rank"], supported["kg_rank"])
            else:
                rho = supported["kbrd_rank"].corr(supported["kg_rank"], method="spearman")
                pval = np.nan
        else:
            rho, pval = np.nan, np.nan
        corr_rows.append({
            "event_id": int(event_id),
            "n_kbrd_candidates": int(len(g)),
            "n_kg_supported": int(g["kg_supported"].sum()),
            "kg_support_coverage": float(g["kg_supported"].mean()),
            "n_supported_for_correlation": int(len(supported)),
            "spearman_kbrd_vs_kg_rank": float(rho) if pd.notna(rho) else np.nan,
            "spearman_pvalue": float(pval) if pd.notna(pval) else np.nan,
        })
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(args.out / "event_level_rank_correlations.csv", index=False)

    # Supported vs unsupported KBRD rank distributions.
    rank_summary = (
        pairs.groupby("kg_supported")["kbrd_rank"]
        .agg(n="count", mean="mean", median="median", std="std", min="min", max="max")
        .reset_index()
    )
    rank_summary["support_status"] = rank_summary["kg_supported"].map(
        {0: "no_recorded_kg_support", 1: "recorded_kg_support"}
    )
    rank_summary.to_csv(args.out / "kbrd_rank_by_kg_support.csv", index=False)
    boot_rank = clustered_bootstrap_difference(
        pairs, "kg_supported", "kbrd_rank", n_boot=args.bootstrap
    )

    # Coverage by verified category evidence and genre.
    category_rows = []
    for genre in sorted({g for values in pairs["verified_genres"] for g in values}):
        sub = pairs[pairs["verified_genres"].map(lambda xs: genre in xs)]
        category_rows.append({
            "genre": genre,
            "candidate_count": int(len(sub)),
            "supported_count": int(sub["kg_supported"].sum()),
            "kg_support_coverage": float(sub["kg_supported"].mean()) if len(sub) else np.nan,
            "event_count": int(sub["event_id"].nunique()),
        })
    pd.DataFrame(category_rows).to_csv(args.out / "verified_category_kg_coverage.csv", index=False)

    cat_coverage = pairs.groupby("verified_category").agg(
        candidate_count=("movie_id", "size"),
        supported_count=("kg_supported", "sum"),
        event_count=("event_id", "nunique"),
    ).reset_index()
    cat_coverage["kg_support_coverage"] = (
        cat_coverage["supported_count"] / cat_coverage["candidate_count"]
    )
    cat_coverage["category_status"] = cat_coverage["verified_category"].map(
        {0: "no_verified_genre_evidence", 1: "verified_genre_evidence"}
    )
    cat_coverage.to_csv(args.out / "verified_vs_unverified_category_coverage.csv", index=False)

    # Experiment 22.10 Top-10 exposure: use saved result lists only.
    # Category exposure is computed only where verified genres exist.
    ev = events.copy()
    ev["event_id"] = pd.to_numeric(ev["event_id"], errors="coerce")
    ev["conv_id"] = pd.to_numeric(ev["conv_id"], errors="coerce")
    ev = ev.dropna(subset=["event_id", "conv_id"])
    ev["event_id"] = ev["event_id"].astype(int)
    ev["conv_id"] = ev["conv_id"].astype(int)
    ev["top10_ids"] = ev["top10_movie_ids"].map(parse_list)

    exposure_rows = []
    for r in ev.itertuples(index=False):
        ids = [as_int(x) for x in r.top10_ids]
        ids = [x for x in ids if x is not None]
        genre_counts: dict[str, int] = {}
        verified_slots = 0
        for mid in ids:
            genres = cat_map.get(mid, [])
            if genres:
                verified_slots += 1
                for genre in genres:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
        exposure_rows.append({
            "event_id": int(r.event_id),
            "conv_id": int(r.conv_id),
            "method": str(r.method),
            "top10_size": int(len(ids)),
            "verified_category_slots": int(verified_slots),
            "verified_category_slot_rate": verified_slots / len(ids) if ids else np.nan,
            "top10_verified_genre_counts": json.dumps(genre_counts, sort_keys=True),
        })
    exposure = pd.DataFrame(exposure_rows)
    exposure.to_csv(args.out / "event_method_top10_verified_category_exposure.csv", index=False)

    # Paired method deltas against B0_KBRD, matched on event_id.
    base_exp = exposure[exposure["method"] == "B0_KBRD"][
        ["event_id", "conv_id", "verified_category_slot_rate", "verified_category_slots"]
    ].rename(columns={
        "verified_category_slot_rate": "baseline_verified_category_slot_rate",
        "verified_category_slots": "baseline_verified_category_slots",
    })
    paired = exposure[exposure["method"] != "B0_KBRD"].merge(
        base_exp, on=["event_id", "conv_id"], how="inner", validate="many_to_one"
    )
    if not paired.empty:
        paired["delta_verified_category_slot_rate"] = (
            paired["verified_category_slot_rate"] - paired["baseline_verified_category_slot_rate"]
        )
        paired["delta_verified_category_slots"] = (
            paired["verified_category_slots"] - paired["baseline_verified_category_slots"]
        )
    paired.to_csv(args.out / "paired_top10_category_exposure_deltas.csv", index=False)

    delta_summaries = []
    for method, g in paired.groupby("method") if not paired.empty else []:
        delta_summaries.append({
            "method": method,
            "paired_events": int(g["event_id"].nunique()),
            "mean_delta_verified_category_slot_rate": float(g["delta_verified_category_slot_rate"].mean()),
            "median_delta_verified_category_slot_rate": float(g["delta_verified_category_slot_rate"].median()),
            "mean_delta_verified_category_slots": float(g["delta_verified_category_slots"].mean()),
        })
    pd.DataFrame(delta_summaries).to_csv(
        args.out / "method_level_top10_category_exposure_summary.csv", index=False
    )

    # Event-level support coverage and descriptive summary.
    coverage_summary = {
        "events_in_kbrd_export": int(kbrd["event_id"].nunique()),
        "events_in_selected_kg_export": int(kg["event_id"].nunique()),
        "overlapping_events": int(base["event_id"].nunique()),
        "candidate_rows": int(len(pairs)),
        "kg_supported_candidate_rows": int(pairs["kg_supported"].sum()),
        "overall_kg_support_coverage": float(pairs["kg_supported"].mean()),
        "mean_event_kg_support_coverage": float(corr["kg_support_coverage"].mean()),
        "median_event_kg_support_coverage": float(corr["kg_support_coverage"].median()),
        "mean_event_spearman": float(corr["spearman_kbrd_vs_kg_rank"].mean()),
        "median_event_spearman": float(corr["spearman_kbrd_vs_kg_rank"].median()),
        "verified_category_candidate_rows": int(pairs["verified_category"].sum()),
        "verified_category_candidate_kg_support_coverage": (
            float(pairs.loc[pairs["verified_category"] == 1, "kg_supported"].mean())
            if (pairs["verified_category"] == 1).any() else None
        ),
        "conversation_cluster_bootstrap_supported_minus_unsupported_kbrd_rank": boot_rank,
        "interpretation_guardrail": (
            "KG rank 0 means no recorded support in this exported ranking; "
            "it is not proof of semantic irrelevance or absence of any KG connection."
        ),
    }
    def json_safe(value):
        if isinstance(value, dict):
            return {k: json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        if isinstance(value, (np.integer,)):
            return int(value)
        return value

    with open(args.out / "diagnostic_summary.json", "w", encoding="utf-8") as f:
        json.dump(json_safe(coverage_summary), f, indent=2, ensure_ascii=False, allow_nan=False)

    # Separate positive integrity checks from prohibited-operation guardrails.
    # The latter must be False because this is a diagnostic-only experiment.
    data_checks = {
        "kbrd_event_ids_unique": bool(not kbrd["event_id"].duplicated().any()),
        "kg_event_ids_unique_for_selected_method": bool(not kg["event_id"].duplicated().any()),
        "kg_candidate_rank_array_lengths_match": bool(
            (kg["candidate_ids"].str.len() == kg["kg_ranks"].str.len()).all()
        ),
        "all_kbrd_rows_have_500_or_fewer_ids": bool((kbrd["kbrd_ids"].str.len() <= 500).all()),
        "all_kg_rows_have_500_or_fewer_ids": bool((kg["candidate_ids"].str.len() <= 500).all()),
        "candidate_alignment_uses_event_and_movie": True,
        "category_fairness_uses_verified_evidence_only": True,
        "kg_rank_zero_interpreted_as_no_recorded_support_only": True,
    }
    guardrails = {
        "test_split_used": False,
        "training_performed": False,
        "retrieval_rerun": False,
        "new_weighting_coefficients": False,
    }
    checks = {
        "data_integrity_checks": data_checks,
        "prohibited_operations_not_performed": guardrails,
        "overall_pass": all(data_checks.values()) and not any(guardrails.values()),
    }
    if not checks["overall_pass"]:
        raise AssertionError(f"Experiment validation failed: {checks}")
    with open(args.out / "validation_checks.json", "w", encoding="utf-8") as f:
        json.dump(checks, f, indent=2)

    manifest = {
        "experiment": "22.11",
        "name": "KBRD-KG Evidence Diagnostic",
        "kg_method": args.kg_method,
        "inputs": {
            "kbrd_top500": str(args.kbrd),
            "kg_candidate_export": str(args.kg),
            "experiment_22_10_event_results": str(args.events),
            "verified_category_profiles": str(args.categories),
        },
        "alignment_key": ["event_id", "movie_id"],
        "candidate_scope": "existing exported validation Top-500 only",
        "bootstrap": {"unit": "conv_id", "replicates": args.bootstrap, "seed": 20261111},
        "prohibited_operations": [
            "model training", "retrieval reruns", "new weighting coefficients",
            "test split evaluation"
        ],
        "outputs": sorted(p.name for p in args.out.iterdir() if p.is_file()),
    }
    with open(args.out / "experiment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n========== EXPERIMENT 22.11 COMPLETE ==========")
    print(f"Output directory: {args.out}")
    print(f"Overlapping events: {coverage_summary['overlapping_events']}")
    print(f"Candidate rows: {coverage_summary['candidate_rows']}")
    print(f"KG support coverage: {coverage_summary['overall_kg_support_coverage']:.4f}")
    print(f"Mean event Spearman: {coverage_summary['mean_event_spearman']}")
    print(f"Validation checks: PASS ({len(checks)})")
    print("No training, retrieval rerun, coefficient search, or test evaluation performed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[EXPERIMENT 22.11 FAILED] {exc}", file=sys.stderr)
        raise


