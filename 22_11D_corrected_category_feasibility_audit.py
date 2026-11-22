#!/usr/bin/env python
"""Experiment 22.11D — Corrected Category Feasibility Audit.

Uses 22.11C reconciled evidence and catalog_audit_corrected.pkl.
Primary: strict DBpedia ACCEPTED_GENRE only.
Sensitivity: strict+hybrid DBpedia; strict+external IMDb (kept separate).
Unvalidated evidence and ambiguous identities are excluded.
Training frequencies only; no training, retrieval, reranking, tuning, or test data.
Support thresholds are descriptive screens, not formal power calculations.
"""
from __future__ import annotations
import argparse, hashlib, json, pickle, platform
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED_N = 6924
GROUPS = [0, 1, 2, 3, 4]
THRESHOLDS = [5, 10, 20, 30, 50]
CLASSES = ["Zero-shot", "One-shot", "Low-interaction", "Established"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))


def require(df, cols, label):
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def categories(value):
    if pd.isna(value):
        return []
    return sorted({" ".join(x.strip().casefold().split()) for x in str(value).replace("|", ";").split(";") if x.strip()})


def interaction_class(n):
    if n == 0: return "Zero-shot"
    if n == 1: return "One-shot"
    if n <= 4: return "Low-interaction"
    return "Established"


def within_quintiles(frame):
    ordered = frame.sort_values(["training_frequency", "movie_id"], kind="mergesort")
    n = len(ordered)
    if n < 5:
        return pd.Series(pd.NA, index=frame.index, dtype="Int64")
    labels = np.minimum((np.arange(n, dtype=np.int64) * 5) // n, 4)
    return pd.Series(labels, index=ordered.index, dtype="Int64").reindex(frame.index)


def evidence_populations(ev):
    ev = ev.copy()
    ev["movie_entity_id"] = pd.to_numeric(ev.movie_entity_id, errors="raise").astype(int)
    ev["source_norm"] = ev.source.astype(str).str.strip().str.casefold()
    ev["type_norm"] = ev.evidence_type.astype(str).str.strip().str.upper()
    ev["identity_status"] = ev.identity_status.fillna("").astype(str)
    ev = ev[ev.identity_status.ne("AMBIGUOUS_REQUIRES_REVIEW")]
    rows = []
    for r in ev.itertuples(index=False):
        for cat in categories(r.category):
            rows.append({"movie_id": int(r.movie_entity_id), "category": cat,
                         "source": r.source_norm, "evidence_type": r.type_norm})
    norm = pd.DataFrame(rows, columns=["movie_id", "category", "source", "evidence_type"])
    norm = norm.drop_duplicates(["movie_id", "category", "source", "evidence_type"])
    strict = norm[(norm.source == "dbpedia") & (norm.evidence_type == "ACCEPTED_GENRE")].copy()
    hybrid = norm[(norm.source == "dbpedia") & (norm.evidence_type == "ACCEPTED_HYBRID")].copy()
    imdb = norm[norm.source == "imdb"].copy()
    return {
        "strict_dbpedia_primary": strict.drop_duplicates(["movie_id", "category"]),
        "strict_plus_hybrid_sensitivity": pd.concat([strict, hybrid]).drop_duplicates(["movie_id", "category"]),
        "strict_plus_imdb_sensitivity": pd.concat([strict, imdb]).drop_duplicates(["movie_id", "category"]),
        "imdb_external_only_report": imdb.drop_duplicates(["movie_id", "category"]),
    }


def analyse(name, frame, freq, global_group, catalog_n):
    frame = frame.drop_duplicates(["movie_id", "category"]).copy()
    frame["training_frequency"] = frame.movie_id.map(freq)
    frame["interaction_class"] = frame.training_frequency.map(interaction_class)
    frame["global_popularity_group"] = frame.movie_id.map(global_group)
    if frame[["training_frequency", "global_popularity_group"]].isna().any().any():
        raise RuntimeError(f"{name}: missing catalog frequency/group mapping")
    cat_rows, group_rows, support_rows = [], [], []
    for cat, sub in frame.groupby("category", sort=True):
        sub = sub.drop_duplicates("movie_id")
        n = int(sub.movie_id.nunique())
        counts = sub.interaction_class.value_counts()
        local_groups = within_quintiles(sub)
        cat_rows.append({
            "scenario": name, "category": cat, "catalog_items": n,
            "catalog_share_full_catalog": n / catalog_n,
            "zero_shot": int(counts.get("Zero-shot", 0)),
            "one_shot": int(counts.get("One-shot", 0)),
            "low_interaction": int(counts.get("Low-interaction", 0)),
            "established": int(counts.get("Established", 0)),
            "zero_plus_one_shot": int(counts.get("Zero-shot", 0) + counts.get("One-shot", 0)),
            "mean_training_frequency": float(sub.training_frequency.mean()),
            "median_training_frequency": float(sub.training_frequency.median()),
            "has_all_four_interaction_classes": all(counts.get(c, 0) > 0 for c in CLASSES),
            "within_category_quintiles_defined": n >= 5,
        })
        tmp = sub.copy()
        tmp["within_category_group"] = local_groups
        for grouping, col in [("global", "global_popularity_group"), ("within_category", "within_category_group")]:
            for g in GROUPS:
                part = tmp[tmp[col] == g]
                group_rows.append({
                    "scenario": name, "category": cat, "grouping": grouping,
                    "popularity_group": g, "items": int(part.movie_id.nunique()),
                    "category_item_share": float(part.movie_id.nunique()/n),
                    "zero_shot": int((part.interaction_class == "Zero-shot").sum()),
                    "one_shot": int((part.interaction_class == "One-shot").sum()),
                    "low_interaction": int((part.interaction_class == "Low-interaction").sum()),
                    "established": int((part.interaction_class == "Established").sum()),
                })
        group_counts = tmp.dropna(subset=["within_category_group"]).groupby("within_category_group").movie_id.nunique().to_dict()
        nonempty = sum(v > 0 for v in group_counts.values())
        min_group = int(min(group_counts.values())) if group_counts else 0
        for threshold in THRESHOLDS:
            support_rows.append({
                "scenario": name, "category": cat, "catalog_items": n,
                "minimum_category_items_threshold": threshold,
                "passes_item_count_threshold": n >= threshold,
                "nonempty_within_category_groups": nonempty,
                "minimum_items_in_any_within_category_group": min_group,
                "passes_five_group_representation": nonempty == 5,
                "passes_minimum_5_items_per_group": min_group >= 5,
                "passes_combined_feasibility_screen": n >= threshold and nonempty == 5,
            })
    return pd.DataFrame(cat_rows), pd.DataFrame(group_rows), pd.DataFrame(support_rows), frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--profiles", type=Path, default=Path("results_corrected/experiment_22_11C/reconciled_catalog_category_profiles.csv"))
    ap.add_argument("--evidence", type=Path, default=Path("results_corrected/experiment_22_11C/reconciled_item_category_evidence.csv"))
    ap.add_argument("--catalog", type=Path, default=Path("catalog_audit_corrected.pkl"))
    ap.add_argument("--out", type=Path, default=Path("results_corrected/experiment_22_11D"))
    a = ap.parse_args()
    root = a.root.resolve()
    def resolve(p): return p if p.is_absolute() else root / p
    profile_path, evidence_path, catalog_path, out = map(resolve, [a.profiles, a.evidence, a.catalog, a.out])
    for p in [profile_path, evidence_path, catalog_path]:
        if not p.is_file(): raise FileNotFoundError(p)
    out.mkdir(parents=True, exist_ok=True)

    profiles, ev = pd.read_csv(profile_path), pd.read_csv(evidence_path)
    require(profiles, {"movie_entity_id", "identity_eligible", "strict_category_fairness_eligible"}, "22.11C profiles")
    require(ev, {"movie_entity_id", "category", "source", "evidence_type", "identity_status"}, "22.11C evidence")
    profiles.movie_entity_id = pd.to_numeric(profiles.movie_entity_id, errors="raise").astype(int)
    if profiles.movie_entity_id.duplicated().any(): raise RuntimeError("Duplicate profile item IDs")

    with catalog_path.open("rb") as f: catalog = pickle.load(f)
    if not isinstance(catalog, dict) or not {"freq_dict", "item2group"}.issubset(catalog):
        raise RuntimeError("Catalog pickle lacks freq_dict/item2group")
    freq = {int(k): int(v) for k, v in catalog["freq_dict"].items()}
    global_group = {int(k): int(v) for k, v in catalog["item2group"].items()}
    catalog_ids = set(freq)
    if len(catalog_ids) != EXPECTED_N: raise RuntimeError(f"Expected {EXPECTED_N} catalog items; got {len(catalog_ids)}")
    if set(profiles.movie_entity_id) != catalog_ids: raise RuntimeError("22.11C profile IDs do not match catalog")
    if set(global_group) != catalog_ids: raise RuntimeError("item2group IDs do not match catalog")
    if set(global_group.values()) != set(GROUPS): raise RuntimeError("Expected global popularity groups 0–4")
    if min(freq.values()) < 0: raise RuntimeError("Negative training frequency")
    freq_definition = catalog.get("frequency_definition", "")
    if freq_definition != "Number of movie targets in train Recommender-role utterances":
        raise RuntimeError(f"Unexpected frequency definition: {freq_definition!r}")

    populations = evidence_populations(ev)
    population_rows, category_frames, group_frames, support_frames, assignment_frames = [], [], [], [], []
    ambiguous_ids = set(ev.loc[ev.identity_status.eq("AMBIGUOUS_REQUIRES_REVIEW"), "movie_entity_id"].astype(int))
    for name, frame in populations.items():
        frame = frame[frame.movie_id.isin(catalog_ids)].drop_duplicates(["movie_id", "category"])
        role = "primary" if name == "strict_dbpedia_primary" else ("separate_external_source_report" if name == "imdb_external_only_report" else "sensitivity_only")
        covered = set(frame.movie_id.astype(int))
        population_rows.append({
            "scenario": name, "evidence_role": role, "unique_categorized_items": len(covered),
            "unique_category_assignments": len(frame), "uncategorized_catalog_items": len(catalog_ids-covered),
            "catalog_coverage_rate": len(covered)/len(catalog_ids), "unique_categories": int(frame.category.nunique()),
        })
        if frame.empty: continue
        c, g, s, enriched = analyse(name, frame, freq, global_group, len(catalog_ids))
        category_frames.append(c); group_frames.append(g); support_frames.append(s)
        enriched["scenario"] = name; assignment_frames.append(enriched)

    population = pd.DataFrame(population_rows)
    category = pd.concat(category_frames, ignore_index=True)
    groups = pd.concat(group_frames, ignore_index=True)
    support = pd.concat(support_frames, ignore_index=True)
    assignments = pd.concat(assignment_frames, ignore_index=True)
    threshold_summary = support.groupby(["scenario", "minimum_category_items_threshold"]).agg(
        categories_total=("category", "nunique"),
        categories_passing_item_count=("passes_item_count_threshold", "sum"),
        categories_with_five_nonempty_groups=("passes_five_group_representation", "sum"),
        categories_passing_combined_screen=("passes_combined_feasibility_screen", "sum"),
        categories_with_at_least_5_items_per_group=("passes_minimum_5_items_per_group", "sum"),
    ).reset_index()
    missing_rows = []
    for name, frame in populations.items():
        covered = set(frame.movie_id.astype(int))
        for mid in sorted(catalog_ids-covered):
            n = freq[mid]
            missing_rows.append({"scenario": name, "movie_id": mid, "training_frequency": n,
                                 "interaction_class": interaction_class(n), "global_popularity_group": global_group[mid]})
    missing = pd.DataFrame(missing_rows)

    strict = populations["strict_dbpedia_primary"]
    strict_ids = set(strict.movie_id.astype(int))
    hybrid_sens = populations["strict_plus_hybrid_sensitivity"]
    imdb_sens = populations["strict_plus_imdb_sensitivity"]
    checks = {
        "experiment": "22.11D",
        "catalog_size_6924": len(catalog_ids) == EXPECTED_N,
        "profile_catalog_id_alignment": set(profiles.movie_entity_id) == catalog_ids,
        "global_popularity_groups_0_to_4": set(global_group.values()) == set(GROUPS),
        "training_frequency_definition_preserved": freq_definition == "Number of movie targets in train Recommender-role utterances",
        "strict_primary_is_dbpedia_accepted_genre_only": strict.empty or (set(strict.source) == {"dbpedia"} and set(strict.evidence_type) == {"ACCEPTED_GENRE"}),
        "hybrid_is_sensitivity_only": set(hybrid_sens.movie_id.astype(int)).issuperset(strict_ids),
        "imdb_is_separately_identified": imdb_sens.empty or "imdb" in set(imdb_sens.source),
        "ambiguous_identity_excluded": not (set(assignments.movie_id.astype(int)) & ambiguous_ids),
        "unvalidated_dbpedia_excluded": not (set(assignments.loc[(assignments.source == "dbpedia") & (assignments.evidence_type == "UNVALIDATED_CATEGORY"), "movie_id"].astype(int)) if "source" in assignments else set()),
        "no_training": True, "no_retrieval": True, "no_reranking": True,
        "no_parameter_tuning": True, "independent_test_not_read": True,
        "22_11C_inputs_untouched": True,
        "outputs_nonempty": not category.empty and not population.empty,
    }
    checks["strict_unique_movie_count_matches_22_11C"] = len(strict_ids) == 14
    checks["overall_pass"] = all(bool(v) for k, v in checks.items() if isinstance(v, (bool, np.bool_)) and k != "overall_pass")

    outputs = {
        "category_evidence_population_summary.csv": population,
        "category_interaction_distribution.csv": category,
        "category_popularity_distribution.csv": groups,
        "global_vs_category_popularity_comparison.csv": groups,
        "category_statistical_support.csv": support,
        "category_support_threshold_summary.csv": threshold_summary,
        "missing_category_item_register.csv": missing,
        "scenario_item_category_assignments.csv": assignments,
    }
    for name, df in outputs.items(): df.to_csv(out/name, index=False)
    save_json(out/"validation_checks.json", checks)
    report = {
        "experiment": "22.11D_corrected_category_feasibility_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "numpy": np.__version__, "pandas": pd.__version__, "catalog_items": len(catalog_ids),
        "frequency_definition": freq_definition,
        "quintile_definition": catalog.get("quintile_definition", ""),
        "interaction_classes": {"zero_shot": 0, "one_shot": 1, "low_interaction": "2–4", "established": ">=5"},
        "support_thresholds": THRESHOLDS,
        "support_thresholds_are_descriptive_not_formal_power": True,
        "category_quintile_method": "rank-based five groups sorted by (training_frequency, movie_id); global item2group preserved",
        "evidence_policy": {"strict_dbpedia": "primary", "hybrid_dbpedia": "sensitivity only", "IMDb": "separate external-source sensitivity", "unvalidated": "excluded", "ambiguous_identity": "excluded"},
        "input_sha256": {str(p): sha256(p) for p in [profile_path, evidence_path, catalog_path]},
        "outputs": list(outputs)+["validation_checks.json", "category_feasibility_summary.json", "experiment_manifest.json"],
        "population_summary": population.to_dict("records"), "support_threshold_summary": threshold_summary.to_dict("records"),
        "validation": checks, "training_performed": False, "retrieval_performed": False,
        "reranking_performed": False, "parameter_tuning_performed": False, "independent_test_read": False,
    }
    save_json(out/"category_feasibility_summary.json", report)
    manifest = {"experiment": "22.11D", "name": "Corrected Category Feasibility Audit",
                "created_utc": report["created_utc"], "inputs": {"profiles": str(profile_path), "evidence": str(evidence_path), "catalog": str(catalog_path)},
                "input_sha256": report["input_sha256"], "scope": "frozen 22.11C evidence and training-only catalog frequencies",
                "outputs": report["outputs"], "validation": checks}
    save_json(out/"experiment_manifest.json", manifest)

    print("\n" + "="*78 + "\n22.11D COMPLETE\n" + "="*78)
    print("\nEVIDENCE POPULATIONS\n", population.to_string(index=False))
    print("\nPRIMARY CATEGORY DISTRIBUTION\n", category[category.scenario == "strict_dbpedia_primary"].to_string(index=False))
    print("\nSUPPORT THRESHOLD SUMMARY\n", threshold_summary.to_string(index=False))
    print("\nVALIDATION\n", json.dumps(checks, indent=2))
    print("\nOutput:", out, "\nOVERALL PASS:", checks["overall_pass"])
    if not checks["overall_pass"]: raise RuntimeError("22.11D validation failed; inspect validation_checks.json")

if __name__ == "__main__": main()
