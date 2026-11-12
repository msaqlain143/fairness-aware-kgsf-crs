
#!/usr/bin/env python
"""
Experiment 22.7 — Corrected Category Feasibility Analysis

Uses the same category acceptance rules as Experiment 22.00.
Training interaction frequencies only.
No model inference or test-set evaluation.
"""

import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

PROFILE = Path(
    "results_category_audit/experiment_21_22A/"
    "catalog_item_category_profiles.csv"
)
CATALOG = Path("catalog_audit_corrected.pkl")
OUT = Path("results_corrected/experiment_22_07")

OUT.mkdir(parents=True, exist_ok=True)


def parse_bool(x):
    if pd.isna(x):
        return False
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {"true", "1", "yes"}


def parse_genres(value, delimiter):
    if pd.isna(value):
        return []

    return sorted({
        " ".join(str(x).strip().casefold().split())
        for x in str(value).split(delimiter)
        if str(x).strip()
    })


print("=" * 75)
print("EXPERIMENT 22.7 — CORRECTED CATEGORY FEASIBILITY")
print("=" * 75)

df = pd.read_csv(PROFILE)

with open(CATALOG, "rb") as f:
    catalog = pickle.load(f)

freq = {int(k): int(v) for k, v in catalog["freq_dict"].items()}
catalog_ids = set(freq)

df["movie_entity_id"] = df["movie_entity_id"].astype(int)

if df["movie_entity_id"].duplicated().any():
    raise ValueError("Duplicate movie IDs in category profiles.")

if set(df["movie_entity_id"]) != catalog_ids:
    raise ValueError("Profile and catalog IDs do not match.")

df["training_frequency"] = df["movie_entity_id"].map(freq)

if df["training_frequency"].isna().any():
    raise ValueError("Missing training interaction frequencies.")

# Apply exactly the category acceptance rules used in 22.00.
item_categories = {}

for row in df.to_dict("records"):

    movie_id = int(row["movie_entity_id"])

    if str(row.get("identity_resolution_status", "")) == \
            "AMBIGUOUS_REQUIRES_REVIEW":
        item_categories[movie_id] = []
        continue

    categories = set()

    imdb_conflict = parse_bool(
        row.get("imdb_conflicting_records", False)
    )

    imdb_valid = (
        parse_bool(row.get("imdb_genre_evidence", False))
        and not imdb_conflict
    )

    dbpedia_valid = parse_bool(
        row.get("accepted_genre_evidence", False)
    )

    if imdb_valid:
        categories.update(
            parse_genres(row.get("imdb_genres"), "|")
        )

    if dbpedia_valid:
        categories.update(
            parse_genres(row.get("canonical_genres"), ";")
        )

    item_categories[movie_id] = sorted(categories)

# Attach accepted multi-label categories.
df["verified_categories"] = df["movie_entity_id"].map(item_categories)

df["category_count"] = df["verified_categories"].map(len)

# Interaction classes: retain the existing global definitions.
def interaction_class(n):
    if n == 0:
        return "Zero-shot"
    if n == 1:
        return "One-shot"
    if n <= 4:
        return "Low-interaction"
    return "Established"

df["interaction_class"] = df["training_frequency"].map(
    interaction_class
)

# Expand multi-label assignments.
rows = []

for row in df.itertuples(index=False):

    for category in row.verified_categories:

        rows.append({
            "movie_id": row.movie_entity_id,
            "category": category,
            "training_frequency": row.training_frequency,
            "interaction_class": row.interaction_class,
            "global_popularity_group": catalog["item2group"][
                int(row.movie_entity_id)
            ]
        })

expanded = pd.DataFrame(rows)

if expanded.empty:
    raise ValueError("No verified category assignments found.")

# Category-level item and interaction feasibility.
category_summary = (
    expanded.groupby("category")
    .agg(
        catalog_items=("movie_id", "nunique"),
        zero_shot=("interaction_class",
                   lambda x: (x == "Zero-shot").sum()),
        one_shot=("interaction_class",
                  lambda x: (x == "One-shot").sum()),
        low_interaction=("interaction_class",
                         lambda x: (x == "Low-interaction").sum()),
        established=("interaction_class",
                     lambda x: (x == "Established").sum()),
        mean_training_frequency=("training_frequency", "mean"),
        median_training_frequency=("training_frequency", "median")
    )
    .reset_index()
)

# Calculate within-category popularity quintiles.
expanded["within_category_group"] = pd.Series(
    index=expanded.index, dtype="object"
)

for category, indexes in expanded.groupby("category").groups.items():

    subset = expanded.loc[indexes]

    if subset["movie_id"].nunique() < 5:
        expanded.loc[indexes, "within_category_group"] = (
            "INSUFFICIENT_SUPPORT"
        )
        continue

    # Rank items using training-only interaction frequency.
    ranks = subset["training_frequency"].rank(
        method="first", pct=True
    )

    groups = np.minimum(
        np.ceil(ranks.to_numpy() * 5).astype(int), 5
    )

    expanded.loc[indexes, "within_category_group"] = [
        f"CQ{g}" for g in groups
    ]

# Category x global popularity distribution.
global_groups = (
    expanded.groupby(
        ["category", "global_popularity_group"]
    )
    .size()
    .reset_index(name="item_count")
)

# Category x within-category popularity distribution.
within_groups = (
    expanded.groupby(
        ["category", "within_category_group"]
    )
    .size()
    .reset_index(name="item_count")
)

# Export evidence.
df.to_csv(
    OUT / "item_category_feasibility.csv", index=False
)

expanded.to_csv(
    OUT / "multi_label_category_interactions.csv", index=False
)

category_summary.to_csv(
    OUT / "category_feasibility.csv", index=False
)

global_groups.to_csv(
    OUT / "global_popularity_by_category.csv", index=False
)

within_groups.to_csv(
    OUT / "within_category_popularity_groups.csv", index=False
)

# Summary and integrity checks.
summary = {
    "experiment": "22.07_corrected_category_feasibility",
    "catalog_items": len(catalog_ids),
    "categorized_items": int(
        df["category_count"].gt(0).sum()
    ),
    "uncategorized_items": int(
        df["category_count"].eq(0).sum()
    ),
    "total_category_assignments": len(expanded),
    "distinct_verified_categories": int(
        expanded["category"].nunique()
    ),
    "multi_label_items": int(
        df["category_count"].gt(1).sum()
    ),
    "training_frequency_alignment": bool(
        df["training_frequency"].notna().all()
    ),
    "test_split_evaluated": False,
    "category_source_rules": {
        "IMDb": "imdb_genre_evidence AND NOT imdb_conflicting_records",
        "DBpedia": "accepted_genre_evidence",
        "ambiguous_identity": "withheld"
    }
}

import json

(OUT / "category_feasibility_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\n========== CATEGORY SUMMARY ==========")
print(category_summary.to_string(index=False))

print("\n========== FINAL COUNTS ==========")
print(json.dumps(summary, indent=2))

print("\nOutputs:", OUT.resolve())
print("TEST SPLIT EVALUATED: NO")
print("\nEXPERIMENT 22.7 COMPLETE")
