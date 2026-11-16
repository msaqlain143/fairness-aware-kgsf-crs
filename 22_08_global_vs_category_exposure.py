
#!/usr/bin/env python
"""
KG-FairCRS — Experiment 22.08
Global vs Category-Aware Exposure Baseline

Purpose:
1. Measure frozen KGSF recommendation exposure.
2. Compare global and within-category popularity groups.
3. Evaluate exposure at K=10, 20, 50.
4. Identify zero-exposure catalog items.
5. Perform category support threshold sensitivity.

Data restrictions:
- Training frequencies only for popularity groups.
- Verified genre evidence only.
- Independent-test predictions only for exposure.
- No model training or parameter optimization.
- No use of test labels for popularity grouping.

Existing experiments are not modified.
"""

from pathlib import Path
from collections import defaultdict
import pickle
import json
import math

import numpy as np
import pandas as pd


# =========================================================
# 1. CONFIGURATION
# =========================================================

ROOT = Path("results_final_test_m2")

SCORE_FILE = (
    ROOT / "experiment_22_27A"
    / "kgsf_independent_test_full_scores.npz"
)

MANIFEST_FILE = (
    ROOT / "experiment_22_27A"
    / "kgsf_independent_test_event_manifest.csv"
)

PROFILE_FILE = (
    Path("results_category_audit")
    / "experiment_21_22A"
    / "catalog_item_category_profiles.csv"
)

CATALOG_FILE = Path("catalog_audit_corrected.pkl")

OUT = Path("results_corrected/experiment_22_08")

TOP_K_VALUES = [10, 20, 50]

CATEGORY_THRESHOLDS = [10, 20, 30, 50]

PRIMARY_CATEGORY_THRESHOLD = 20

EXPECTED_EVENTS = 4003
EXPECTED_CATALOG = 6924

OUT.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def parse_bool(value):

    if pd.isna(value):
        return False

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    return str(value).strip().lower() in {
        "true", "1", "yes"
    }


def parse_genres(value, delimiter):

    if pd.isna(value):
        return []

    return sorted({
        " ".join(
            str(item).strip().casefold().split()
        )
        for item in str(value).split(delimiter)
        if str(item).strip()
    })


def interaction_class(n):

    if n == 0:
        return "Zero-shot"

    if n == 1:
        return "One-shot"

    if n <= 4:
        return "Low-interaction"

    return "Established"


def popularity_groups(movie_ids, frequencies, prefix):

    """
    Five popularity groups using average percentile ranks.

    Equal-frequency items receive equal ranks and cannot
    be arbitrarily split because of row ordering.

    Group 1 = lower interaction frequency.
    Group 5 = higher interaction frequency.
    """

    frame = pd.DataFrame({
        "movie_id": movie_ids,
        "frequency": frequencies
    })

    ranks = frame["frequency"].rank(
        method="average",
        pct=True
    )

    group_numbers = np.minimum(
        np.ceil(ranks.to_numpy() * 5).astype(int),
        5
    )

    frame["popularity_group"] = [
        f"{prefix}{group}"
        for group in group_numbers
    ]

    return frame


def calculate_metrics(
    exposure,
    group_frame,
    group_column,
    k,
    category=None,
    threshold=None
):

    """
    Calculate catalog share, exposure share and
    exposure ratio for popularity groups.
    """

    data = group_frame.copy()

    data["exposure"] = data["movie_id"].map(
        exposure
    ).fillna(0.0)

    total_items = len(data)
    total_exposure = data["exposure"].sum()

    results = []

    for group, subset in data.groupby(
        group_column,
        sort=True
    ):

        catalog_share = len(subset) / total_items

        group_exposure = subset["exposure"].sum()

        exposure_share = (
            group_exposure / total_exposure
            if total_exposure > 0 else 0.0
        )

        exposure_ratio = (
            exposure_share / catalog_share
            if catalog_share > 0 else np.nan
        )

        results.append({

            "category": category,
            "threshold": threshold,
            "K": k,
            "popularity_group": group,

            "catalog_items": len(subset),

            "catalog_share": catalog_share,

            "exposure": group_exposure,

            "exposure_share": exposure_share,

            "exposure_ratio": exposure_ratio,

            "zero_exposure_items": int(
                (subset["exposure"] == 0).sum()
            ),

            "mean_training_frequency": (
                subset["training_frequency"].mean()
            )

        })

    return results


# =========================================================
# 3. INPUT VALIDATION
# =========================================================

print("=" * 85)
print("EXPERIMENT 22.08 — GLOBAL VS CATEGORY EXPOSURE")
print("=" * 85)

for path in [
    SCORE_FILE,
    MANIFEST_FILE,
    PROFILE_FILE,
    CATALOG_FILE
]:

    if not path.is_file():
        raise FileNotFoundError(path)

    print("FOUND:", path)


# =========================================================
# 4. LOAD FROZEN CATALOG
# =========================================================

print("\n[1] LOADING FROZEN CATALOG")

with open(CATALOG_FILE, "rb") as f:
    catalog = pickle.load(f)

freq = {
    int(movie_id): int(count)
    for movie_id, count in catalog["freq_dict"].items()
}

catalog_ids = set(freq)

if len(catalog_ids) != EXPECTED_CATALOG:
    raise RuntimeError(
        f"Expected {EXPECTED_CATALOG} catalog items, "
        f"found {len(catalog_ids)}"
    )

if any(value < 0 for value in freq.values()):
    raise RuntimeError("Negative training frequency detected.")

print("Catalog items:", len(catalog_ids))

print(
    "Interaction classes:",
    dict(
        pd.Series([
            interaction_class(freq[movie])
            for movie in catalog_ids
        ]).value_counts()
    )
)


# =========================================================
# 5. LOAD VERIFIED CATEGORY EVIDENCE
# =========================================================

print("\n[2] LOADING VERIFIED CATEGORY EVIDENCE")

profiles = pd.read_csv(PROFILE_FILE)

required_profile_columns = {
    "movie_entity_id",
    "identity_resolution_status",
    "imdb_conflicting_records",
    "imdb_genre_evidence",
    "imdb_genres",
    "accepted_genre_evidence",
    "canonical_genres"
}

missing = required_profile_columns - set(profiles.columns)

if missing:
    raise ValueError(
        f"Missing category profile columns: {missing}"
    )

profiles["movie_entity_id"] = (
    profiles["movie_entity_id"].astype(int)
)

if profiles["movie_entity_id"].duplicated().any():
    raise RuntimeError("Duplicate category profile IDs.")

if set(profiles["movie_entity_id"]) != catalog_ids:
    raise RuntimeError(
        "Category profile and catalog IDs do not match."
    )

item_categories = {}

for row in profiles.to_dict("records"):

    movie_id = int(row["movie_entity_id"])

    if str(
        row.get("identity_resolution_status", "")
    ) == "AMBIGUOUS_REQUIRES_REVIEW":

        item_categories[movie_id] = []
        continue

    categories = set()

    imdb_conflict = parse_bool(
        row.get("imdb_conflicting_records", False)
    )

    imdb_valid = (
        parse_bool(
            row.get("imdb_genre_evidence", False)
        )
        and not imdb_conflict
    )

    dbpedia_valid = parse_bool(
        row.get("accepted_genre_evidence", False)
    )

    if imdb_valid:

        categories.update(
            parse_genres(
                row.get("imdb_genres"),
                "|"
            )
        )

    if dbpedia_valid:

        categories.update(
            parse_genres(
                row.get("canonical_genres"),
                ";"
            )
        )

    item_categories[movie_id] = sorted(categories)


categorized_items = sum(
    len(categories) > 0
    for categories in item_categories.values()
)

print("Categorized movies:", categorized_items)

print(
    "Uncategorized movies:",
    len(catalog_ids) - categorized_items
)

print(
    "Verified categories:",
    len({
        category
        for categories in item_categories.values()
        for category in categories
    })
)


# =========================================================
# 6. CREATE GLOBAL POPULARITY GROUPS
# =========================================================

print("\n[3] CONSTRUCTING GLOBAL POPULARITY GROUPS")

catalog_order = sorted(catalog_ids)

global_groups = popularity_groups(
    catalog_order,
    [freq[movie] for movie in catalog_order],
    "G"
)

global_groups["training_frequency"] = (
    global_groups["movie_id"].map(freq)
)

global_groups["interaction_class"] = (
    global_groups["training_frequency"].map(
        interaction_class
    )
)

global_groups.to_csv(
    OUT / "global_popularity_groups.csv",
    index=False
)

print(
    global_groups["popularity_group"]
    .value_counts()
    .sort_index()
    .to_string()
)


# =========================================================
# 7. CREATE WITHIN-CATEGORY POPULARITY GROUPS
# =========================================================

print("\n[4] CONSTRUCTING WITHIN-CATEGORY GROUPS")

category_records = []

for category in sorted({
    category
    for categories in item_categories.values()
    for category in categories
}):

    movie_list = sorted([
        movie
        for movie, categories in item_categories.items()
        if category in categories
    ])

    group_frame = popularity_groups(
        movie_list,
        [freq[movie] for movie in movie_list],
        "CQ"
    )

    group_frame["category"] = category

    group_frame["training_frequency"] = (
        group_frame["movie_id"].map(freq)
    )

    group_frame["interaction_class"] = (
        group_frame["training_frequency"].map(
            interaction_class
        )
    )

    group_frame["category_size"] = len(movie_list)

    category_records.append(group_frame)


if not category_records:
    raise RuntimeError("No verified category assignments.")

within_groups = pd.concat(
    category_records,
    ignore_index=True
)

within_groups.to_csv(
    OUT / "within_category_popularity_groups.csv",
    index=False
)

category_sizes = (
    within_groups.groupby("category")["movie_id"]
    .nunique()
    .sort_values(ascending=False)
)

print("\nCategory sizes:")
print(category_sizes.to_string())


# =========================================================
# 8. LOAD FROZEN KGSF SCORES
# =========================================================

print("\n[5] LOADING FROZEN KGSF SCORES")

with np.load(
    SCORE_FILE,
    allow_pickle=False
) as data:

    scores = data["scores"]
    movie_ids = data["movie_ids"].astype(np.int64)
    event_ids = data["event_ids"].astype(np.int64)

print("Score matrix:", scores.shape)
print("Score dtype:", scores.dtype)

if scores.shape != (
    EXPECTED_EVENTS,
    EXPECTED_CATALOG
):

    raise RuntimeError(
        f"Unexpected score matrix shape: {scores.shape}"
    )

if len(movie_ids) != EXPECTED_CATALOG:
    raise RuntimeError("Invalid movie ID array.")

if len(event_ids) != EXPECTED_EVENTS:
    raise RuntimeError("Invalid event ID array.")

if len(np.unique(movie_ids)) != len(movie_ids):
    raise RuntimeError("Duplicate movie IDs in score file.")

if len(np.unique(event_ids)) != len(event_ids):
    raise RuntimeError("Duplicate event IDs in score file.")

if set(movie_ids.tolist()) != catalog_ids:
    raise RuntimeError(
        "Score catalog and frozen training catalog differ."
    )

if not np.isfinite(scores).all():
    raise RuntimeError("Non-finite recommendation scores.")

print("Finite scores: PASS")
print("Catalog alignment: PASS")


# =========================================================
# 9. VALIDATE EVENT MANIFEST
# =========================================================

print("\n[6] VALIDATING EVENT MANIFEST")

manifest = pd.read_csv(MANIFEST_FILE)

required_manifest_columns = {
    "event_id",
    "score_row",
    "target_movie_id",
    "conv_id"
}

missing = (
    required_manifest_columns - set(manifest.columns)
)

if missing:
    raise ValueError(
        f"Missing manifest columns: {missing}"
    )

if len(manifest) != EXPECTED_EVENTS:
    raise RuntimeError("Manifest event count mismatch.")

if manifest["event_id"].duplicated().any():
    raise RuntimeError("Duplicate manifest events.")

if manifest["score_row"].duplicated().any():
    raise RuntimeError("Duplicate score rows.")

manifest = manifest.sort_values(
    "score_row"
).reset_index(drop=True)

expected_rows = np.arange(EXPECTED_EVENTS)

if not np.array_equal(
    manifest["score_row"].to_numpy(),
    expected_rows
):

    raise RuntimeError(
        "Score rows are not a complete zero-based sequence."
    )

if not np.array_equal(
    manifest["event_id"].to_numpy(dtype=np.int64),
    event_ids
):

    raise RuntimeError(
        "Manifest and score matrix event alignment failed."
    )

print("Events:", len(manifest))
print("Unique conversations:", manifest["conv_id"].nunique())
print("Event alignment: PASS")


# =========================================================
# 10. COMPUTE TOP-50 POSITIONAL EXPOSURE
# =========================================================

print("\n[7] COMPUTING RECOMMENDATION EXPOSURE")

max_k = max(TOP_K_VALUES)

exposure = {
    k: np.zeros(
        EXPECTED_CATALOG,
        dtype=np.float64
    )
    for k in TOP_K_VALUES
}

discounts = 1.0 / np.log2(
    np.arange(1, max_k + 1) + 1
)

for event_row in range(EXPECTED_EVENTS):

    # Deterministic ranking:
    # descending score, ascending movie ID for ties.

    ranking = np.lexsort((
        movie_ids,
        -scores[event_row]
    ))

    top_items = ranking[:max_k]

    for k in TOP_K_VALUES:

        selected = top_items[:k]

        np.add.at(
            exposure[k],
            selected,
            discounts[:k]
        )

    if (event_row + 1) % 500 == 0:

        print(
            f"Processed {event_row + 1}/"
            f"{EXPECTED_EVENTS} events"
        )


print("Exposure calculation: COMPLETE")


# =========================================================
# 11. MAP EXPOSURE TO MOVIE IDS
# =========================================================

print("\n[8] MAPPING EXPOSURE TO CATALOG ITEMS")

exposure_maps = {}

for k in TOP_K_VALUES:

    exposure_maps[k] = dict(
        zip(
            movie_ids.tolist(),
            exposure[k].tolist()
        )
    )

    if not np.isfinite(exposure[k]).all():
        raise RuntimeError("Invalid exposure values.")

    print(
        f"K={k} | Total exposure="
        f"{exposure[k].sum():.4f} | "
        f"Exposed items="
        f"{np.count_nonzero(exposure[k]):,}"
    )


# =========================================================
# 12. GLOBAL EXPOSURE METRICS
# =========================================================

print("\n[9] GLOBAL EXPOSURE METRICS")

global_metrics = []

for k in TOP_K_VALUES:

    global_metrics.extend(
        calculate_metrics(
            exposure_maps[k],
            global_groups,
            "popularity_group",
            k
        )
    )

global_metrics = pd.DataFrame(global_metrics)

global_metrics.to_csv(
    OUT / "global_exposure_metrics.csv",
    index=False
)


# =========================================================
# 13. CATEGORY EXPOSURE METRICS
# =========================================================

print("\n[10] CATEGORY EXPOSURE METRICS")

category_metrics = []

for category, subset in within_groups.groupby(
    "category",
    sort=True
):

    category_size = subset["movie_id"].nunique()

    if category_size < PRIMARY_CATEGORY_THRESHOLD:
        continue

    for k in TOP_K_VALUES:

        category_metrics.extend(
            calculate_metrics(
                exposure_maps[k],
                subset,
                "popularity_group",
                k,
                category=category
            )
        )

category_metrics = pd.DataFrame(category_metrics)

category_metrics.to_csv(
    OUT / "category_exposure_metrics.csv",
    index=False
)

print(
    "Primary eligible categories:",
    category_metrics["category"].nunique()
)


# =========================================================
# 14. GLOBAL VS CATEGORY EXPOSURE COMPARISON
# =========================================================

print("\n[11] EXPOSURE COMPARISON")

comparison = []

for k in TOP_K_VALUES:

    global_subset = global_metrics[
        global_metrics["K"] == k
    ]

    for _, row in global_subset.iterrows():

        comparison.append({

            "K": k,

            "analysis": "GLOBAL",

            "category": None,

            "popularity_group": row["popularity_group"],

            "catalog_share": row["catalog_share"],

            "exposure_share": row["exposure_share"],

            "exposure_ratio": row["exposure_ratio"]

        })

    category_subset = category_metrics[
        category_metrics["K"] == k
    ]

    for _, row in category_subset.iterrows():

        comparison.append({

            "K": k,

            "analysis": "CATEGORY",

            "category": row["category"],

            "popularity_group": row["popularity_group"],

            "catalog_share": row["catalog_share"],

            "exposure_share": row["exposure_share"],

            "exposure_ratio": row["exposure_ratio"]

        })


comparison = pd.DataFrame(comparison)

comparison.to_csv(
    OUT / "exposure_comparison.csv",
    index=False
)


# =========================================================
# 15. ZERO-EXPOSURE ITEMS
# =========================================================

print("\n[12] ZERO-EXPOSURE ANALYSIS")

zero_exposure_rows = []

category_lookup = defaultdict(list)

for movie, categories in item_categories.items():

    for category in categories:
        category_lookup[movie].append(category)


for k in TOP_K_VALUES:

    for _, row in global_groups.iterrows():

        movie = int(row["movie_id"])

        value = exposure_maps[k][movie]

        if value == 0:

            zero_exposure_rows.append({

                "K": k,

                "movie_id": movie,

                "training_frequency": int(freq[movie]),

                "interaction_class": interaction_class(
                    freq[movie]
                ),

                "global_popularity_group": row[
                    "popularity_group"
                ],

                "verified_categories": json.dumps(
                    category_lookup[movie]
                ),

                "exposure": value

            })


zero_exposure = pd.DataFrame(zero_exposure_rows)

zero_exposure.to_csv(
    OUT / "zero_exposure_items.csv",
    index=False
)


# =========================================================
# 16. CATEGORY THRESHOLD SENSITIVITY
# =========================================================

print("\n[13] CATEGORY THRESHOLD SENSITIVITY")

threshold_rows = []

for threshold in CATEGORY_THRESHOLDS:

    eligible_categories = [
        category
        for category, size in category_sizes.items()
        if size >= threshold
    ]

    eligible = within_groups[
        within_groups["category"].isin(
            eligible_categories
        )
    ]

    for category, subset in eligible.groupby(
        "category",
        sort=True
    ):

        for k in TOP_K_VALUES:

            threshold_rows.extend(
                calculate_metrics(
                    exposure_maps[k],
                    subset,
                    "popularity_group",
                    k,
                    category=category,
                    threshold=threshold
                )
            )

    print(
        f"Minimum {threshold} | "
        f"Categories={len(eligible_categories)} | "
        f"Assignments={len(eligible)}"
    )


threshold_sensitivity = pd.DataFrame(
    threshold_rows
)

threshold_sensitivity.to_csv(
    OUT / "threshold_sensitivity.csv",
    index=False
)


# =========================================================
# 17. VALIDATION CHECKS
# =========================================================

print("\n[14] FINAL VALIDATION")

checks = {

    "experiment": "22_08",

    "split": "independent_test",

    "events": EXPECTED_EVENTS,

    "catalog_items": EXPECTED_CATALOG,

    "score_matrix_shape": list(scores.shape),

    "unique_event_ids": int(
        manifest.event_id.nunique()
    ),

    "unique_catalog_movie_ids": int(
        len(np.unique(movie_ids))
    ),

    "finite_scores": bool(
        np.isfinite(scores).all()
    ),

    "event_alignment": True,

    "catalog_alignment": True,

    "category_profile_alignment": True,

    "training_frequency_only": True,

    "verified_categories_only": True,

    "popularity_tie_method": "average",

    "ranking_tie_breaking": (
        "descending_score_then_ascending_movie_id"
    ),

    "top_k_values": TOP_K_VALUES,

    "category_thresholds": CATEGORY_THRESHOLDS,

    "primary_category_threshold": (
        PRIMARY_CATEGORY_THRESHOLD
    ),

    "training_performed": False,

    "parameter_search_performed": False,

    "test_labels_used_for_popularity": False,

    "overall_pass": True

}

for k in TOP_K_VALUES:

    checks[f"K{k}_exposure_finite"] = bool(
        np.isfinite(exposure[k]).all()
    )

    checks[f"K{k}_total_exposure_positive"] = bool(
        exposure[k].sum() > 0
    )

    checks[f"K{k}_zero_exposure_count"] = int(
        np.sum(exposure[k] == 0)
    )


checks["global_metrics_nonempty"] = bool(
    not global_metrics.empty
)

checks["category_metrics_nonempty"] = bool(
    not category_metrics.empty
)

checks["threshold_sensitivity_nonempty"] = bool(
    not threshold_sensitivity.empty
)

checks["overall_pass"] = bool(
    checks["overall_pass"]
    and checks["global_metrics_nonempty"]
    and checks["category_metrics_nonempty"]
    and checks["threshold_sensitivity_nonempty"]
    and all(
        checks[f"K{k}_exposure_finite"]
        and checks[f"K{k}_total_exposure_positive"]
        for k in TOP_K_VALUES
    )
)

with open(
    OUT / "validation_checks.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(checks, f, indent=2)


# =========================================================
# 18. EXPERIMENT MANIFEST
# =========================================================

experiment_manifest = {

    "experiment": "22_08_global_vs_category_exposure",

    "split": "independent_test",

    "recommendation_model": "Frozen KGSF",

    "events": EXPECTED_EVENTS,

    "catalog_items": EXPECTED_CATALOG,

    "exposure_method": (
        "Position-discounted cumulative recommendation exposure"
    ),

    "discount": "1/log2(rank+1)",

    "top_k_values": TOP_K_VALUES,

    "global_groups": "Five training-frequency groups",

    "category_groups": (
        "Five within-category training-frequency groups"
    ),

    "popularity_tie_method": "average",

    "ranking_tie_breaking": (
        "descending_score_then_ascending_movie_id"
    ),

    "category_thresholds": CATEGORY_THRESHOLDS,

    "primary_category_threshold": (
        PRIMARY_CATEGORY_THRESHOLD
    ),

    "category_sources": [
        "Accepted IMDb genre evidence",
        "Accepted DBpedia genre evidence"
    ],

    "uncategorized_items_retained_globally": True,

    "training_performed": False,

    "parameter_search_performed": False,

    "fairness_calibration_performed": False,

    "validation": checks

}

with open(
    OUT / "experiment_manifest.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        experiment_manifest,
        f,
        indent=2
    )


# =========================================================
# 19. FINAL SUMMARY
# =========================================================

print("\n" + "=" * 85)

print("EXPERIMENT 22.08 COMPLETE")

print("=" * 85)

print("\nGLOBAL EXPOSURE SUMMARY")

print(
    global_metrics[
        [
            "K",
            "popularity_group",
            "catalog_share",
            "exposure_share",
            "exposure_ratio",
            "zero_exposure_items"
        ]
    ].to_string(index=False)
)

print("\nCATEGORY EXPOSURE SUMMARY")

print(
    category_metrics[
        [
            "category",
            "K",
            "popularity_group",
            "catalog_share",
            "exposure_share",
            "exposure_ratio"
        ]
    ].head(40).to_string(index=False)
)

print("\nZERO-EXPOSURE COUNTS")

print(
    zero_exposure.groupby("K")
    .size()
    .to_string()
)

print("\nVALIDATION")

print(json.dumps(checks, indent=2))

print("\nOUTPUT DIRECTORY:")
print(OUT.resolve())

print("\nOVERALL PASS:", checks["overall_pass"])

print("\nEXPERIMENT 22.08 FINISHED")
