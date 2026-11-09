
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


# ============================================================
# EXPERIMENT 21.23B
# CATEGORY-AWARE ITEM EXPOSURE FAIRNESS AUDIT
# ============================================================

ROOT = Path(__file__).resolve().parent

CATALOG_FILE = (
    ROOT / "data/dataset/redial/nltk/movie_ids.json"
)

PROFILE_FILE = (
    ROOT / "results_category_audit/experiment_21_22A/"
    "catalog_item_category_profiles.csv"
)

EVENT_FILE = (
    ROOT / "results_corrected/"
    "adaptive_m2_validation_events_clustered.csv"
)

OUT = ROOT / "results_corrected/experiment_21_23B"
OUT.mkdir(parents=True, exist_ok=True)

TOP_K = 10

print("=" * 75)
print("EXPERIMENT 21.23B: CATEGORY-AWARE EXPOSURE FAIRNESS AUDIT")
print("=" * 75)


# ============================================================
# 1. HELPER FUNCTIONS
# ============================================================

def parse_bool(value):

    if pd.isna(value):
        return False

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    return str(value).strip().lower() in {
        "true", "1", "yes"
    }


def parse_list(value, delimiter="|"):

    if pd.isna(value):
        return []

    return sorted({
        str(x).strip()
        for x in str(value).split(delimiter)
        if str(x).strip()
    })


def normalize_category(value):

    return " ".join(str(value).strip().casefold().split())


def gini(values):

    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return 0.0

    if np.any(values < 0):
        raise ValueError("Gini requires nonnegative values.")

    total = values.sum()

    if total == 0:
        return 0.0

    values = np.sort(values)

    n = len(values)

    return float(
        (2 * np.dot(np.arange(1, n + 1), values))
        / (n * total)
        - (n + 1) / n
    )


def safe_mean(series):

    return float(pd.to_numeric(
        series, errors="coerce"
    ).mean())


def load_json_list(value):

    result = json.loads(value)

    if not isinstance(result, list):
        raise ValueError("Expected a JSON list.")

    return [int(x) for x in result]


# ============================================================
# 2. LOAD ORIGINAL CATALOG
# ============================================================

with open(CATALOG_FILE, encoding="utf-8") as f:
    original_movie_ids = json.load(f)

catalog_ids = set(map(int, original_movie_ids))

assert len(catalog_ids) == 6924

print("\n[1] ORIGINAL CATALOG")
print("Catalog population:", len(catalog_ids))


# ============================================================
# 3. LOAD CATEGORY PROFILES
# ============================================================

profiles = pd.read_csv(PROFILE_FILE)

profiles["movie_entity_id"] = (
    profiles["movie_entity_id"].astype(int)
)

assert profiles["movie_entity_id"].is_unique

assert set(profiles["movie_entity_id"]) == catalog_ids

profiles = profiles.set_index("movie_entity_id")

# Category assignments use independently supported evidence.
# IMDb conflicts are excluded from verified IMDb evidence.
# Accepted DBpedia genres can still be used independently.

item_categories = {}
item_category_source = {}
item_category_status = {}

all_categories = {}
identity_review_ids = set()

for movie_id, row in profiles.iterrows():

    movie_id = int(movie_id)

    ambiguous = (
        str(row["identity_resolution_status"])
        == "AMBIGUOUS_REQUIRES_REVIEW"
    )

    if ambiguous:
        identity_review_ids.add(movie_id)

        item_categories[movie_id] = []
        item_category_source[movie_id] = "WITHHELD"
        item_category_status[movie_id] = "IDENTITY_REVIEW"

        continue

    categories = {}

    imdb_genres = parse_list(
    row["imdb_genres"], delimiter="|"
    )

    # DBpedia
    dbpedia_genres = parse_list(
        row["canonical_genres"],
        delimiter=";"
    )

    imdb_conflict = parse_bool(
        row.get("imdb_conflicting_records", False)
    )

    imdb_valid = (
        parse_bool(row["imdb_genre_evidence"])
        and not imdb_conflict
    )

    if imdb_valid:

        for genre in imdb_genres:

            key = normalize_category(genre)

            if key:
                categories[key] = genre

    # Accepted DBpedia genre evidence.
    dbpedia_valid = parse_bool(
        row["accepted_genre_evidence"]
    )

    if dbpedia_valid:

        for genre in parse_list(
            row["canonical_genres"], delimiter="|"
        ):

            key = normalize_category(genre)

            if key:
                categories[key] = genre

    assigned = sorted(categories.keys())

    item_categories[movie_id] = assigned

    for key, display_name in categories.items():
        all_categories[key] = display_name

    if imdb_valid and dbpedia_valid:
        source = "IMDb_AND_DBPEDIA"

    elif imdb_valid:
        source = "IMDb"

    elif dbpedia_valid:
        source = "DBPEDIA"

    else:
        source = "UNKNOWN"

    item_category_source[movie_id] = source

    if assigned:
        item_category_status[movie_id] = "VERIFIED_CATEGORY"
    elif imdb_conflict:
        item_category_status[movie_id] = "CONFLICTED_IMDB_EVIDENCE"
    else:
        item_category_status[movie_id] = "UNKNOWN"


verified_items = {
    movie_id
    for movie_id, categories in item_categories.items()
    if categories
}

category_names = sorted(all_categories.keys())

print("\n[2] CATEGORY PROFILE")
print("Catalog items:", len(profiles))
print("Verified category items:", len(verified_items))
print("Unknown category items:", len(catalog_ids - verified_items))
print("Identity review items:", len(identity_review_ids))
print("Unique category labels:", len(category_names))


# ============================================================
# 4. LOAD VALIDATION EVENTS
# ============================================================

events = pd.read_csv(EVENT_FILE)

required_columns = [
    "event_id",
    "method",
    "conv_id",
    "target_movie_id",
    "frequency_group",
    "hit10",
    "hit50",
    "ndcg10",
    "rank_top50",
    "top10_movie_ids",
    "top10_group_ids"
]

missing = [
    col for col in required_columns
    if col not in events.columns
]

if missing:
    raise ValueError(f"Missing event columns: {missing}")

events["target_movie_id"] = (
    events["target_movie_id"].astype(int)
)

methods = sorted(events["method"].unique())

print("\n[3] VALIDATION EVENTS")
print("Total events:", len(events))
print("Methods:", len(methods))
print("Conversations:", events["conv_id"].nunique())

assert len(methods) == 7


# ============================================================
# 5. OPPORTUNITY CONSISTENCY
# ============================================================

opportunity_sets = {}

for method in methods:

    subset = events[events["method"] == method]

    opportunity_sets[method] = set(
        zip(
            subset["conv_id"].astype(str),
            subset["target_movie_id"].astype(int),
            subset["frequency_group"].astype(str)
        )
    )

reference = opportunity_sets[methods[0]]

for method in methods[1:]:

    if opportunity_sets[method] != reference:
        raise ValueError(
            f"Recommendation opportunities differ: {method}"
        )

event_counts = events.groupby("method").size()

if event_counts.nunique() != 1:
    raise ValueError("Event counts differ across methods.")

print("\n[4] OPPORTUNITY VALIDATION")
print("Identical opportunities: PASS")
print("Identical event counts: PASS")


# ============================================================
# 6. PREPARE OUTPUT COLLECTIONS
# ============================================================

item_results = []
category_results = []
unknown_results = []
relevance_results = []
group_code_results = []

validation = {
    "experiment": "21.23B",
    "catalog_population": len(catalog_ids),
    "methods": len(methods),
    "all_catalog_items_preserved": True,
    "identical_recommendation_opportunities": True,
    "invalid_recommendation_ids": 0,
    "duplicate_top10_recommendations": 0,
    "recommendation_slot_conservation": True,
    "zero_exposure_items_included": True,
    "fractional_category_allocation": True,
    "ambiguous_identity_items_excluded_from_verified_categories": True
}


# ============================================================
# 7. MAIN EXPOSURE AUDIT
# ============================================================

for method in methods:

    print("\n" + "-" * 65)
    print("METHOD:", method)

    subset = events[
        events["method"] == method
    ]

    exposure = Counter()
    group_exposure = Counter()

    category_exposure = Counter()

    unknown_slots = 0
    identity_review_slots = 0
    verified_slots = 0

    invalid_ids = 0
    duplicate_lists = 0
    total_slots = 0

    for _, row in subset.iterrows():

        movies = load_json_list(
            row["top10_movie_ids"]
        )

        groups = load_json_list(
            row["top10_group_ids"]
        )

        if len(movies) != TOP_K or len(groups) != TOP_K:
            raise ValueError(
                f"Invalid top-10 list in {method}"
            )

        if len(set(movies)) != TOP_K:
            duplicate_lists += 1
            raise ValueError(
                f"Duplicate recommendation in {method}"
            )

        if any(movie not in catalog_ids for movie in movies):
            invalid_ids += 1
            raise ValueError(
                f"Invalid catalog ID in {method}"
            )

        for movie, group in zip(movies, groups):

            movie = int(movie)

            exposure[movie] += 1
            group_exposure[str(group)] += 1

            total_slots += 1

            categories = item_categories[movie]

            if categories:

                verified_slots += 1

                fractional_weight = 1.0 / len(categories)

                for category in categories:

                    category_exposure[category] += (
                        fractional_weight
                    )

            else:

                unknown_slots += 1

                if movie in identity_review_ids:
                    identity_review_slots += 1

    # --------------------------------------------------------
    # ITEM-LEVEL METRICS
    # --------------------------------------------------------

    exposure_array = np.array(
        [exposure.get(x, 0) for x in catalog_ids],
        dtype=float
    )

    exposed_items = sum(
        value > 0 for value in exposure_array
    )

    coverage = exposed_items / len(catalog_ids)

    item_gini = gini(exposure_array)

    item_results.append({
        "method": method,
        "events": len(subset),
        "recommendation_slots": total_slots,
        "catalog_items": len(catalog_ids),
        "unique_exposed_items": exposed_items,
        "zero_exposure_items": len(catalog_ids) - exposed_items,
        "catalog_coverage": coverage,
        "item_exposure_gini": item_gini,
        "maximum_item_exposure": int(exposure_array.max()),
        "minimum_item_exposure": int(exposure_array.min()),
        "mean_item_exposure": float(exposure_array.mean())
    })

    # --------------------------------------------------------
    # CATEGORY-LEVEL METRICS
    # --------------------------------------------------------

    category_values = np.array(
        [
            category_exposure.get(category, 0)
            for category in category_names
        ],
        dtype=float
    )

    category_gini = gini(category_values)

    known_exposure = sum(category_exposure.values())

    for category in category_names:

        fractional_exposure = category_exposure.get(
            category, 0
        )

        category_results.append({
            "method": method,
            "category": all_categories[category],
            "category_key": category,
            "fractional_exposure": fractional_exposure,
            "share_of_all_recommendation_slots": (
                fractional_exposure / total_slots
                if total_slots else 0
            ),
            "share_of_known_category_exposure": (
                fractional_exposure / known_exposure
                if known_exposure else 0
            )
        })

    # --------------------------------------------------------
    # UNKNOWN CATEGORY EXPOSURE
    # --------------------------------------------------------

    unknown_results.append({
        "method": method,
        "total_slots": total_slots,
        "verified_category_slots": verified_slots,
        "unknown_category_slots": unknown_slots,
        "identity_review_slots": identity_review_slots,
        "verified_category_exposure_share": (
            verified_slots / total_slots
        ),
        "unknown_category_exposure_share": (
            unknown_slots / total_slots
        ),
        "identity_review_exposure_share": (
            identity_review_slots / total_slots
        ),
        "fractional_category_exposure_conserved": (
            np.isclose(known_exposure, verified_slots)
        )
    })

    # --------------------------------------------------------
    # RELEVANCE METRICS
    # --------------------------------------------------------

    ranks = pd.to_numeric(
        subset["rank_top50"], errors="coerce"
    )

    mrr10 = np.where(
        ranks.between(1, TOP_K),
        1.0 / ranks,
        0.0
    )

    relevance_results.append({
        "method": method,
        "events": len(subset),
        "HR@10": safe_mean(subset["hit10"]),
        "HR@50": safe_mean(subset["hit50"]),
        "NDCG@10": safe_mean(subset["ndcg10"]),
        "MRR@10": float(np.mean(mrr10)),
        "item_exposure_gini": item_gini,
        "catalog_coverage": coverage,
        "category_exposure_gini": category_gini,
        "unknown_category_exposure_share": (
            unknown_slots / total_slots
        )
    })

    # --------------------------------------------------------
    # UNINTERPRETED GROUP CODE DISTRIBUTION
    # --------------------------------------------------------

    for group, count in sorted(group_exposure.items()):

        group_code_results.append({
            "method": method,
            "group_code": group,
            "recommendation_slots": count,
            "recommendation_slot_share": count / total_slots
        })

    print("Events:", len(subset))
    print("Slots:", total_slots)
    print("Unique exposed items:", exposed_items)
    print("Catalog coverage:", round(coverage, 5))
    print("Item Gini:", round(item_gini, 5))
    print("Category Gini:", round(category_gini, 5))
    print(
        "Unknown category exposure:",
        round(unknown_slots / total_slots, 5)
    )


# ============================================================
# 8. CREATE DATAFRAMES
# ============================================================

item_df = pd.DataFrame(item_results)

category_df = pd.DataFrame(category_results)

unknown_df = pd.DataFrame(unknown_results)

relevance_df = pd.DataFrame(relevance_results)

group_df = pd.DataFrame(group_code_results)


# ============================================================
# 9. FINAL VALIDATION
# ============================================================

assert len(item_df) == 7
assert len(unknown_df) == 7
assert len(relevance_df) == 7

assert item_df["recommendation_slots"].eq(
    len(events[events["method"] == methods[0]]) * TOP_K
).all()

assert unknown_df[
    "fractional_category_exposure_conserved"
].all()

assert (
    item_df["unique_exposed_items"]
    + item_df["zero_exposure_items"]
).eq(len(catalog_ids)).all()

assert validation["invalid_recommendation_ids"] == 0
assert validation["duplicate_top10_recommendations"] == 0


# ============================================================
# 10. EXPORT RESULTS
# ============================================================

item_df.to_csv(
    OUT / "item_exposure_metrics.csv",
    index=False
)

category_df.to_csv(
    OUT / "category_exposure_by_method.csv",
    index=False
)

unknown_df.to_csv(
    OUT / "unknown_category_exposure.csv",
    index=False
)

relevance_df.to_csv(
    OUT / "relevance_fairness_comparison.csv",
    index=False
)

group_df.to_csv(
    OUT / "uninterpreted_top10_group_codes.csv",
    index=False
)

# Concise category exposure summary.
category_summary = (
    category_df.groupby("method")
    .agg(
        unique_categories=("category_key", "nunique"),
        categories_with_exposure=(
            "fractional_exposure",
            lambda x: int((x > 0).sum())
        ),
        category_exposure_gini=(
            "fractional_exposure",
            gini
        )
    )
    .reset_index()
)

category_summary.to_csv(
    OUT / "category_exposure_metrics.csv",
    index=False
)

summary = {
    "experiment": "21.23B",
    "split": "validation_only",
    "catalog_population": len(catalog_ids),
    "event_population": len(events),
    "method_count": len(methods),
    "top_k": TOP_K,
    "verified_category_count": len(category_names),
    "ambiguous_identity_count": len(identity_review_ids),
    "methods": relevance_results,
    "methodological_notes": [
        "Zero-exposure catalog items are included in item Gini.",
        "Multi-genre exposure is allocated fractionally.",
        "Ambiguous identities are excluded from verified categories.",
        "Unknown-category exposure is reported separately.",
        "Group codes are reported without assuming their semantics.",
        "Results use validation data only."
    ],
    "validation": validation
}

(OUT / "experiment_21_23B_summary.json").write_text(
    json.dumps(summary, indent=4),
    encoding="utf-8"
)

(OUT / "experiment_21_23B_validation.json").write_text(
    json.dumps(validation, indent=4),
    encoding="utf-8"
)


# ============================================================
# 11. TERMINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL ITEM EXPOSURE COMPARISON")
print("=" * 75)

print(
    item_df[
        [
            "method",
            "unique_exposed_items",
            "catalog_coverage",
            "item_exposure_gini"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 75)
print("FINAL CATEGORY AND RELEVANCE COMPARISON")
print("=" * 75)

print(
    relevance_df.to_string(index=False)
)

print("\n" + "=" * 75)
print("VALIDATION")
print("=" * 75)

for key, value in validation.items():
    print(f"{key}: {value}")

print("\nOUTPUT DIRECTORY:", OUT)

for file in sorted(OUT.iterdir()):
    print(file.name, file.stat().st_size)

print("\nEXPERIMENT 21.23B COMPLETED")