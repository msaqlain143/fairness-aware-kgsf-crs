

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results_category_audit"

OUT = BASE / "experiment_21_22A"
OUT.mkdir(parents=True, exist_ok=True)

CATALOG_FILE = BASE / "catalog_metadata_reconciliation.csv"

IMDB_FILE = (
    BASE / "experiment_21_17E" /
    "imdb_official_identifier_matches.csv"
)

DBPEDIA_PROFILE_FILE = (
    BASE / "experiment_21_19C" /
    "catalog_item_category_profiles.csv"
)

GRANULARITY_FILE = (
    BASE / "experiment_21_20C" /
    "movie_entity_granularity.csv"
)

EDGE_FILE = (
    BASE / "experiment_21_18A2" /
    "movie_category_edges_raw.csv"
)


def clean(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def split_genres(value):
    value = clean(value)

    if not value:
        return []

    return sorted({
        x.strip()
        for x in value.split(",")
        if x.strip()
    })


print("=" * 75)
print("EXPERIMENT 21.22A: CATEGORY PROFILE CONSTRUCTION")
print("=" * 75)


# --------------------------------------------------
# 1. LOAD SOURCES
# --------------------------------------------------

catalog = pd.read_csv(CATALOG_FILE)
imdb = pd.read_csv(IMDB_FILE)
dbpedia = pd.read_csv(DBPEDIA_PROFILE_FILE)
granularity = pd.read_csv(GRANULARITY_FILE)
edges = pd.read_csv(EDGE_FILE)

for df in (catalog, imdb, dbpedia, granularity):
    df["movie_entity_id"] = pd.to_numeric(
        df["movie_entity_id"], errors="raise"
    ).astype(int)

print("\n[1] SOURCE POPULATIONS")
print("Catalog:", len(catalog))
print("IMDb records:", len(imdb))
print("IMDb unique IDs:", imdb["movie_entity_id"].nunique())
print("DBpedia profiles:", len(dbpedia))
print("Granularity review records:", len(granularity))
print("Original category edges:", len(edges))


# --------------------------------------------------
# 2. VALIDATE CATALOG
# --------------------------------------------------

assert len(catalog) == 6924
assert catalog["movie_entity_id"].is_unique
assert dbpedia["movie_entity_id"].is_unique

catalog_ids = set(catalog["movie_entity_id"])

assert set(dbpedia["movie_entity_id"]) == catalog_ids


# --------------------------------------------------
# 3. IDENTIFY AMBIGUOUS ENTITIES
# --------------------------------------------------

ambiguous_ids = set(
    granularity["movie_entity_id"].astype(int)
)

print("\n[2] IDENTITY REVIEW")
print("Ambiguous entities:", len(ambiguous_ids))

# Do not automatically merge or discard records.


# --------------------------------------------------
# 4. AGGREGATE IMDb EVIDENCE
# --------------------------------------------------

imdb["imdb_genres_clean"] = (
    imdb["imdb_genres"].apply(split_genres)
)

imdb_aggregate = []

for movie_id, group in imdb.groupby("movie_entity_id"):

    genre_sets = [
        set(x)
        for x in group["imdb_genres_clean"]
        if x
    ]

    all_genres = sorted(
        set().union(*genre_sets)
    ) if genre_sets else []

    distinct_titles = (
        group["imdb_primary_title"]
        .dropna()
        .astype(str)
        .nunique()
    )

    distinct_imdb_ids = group["imdb_id"].nunique()

    imdb_aggregate.append({
        "movie_entity_id": int(movie_id),
        "imdb_record_count": len(group),
        "imdb_unique_ids": int(distinct_imdb_ids),
        "imdb_unique_titles": int(distinct_titles),
        "imdb_genres": "|".join(all_genres),
        "imdb_genre_evidence": bool(all_genres),
        "imdb_conflicting_records": bool(
            len({tuple(x) for x in genre_sets}) > 1
        )
    })

imdb_agg = pd.DataFrame(imdb_aggregate)

print("\n[3] IMDb AGGREGATION")
print("Unique movie IDs:", len(imdb_agg))
print("Genre-supported records:",
      int(imdb_agg["imdb_genre_evidence"].sum()))


# --------------------------------------------------
# 5. PREPARE DBPEDIA EVIDENCE
# --------------------------------------------------

dbpedia["canonical_genres"] = (
    dbpedia["canonical_genres"].fillna("")
)

dbpedia["accepted_genre_evidence"] = (
    pd.to_numeric(
        dbpedia["accepted_genre_records"],
        errors="coerce"
    ).fillna(0) > 0
)

dbpedia["accepted_hybrid_evidence"] = (
    pd.to_numeric(
        dbpedia["accepted_hybrid_records"],
        errors="coerce"
    ).fillna(0) > 0
)

dbpedia["unvalidated_evidence"] = (
    pd.to_numeric(
        dbpedia["unvalidated_records"],
        errors="coerce"
    ).fillna(0) > 0
)


# --------------------------------------------------
# 6. MERGE EVIDENCE
# --------------------------------------------------

profiles = catalog.merge(
    imdb_agg,
    on="movie_entity_id",
    how="left",
    validate="one_to_one"
)

profiles = profiles.merge(
    dbpedia[
        [
            "movie_entity_id",
            "canonical_genres",
            "accepted_genre_evidence",
            "accepted_hybrid_evidence",
            "unvalidated_evidence",
            "total_category_evidence_records",
            "has_original_category_evidence"
        ]
    ],
    on="movie_entity_id",
    how="left",
    validate="one_to_one"
)

profiles["identity_resolution_status"] = np.where(
    profiles["movie_entity_id"].isin(ambiguous_ids),
    "AMBIGUOUS_REQUIRES_REVIEW",
    "NO_KNOWN_GRANULARITY_CONFLICT"
)

profiles["imdb_genres"] = (
    profiles["imdb_genres"].fillna("")
)

profiles["canonical_genres"] = (
    profiles["canonical_genres"].fillna("")
)

profiles["imdb_genre_evidence"] = (
    profiles["imdb_genre_evidence"]
    .fillna(False).astype(bool)
)

profiles["accepted_genre_evidence"] = (
    profiles["accepted_genre_evidence"]
    .fillna(False).astype(bool)
)

profiles["accepted_hybrid_evidence"] = (
    profiles["accepted_hybrid_evidence"]
    .fillna(False).astype(bool)
)

profiles["unvalidated_evidence"] = (
    profiles["unvalidated_evidence"]
    .fillna(False).astype(bool)
)


# --------------------------------------------------
# 7. ASSIGN CATEGORY EVIDENCE STATUS
# --------------------------------------------------

def assign_status(row):

    if row["identity_resolution_status"] == (
        "AMBIGUOUS_REQUIRES_REVIEW"
    ):
        return "IDENTITY_REVIEW_REQUIRED"

    if (
        row["imdb_genre_evidence"]
        and row["accepted_genre_evidence"]
    ):
        return "MULTI_SOURCE_GENRE_EVIDENCE"

    if row["imdb_genre_evidence"]:
        return "IMDB_GENRE_EVIDENCE"

    if row["accepted_genre_evidence"]:
        return "DBPEDIA_GENRE_EVIDENCE"

    if row["accepted_hybrid_evidence"]:
        return "HYBRID_EVIDENCE_ONLY"

    if row["unvalidated_evidence"]:
        return "UNVALIDATED_EVIDENCE_ONLY"

    return "UNKNOWN"


profiles["category_availability"] = profiles.apply(
    assign_status, axis=1
)


# --------------------------------------------------
# 8. CREATE EVIDENCE REGISTER
# --------------------------------------------------

evidence_records = []

for _, row in profiles.iterrows():

    movie_id = int(row["movie_entity_id"])

    for genre in split_genres(row["imdb_genres"]):

        evidence_records.append({
            "movie_entity_id": movie_id,
            "category": genre,
            "source": "IMDb",
            "evidence_type": "EXTERNAL_GENRE",
            "identity_status": row[
                "identity_resolution_status"
            ]
        })

    for genre in split_genres(
        row["canonical_genres"].replace("|", ",")
    ):

        evidence_records.append({
            "movie_entity_id": movie_id,
            "category": genre,
            "source": "DBpedia",
            "evidence_type": "ACCEPTED_GENRE",
            "identity_status": row[
                "identity_resolution_status"
            ]
        })


evidence = pd.DataFrame(
    evidence_records,
    columns=[
        "movie_entity_id",
        "category",
        "source",
        "evidence_type",
        "identity_status"
    ]
)


# --------------------------------------------------
# 9. VALIDATION
# --------------------------------------------------

assert len(profiles) == 6924
assert profiles["movie_entity_id"].is_unique
assert set(profiles["movie_entity_id"]) == catalog_ids

assert (
    profiles["identity_resolution_status"]
    .eq("AMBIGUOUS_REQUIRES_REVIEW")
    .sum()
    == len(ambiguous_ids)
)

assert not profiles["category_availability"].isna().any()

# Ambiguous records are not treated as verified evidence.
verified = profiles[
    profiles["identity_resolution_status"]
    != "AMBIGUOUS_REQUIRES_REVIEW"
]

assert (
    verified["category_availability"]
    .eq("UNKNOWN").sum()
    + verified["category_availability"].isin([
        "IMDB_GENRE_EVIDENCE",
        "DBPEDIA_GENRE_EVIDENCE",
        "MULTI_SOURCE_GENRE_EVIDENCE",
        "HYBRID_EVIDENCE_ONLY",
        "UNVALIDATED_EVIDENCE_ONLY"
    ]).sum()
    == len(verified)
)


# --------------------------------------------------
# 10. EXPORT
# --------------------------------------------------

profiles.to_csv(
    OUT / "catalog_item_category_profiles.csv",
    index=False
)

evidence.to_csv(
    OUT / "item_category_evidence.csv",
    index=False
)

coverage = (
    profiles["category_availability"]
    .value_counts()
    .to_dict()
)

summary = {
    "experiment": "21.22A",
    "catalog_population": len(profiles),
    "unique_catalog_items": int(
        profiles["movie_entity_id"].nunique()
    ),
    "imdb_source_records": len(imdb),
    "imdb_unique_movie_ids": int(
        imdb["movie_entity_id"].nunique()
    ),
    "dbpedia_profile_records": len(dbpedia),
    "ambiguous_identity_entities": len(ambiguous_ids),
    "category_availability_distribution": {
        str(k): int(v) for k, v in coverage.items()
    },
    "evidence_records": len(evidence),
    "unknown_category_items": int(
        profiles["category_availability"].eq("UNKNOWN").sum()
    ),
    "catalog_population_preserved": True,
    "automatic_entity_merging": False,
    "unknown_categories_invented": False
}

(OUT / "category_coverage_summary.json").write_text(
    json.dumps(summary, indent=4),
    encoding="utf-8"
)

validation = {
    "experiment": "21.22A",
    "catalog_population_preserved": True,
    "unique_movie_ids": True,
    "all_catalog_items_present": True,
    "ambiguous_entities_flagged": True,
    "no_automatic_entity_merging": True,
    "no_automatic_genre_assignment": True,
    "unknown_status_preserved": True
}

(OUT / "category_profile_validation.json").write_text(
    json.dumps(validation, indent=4),
    encoding="utf-8"
)

print("\n[4] CATEGORY DISTRIBUTION")

for key, value in coverage.items():
    print(f"{key}: {value}")

print("\n[5] VALIDATION")
for key, value in validation.items():
    print(f"{key}: {value}")

print("\n[6] OUTPUTS")
for file in OUT.iterdir():
    print(file.name, file.stat().st_size)

print("\nEXPERIMENT 21.22A COMPLETED")
