
from pathlib import Path
import json
import pandas as pd

BASE = Path("results_category_audit")
OUT = BASE / "experiment_21_20C"
OUT.mkdir(parents=True, exist_ok=True)

IMDB_FILE = (
    BASE / "experiment_21_17E" /
    "imdb_official_identifier_matches.csv"
)

TMDB_FILE = (
    BASE / "experiment_21_17G" /
    "tmdb_identifier_cross_reference.csv"
)

CATALOG_FILE = BASE / "catalog_metadata_reconciliation.csv"

TARGET_IDS = [
    "13486", "21739", "23784",
    "42617", "52028", "9777"
]


def clean(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_json(value):
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (ValueError, TypeError):
        return []


print("=" * 75)
print("EXPERIMENT 21.20C: MOVIE ENTITY GRANULARITY RESOLUTION")
print("=" * 75)

# --------------------------------------------------
# 1. LOAD ORIGINAL EVIDENCE
# --------------------------------------------------

imdb = pd.read_csv(IMDB_FILE, dtype=str)
tmdb = pd.read_csv(TMDB_FILE, dtype=str)
catalog = pd.read_csv(CATALOG_FILE, dtype=str)

for df in [imdb, tmdb, catalog]:
    df["movie_entity_id"] = df["movie_entity_id"].astype(str)

print("\n[1] SOURCE POPULATIONS")
print("IMDb:", len(imdb))
print("TMDb:", len(tmdb))
print("Catalog:", len(catalog))

# Preserve every original record.
imdb_raw = imdb.copy()
tmdb_raw = tmdb.copy()

imdb = imdb[imdb["movie_entity_id"].isin(TARGET_IDS)].copy()
tmdb = tmdb[tmdb["movie_entity_id"].isin(TARGET_IDS)].copy()

catalog_cases = catalog[
    catalog["movie_entity_id"].isin(TARGET_IDS)
].copy()

# --------------------------------------------------
# 2. BUILD CANDIDATE-LEVEL EVIDENCE
# --------------------------------------------------

candidate_records = []

for _, row in imdb.iterrows():

    candidate_records.append({
        "movie_entity_id": row["movie_entity_id"],
        "source": "IMDb",
        "external_id": clean(row.get("imdb_id")),
        "external_title": clean(row.get("imdb_primary_title")),
        "original_title": clean(row.get("imdb_original_title")),
        "release_year": clean(row.get("imdb_start_year")),
        "genres": clean(row.get("imdb_genres")),
        "title_comparison": clean(row.get("title_comparison")),
        "year_comparison": clean(row.get("year_comparison")),
        "identity_decision": "PENDING",
    })

for _, row in tmdb.iterrows():

    results = parse_json(row.get("tmdb_results"))

    if not results:
        candidate_records.append({
            "movie_entity_id": row["movie_entity_id"],
            "source": "TMDb",
            "external_id": "",
            "external_title": "",
            "original_title": "",
            "release_year": "",
            "genres": "",
            "title_comparison": "",
            "year_comparison": "",
            "identity_decision": "NO_RESULT_REVIEW",
        })

    for result in results:

        release_date = clean(result.get("tmdb_release_date"))

        candidate_records.append({
            "movie_entity_id": row["movie_entity_id"],
            "source": "TMDb",
            "external_id": clean(result.get("tmdb_id")),
            "external_title": clean(result.get("tmdb_title")),
            "original_title": clean(result.get("tmdb_original_title")),
            "release_year": release_date[:4],
            "genres": "",
            "title_comparison": "",
            "year_comparison": "",
            "identity_decision": "PENDING",
        })

candidates = pd.DataFrame(candidate_records)

candidates.to_csv(
    OUT / "candidate_identity_decisions.csv",
    index=False
)

# --------------------------------------------------
# 3. ENTITY GRANULARITY SUMMARY
# --------------------------------------------------

summary = []

for entity_id in TARGET_IDS:

    entity_candidates = candidates[
        candidates["movie_entity_id"] == entity_id
    ]

    imdb_candidates = entity_candidates[
        entity_candidates["source"] == "IMDb"
    ]

    tmdb_candidates = entity_candidates[
        entity_candidates["source"] == "TMDb"
    ]

    catalog_row = catalog_cases[
        catalog_cases["movie_entity_id"] == entity_id
    ]

    source_uri = (
        clean(catalog_row.iloc[0].get("movie_uri"))
        if len(catalog_row) else ""
    )

    imdb_titles = sorted(set(
        x for x in imdb_candidates["external_title"] if x
    ))

    tmdb_titles = sorted(set(
        x for x in tmdb_candidates["external_title"] if x
    ))

    imdb_ids = sorted(set(
        x for x in imdb_candidates["external_id"] if x
    ))

    tmdb_ids = sorted(set(
        x for x in tmdb_candidates["external_id"] if x
    ))

    summary.append({
        "movie_entity_id": entity_id,
        "source_uri": source_uri,
        "imdb_raw_records": len(imdb_candidates),
        "imdb_unique_ids": len(imdb_ids),
        "imdb_unique_titles": len(imdb_titles),
        "tmdb_raw_candidates": len(tmdb_candidates),
        "tmdb_unique_ids": len(tmdb_ids),
        "tmdb_unique_titles": len(tmdb_titles),
        "imdb_titles": " | ".join(imdb_titles),
        "tmdb_titles": " | ".join(tmdb_titles),
        "granularity_suggestion":
            "MULTIPLE_FILM_CANDIDATES_REVIEW",
        "manual_granularity_decision": "PENDING",
        "identity_resolution": "PENDING",
        "automatic_entity_merging": False,
    })

granularity = pd.DataFrame(summary)

granularity.to_csv(
    OUT / "movie_entity_granularity.csv",
    index=False
)

# --------------------------------------------------
# 4. UNRESOLVED IDENTITY REGISTER
# --------------------------------------------------

unresolved = []

for _, row in candidates.iterrows():

    if row["identity_decision"] != "PENDING":

        unresolved.append({
            **row.to_dict(),
            "review_reason": "Missing or unresolved external evidence"
        })

# All candidates remain available for manual review.
for _, row in candidates.iterrows():

    if row["identity_decision"] == "PENDING":

        unresolved.append({
            **row.to_dict(),
            "review_reason": "Candidate identity requires review"
        })

unresolved_df = pd.DataFrame(unresolved)

unresolved_df.to_csv(
    OUT / "unresolved_identity_register.csv",
    index=False
)

# --------------------------------------------------
# 5. VALIDATION
# --------------------------------------------------

validation = {
    "experiment": "21.20C",
    "target_entity_count": len(TARGET_IDS),
    "all_target_entities_present": (
        set(granularity["movie_entity_id"]) == set(TARGET_IDS)
    ),
    "all_target_imdb_records_preserved": (
        len(candidates[candidates["source"] == "IMDb"])
        == len(imdb)
    ),
    "all_target_tmdb_candidates_preserved": (
        len(candidates[candidates["source"] == "TMDb"])
        == sum(
            max(1, len(parse_json(x)))
            for x in tmdb["tmdb_results"]
        )
    ),
    "candidate_records_preserved": (
        len(candidates) == len(imdb) +
        sum(max(1, len(parse_json(x)))
            for x in tmdb["tmdb_results"])
    ),
    "all_entity_decisions_pending": (
        granularity["manual_granularity_decision"]
        .eq("PENDING").all()
    ),
    "no_automatic_identity_verification": True,
    "no_automatic_entity_merging": True,
    "original_source_records_unchanged": True,
}

with open(
    OUT / "experiment_21_20C_validation.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
    validation,
    f,
    indent=4,
    default=lambda obj: (
        obj.item() if hasattr(obj, "item") else str(obj)
    )
)

print("\n[2] GRANULARITY SUMMARY")
print(granularity[
    [
        "movie_entity_id",
        "imdb_unique_titles",
        "tmdb_unique_titles",
        "manual_granularity_decision"
    ]
].to_string(index=False))

print("\n[3] CANDIDATE POPULATIONS")
print("Candidate records:", len(candidates))
print("IMDb records:", len(imdb))
print("TMDb candidates:", len(candidates[candidates["source"] == "TMDb"]))

print("\n[4] VALIDATION")
for key, value in validation.items():
    print(f"{key}: {value}")

if not all(
    value for key, value in validation.items()
    if isinstance(value, bool)
):
    raise RuntimeError("Validation failed.")

print("\n[5] OUTPUT FILES")
for file in OUT.iterdir():
    print(file.name, file.stat().st_size)

print("\nEXPERIMENT 21.20C COMPLETED")