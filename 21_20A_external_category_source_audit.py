

from pathlib import Path
import pandas as pd
import json

BASE = Path("results_category_audit")

FILES = {
    "imdb": BASE / "experiment_21_17E" /
            "imdb_official_identifier_matches.csv",
    "tmdb": BASE / "experiment_21_17G" /
            "tmdb_identifier_cross_reference.csv",
    "catalog": BASE / "catalog_metadata_reconciliation.csv",
    "profiles": BASE / "experiment_21_19C" /
                "catalog_item_category_profiles.csv"
}

OUT = BASE / "experiment_21_20A"
OUT.mkdir(parents=True, exist_ok=True)

def load(name):
    return pd.read_csv(FILES[name])

print("=" * 75)
print("EXPERIMENT 21.20A: EXTERNAL CATEGORY SOURCE AUDIT")
print("=" * 75)

imdb = load("imdb")
tmdb = load("tmdb")
catalog = load("catalog")
profiles = load("profiles")

# --------------------------------------------------
# 1. SOURCE POPULATIONS
# --------------------------------------------------

for name, df in [
    ("IMDb", imdb),
    ("TMDb", tmdb),
    ("Catalog", catalog),
    ("Profiles", profiles)
]:
    print(f"\n{name}: {len(df)} records")

# --------------------------------------------------
# 2. IMDb GENRE AVAILABILITY
# --------------------------------------------------

imdb["imdb_genres_clean"] = (
    imdb["imdb_genres"].fillna("").astype(str).str.strip()
)

imdb["has_genre"] = (
    imdb["imdb_genres_clean"].ne("")
    & imdb["imdb_genres_clean"].str.lower().ne("nan")
    & imdb["imdb_genres_clean"].str.lower().ne("\\n")
)

print("\n[2] IMDb GENRE COVERAGE")
print("Records:", len(imdb))
print("With genres:", int(imdb["has_genre"].sum()))
print("Without genres:", int((~imdb["has_genre"]).sum()))

print("\nIMDb genre distribution:")
genre_counts = (
    imdb.loc[imdb["has_genre"], "imdb_genres_clean"]
    .value_counts()
)
print(genre_counts.head(30).to_string())

# --------------------------------------------------
# 3. TMDb RESULT INSPECTION
# --------------------------------------------------

def parse_results(value):
    if pd.isna(value) or not str(value).strip():
        return []

    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, TypeError):
        return []

tmdb["parsed_results"] = tmdb["tmdb_results"].apply(parse_results)

tmdb["result_count_parsed"] = (
    tmdb["parsed_results"].apply(len)
)

tmdb["has_results"] = tmdb["result_count_parsed"] > 0

def extract_genre_ids(results):
    genres = set()

    for result in results:
        if not isinstance(result, dict):
            continue

        values = result.get("genre_ids", [])

        if isinstance(values, list):
            genres.update(
                str(x) for x in values if x is not None
            )

    return sorted(genres)

tmdb["candidate_genre_ids"] = (
    tmdb["parsed_results"].apply(extract_genre_ids)
)

tmdb["has_candidate_genres"] = (
    tmdb["candidate_genre_ids"].apply(bool)
)

print("\n[3] TMDb RESULT COVERAGE")
print("Records:", len(tmdb))
print("Records with results:", int(tmdb["has_results"].sum()))
print("Records with candidate genre IDs:",
      int(tmdb["has_candidate_genres"].sum()))

print("\nTMDb result examples:")
for _, row in tmdb[tmdb["has_results"]].head(3).iterrows():
    print("Entity:", row["movie_entity_id"])
    print("Results:", row["result_count_parsed"])
    print("Genre IDs:", row["candidate_genre_ids"])

# --------------------------------------------------
# 4. SOURCE OVERLAP
# --------------------------------------------------

imdb_ids = set(imdb["movie_entity_id"])
tmdb_ids = set(tmdb["movie_entity_id"])
catalog_ids = set(catalog["movie_entity_id"])

print("\n[4] SOURCE OVERLAP")
print("IMDb/TMDb overlap:", len(imdb_ids & tmdb_ids))
print("IMDb/catalog overlap:", len(imdb_ids & catalog_ids))
print("TMDb/catalog overlap:", len(tmdb_ids & catalog_ids))

# --------------------------------------------------
# 5. IDENTITY EVIDENCE
# --------------------------------------------------

if "identity_evidence_assessment" in tmdb.columns:
    print("\n[5] TMDb IDENTITY ASSESSMENT")
    print(
        tmdb["identity_evidence_assessment"]
        .value_counts(dropna=False)
        .to_string()
    )

if "automatic_identity_verification" in tmdb.columns:
    print("\nAutomatic identity verification:")
    print(
        tmdb["automatic_identity_verification"]
        .value_counts(dropna=False)
        .to_string()
    )

# --------------------------------------------------
# 6. SOURCE AVAILABILITY EXPORT
# --------------------------------------------------

imdb_out = imdb[[
    "movie_entity_id",
    "imdb_id",
    "imdb_title_type",
    "imdb_primary_title",
    "imdb_start_year",
    "imdb_genres_clean",
    "has_genre"
]].copy()

tmdb_out = tmdb[[
    "movie_entity_id",
    "request_status",
    "http_status",
    "tmdb_result_count",
    "result_count_parsed",
    "has_results",
    "candidate_genre_ids",
    "has_candidate_genres",
    "identity_evidence_assessment"
]].copy()

availability = catalog[["movie_entity_id", "movie_uri"]].copy()

availability = availability.merge(
    imdb_out,
    on="movie_entity_id",
    how="left",
    validate="one_to_one"
)

availability = availability.merge(
    tmdb_out,
    on="movie_entity_id",
    how="left",
    validate="one_to_one",
    suffixes=("_imdb", "_tmdb")
)

availability = availability.merge(
    profiles[[
        "movie_entity_id",
        "has_original_category_evidence",
        "has_accepted_genre_evidence"
    ]],
    on="movie_entity_id",
    how="left",
    validate="one_to_one"
)

availability.to_csv(
    OUT / "external_category_source_availability.csv",
    index=False
)

# --------------------------------------------------
# 7. VALIDATION
# --------------------------------------------------

validation = {
    "catalog_population_preserved": (
        len(availability) == len(catalog)
    ),
    "unique_catalog_entities": (
        availability["movie_entity_id"].is_unique
    ),
    "imdb_entity_ids_unique": imdb["movie_entity_id"].is_unique,
    "tmdb_entity_ids_unique": tmdb["movie_entity_id"].is_unique,
    "all_catalog_entities_preserved": (
        set(availability["movie_entity_id"]) == catalog_ids
    )
}

with open(
    OUT / "experiment_21_20A_validation.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(validation, f, indent=4)

print("\n[6] FINAL VALIDATION")
for key, value in validation.items():
    print(key, "PASS" if value else "FAIL")

print("\nExperiment 21.20A diagnostic completed.")
