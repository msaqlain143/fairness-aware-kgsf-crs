
"""
EXPERIMENT 21.16F
External Movie Identifier Evidence Validation

Purpose:
- Validate the 100 MovieLens conflict groups against MovieLens 32M.
- Retrieve IMDb and TMDb identifiers.
- Compare each database record's title with MovieLens metadata.
- Preserve unresolved identity cases.
- Never automatically merge entities.

Original source files are read-only.
"""

import json
import re
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(".")

MERGED_FILE = (
    ROOT / "external_validation/redial_reference/movies_merged.csv"
)

LINKS_FILE = (
    ROOT / "external_validation/movielens_32m/links.csv"
)

MOVIES_FILE = (
    ROOT / "external_validation/movielens_32m/movies.csv"
)

PAIRWISE_FILE = (
    ROOT / "results_category_audit/experiment_21_16E/"
    "movielens_conflict_pairwise_analysis.csv"
)

ARCHIVE_FILE = (
    ROOT / "external_validation/movielens_32m/ml-32m.zip"
)

OUTPUT_DIR = (
    ROOT / "results_category_audit/experiment_21_16F"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def normalize_id(value):
    if pd.isna(value):
        return None

    try:
        return int(value)
    except (ValueError, TypeError, OverflowError):
        return None


def normalize_title(value):
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()

    # Remove trailing release year.
    value = re.sub(r"\s*\(\d{4}\)\s*$", "", value)

    # Normalize punctuation and whitespace.
    value = re.sub(r"[^a-z0-9]", "", value)

    return value


def extract_year(value):
    if pd.isna(value):
        return None

    match = re.search(r"\((\d{4})\)\s*$", str(value))

    return int(match.group(1)) if match else None


def file_sha256(path):
    if not path.exists():
        return None

    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def safe_string(value):
    if pd.isna(value):
        return ""

    return str(value)


# ============================================================
# 3. HEADER
# ============================================================

print("=" * 75)
print("EXPERIMENT 21.16F: EXTERNAL MOVIE IDENTIFIER EVIDENCE")
print("=" * 75)


# ============================================================
# 4. LOAD SOURCE DATA
# ============================================================

for path in [
    MERGED_FILE,
    LINKS_FILE,
    MOVIES_FILE,
    PAIRWISE_FILE
]:
    require(path.exists(), f"Missing required file: {path}")

merged = pd.read_csv(MERGED_FILE)
links = pd.read_csv(LINKS_FILE)
movies = pd.read_csv(MOVIES_FILE)
pairs = pd.read_csv(PAIRWISE_FILE)

print("\n[1] SOURCE INVENTORY")

print("Merged records:", len(merged))
print("MovieLens link records:", len(links))
print("MovieLens movie records:", len(movies))
print("Conflict pairwise records:", len(pairs))

require(
    {"databaseId", "movielensId", "movieName"}.issubset(
        merged.columns
    ),
    "Unexpected movies_merged.csv schema."
)

require(
    {"movieId", "imdbId", "tmdbId"}.issubset(links.columns),
    "Unexpected links.csv schema."
)

require(
    {"movieId", "title", "genres"}.issubset(movies.columns),
    "Unexpected movies.csv schema."
)

require(
    {"movielens_id", "database_id_a", "database_id_b"}.issubset(
        pairs.columns
    ),
    "Unexpected pairwise audit schema."
)


# ============================================================
# 5. NORMALIZE IDENTIFIERS
# ============================================================

merged["databaseId_norm"] = merged["databaseId"].map(normalize_id)
merged["movielensId_norm"] = merged["movielensId"].map(normalize_id)

links["movieId_norm"] = links["movieId"].map(normalize_id)
movies["movieId_norm"] = movies["movieId"].map(normalize_id)

pairs["movielens_id_norm"] = pairs["movielens_id"].map(normalize_id)
pairs["database_id_a_norm"] = pairs["database_id_a"].map(normalize_id)
pairs["database_id_b_norm"] = pairs["database_id_b"].map(normalize_id)


# ============================================================
# 6. SOURCE INTEGRITY
# ============================================================

print("\n[2] SOURCE INTEGRITY")

checks = {
    "links_movie_id_unique": bool(
        links["movieId_norm"].is_unique
    ),
    "movies_movie_id_unique": bool(
        movies["movieId_norm"].is_unique
    ),
    "pairwise_records_104": len(pairs) == 104,
    "pairwise_movielens_ids_valid": bool(
        pairs["movielens_id_norm"].notna().all()
    ),
    "database_ids_valid": bool(
        pairs[
            ["database_id_a_norm", "database_id_b_norm"]
        ].notna().all().all()
    )
}

for name, status in checks.items():
    print(f"{name}: {'PASS' if status else 'FAIL'}")

require(
    all(checks.values()),
    "Source integrity validation failed."
)


# ============================================================
# 7. BUILD LOOKUP TABLES
# ============================================================

links_index = links.set_index("movieId_norm").to_dict("index")

movies_index = movies.set_index("movieId_norm").to_dict("index")

database_index = {}

for database_id, group in merged.groupby("databaseId_norm"):

    if pd.isna(database_id):
        continue

    database_index[int(database_id)] = group.to_dict("records")


# ============================================================
# 8. CONFLICT GROUP ANALYSIS
# ============================================================

print("\n[3] EXTERNAL IDENTIFIER CROSS-REFERENCE")

pair_results = []

for _, pair in pairs.iterrows():

    movie_id = pair["movielens_id_norm"]

    db_a = pair["database_id_a_norm"]
    db_b = pair["database_id_b_norm"]

    link = links_index.get(movie_id)
    movie = movies_index.get(movie_id)

    records_a = database_index.get(db_a, [])
    records_b = database_index.get(db_b, [])

    titles_a = [
        safe_string(record["movieName"])
        for record in records_a
    ]

    titles_b = [
        safe_string(record["movieName"])
        for record in records_b
    ]

    title_a = titles_a[0] if titles_a else ""
    title_b = titles_b[0] if titles_b else ""

    reference_title = movie["title"] if movie else ""
    reference_genres = movie["genres"] if movie else ""

    imdb_id = link["imdbId"] if link else None
    tmdb_id = link["tmdbId"] if link else None

    normalized_reference = normalize_title(reference_title)

    match_a = bool(
        normalized_reference
        and normalized_reference == normalize_title(title_a)
    )

    match_b = bool(
        normalized_reference
        and normalized_reference == normalize_title(title_b)
    )

    year_a = extract_year(title_a)
    year_b = extract_year(title_b)
    reference_year = extract_year(reference_title)

    if link is None:
        classification = "NO_EXTERNAL_MOVIELENS_REFERENCE"

    elif match_a and match_b:
        classification = "BOTH_SOURCE_TITLES_MATCH_REFERENCE"

    elif match_a or match_b:
        classification = "ONE_SOURCE_TITLE_MATCHES_REFERENCE"

    else:
        classification = "TITLE_MISMATCH_REQUIRES_REVIEW"

    pair_results.append({

        "movielens_id": movie_id,

        "database_id_a": db_a,
        "database_id_b": db_b,

        "database_title_a": title_a,
        "database_title_b": title_b,

        "database_record_count_a": len(records_a),
        "database_record_count_b": len(records_b),

        "reference_title": reference_title,
        "reference_genres": reference_genres,

        "imdb_id": (
            normalize_id(imdb_id)
            if link is not None else None
        ),

        "tmdb_id": normalize_id(tmdb_id),

        "imdb_available": bool(
            link is not None and not pd.isna(imdb_id)
        ),

        "tmdb_available": bool(
            link is not None and not pd.isna(tmdb_id)
        ),

        "title_a_matches_reference": match_a,
        "title_b_matches_reference": match_b,

        "year_a": year_a,
        "year_b": year_b,
        "reference_year": reference_year,

        "both_source_titles_match": match_a and match_b,

        "candidate_classification": classification,

        "identity_decision": "UNRESOLVED",

        "evidence_scope": (
            "GROUP_LEVEL_REFERENCE_NOT_INDEPENDENT_RECORD_PROOF"
        )

    })


result = pd.DataFrame(pair_results)


# ============================================================
# 9. GROUP-LEVEL SUMMARY
# ============================================================

print("\n[4] GROUP-LEVEL EVIDENCE")

group_results = []

for movie_id, group in result.groupby("movielens_id"):

    group_results.append({

        "movielens_id": movie_id,

        "pairwise_comparisons": len(group),

        "unique_database_ids": len(
            set(group["database_id_a"])
            | set(group["database_id_b"])
        ),

        "reference_title": group["reference_title"].iloc[0],

        "imdb_id": group["imdb_id"].iloc[0],

        "tmdb_id": group["tmdb_id"].iloc[0],

        "imdb_available": bool(
            group["imdb_available"].iloc[0]
        ),

        "tmdb_available": bool(
            group["tmdb_available"].iloc[0]
        ),

        "all_source_titles_match": bool(
            group["both_source_titles_match"].all()
        ),

        "pairwise_title_mismatches": int(
            (~group["both_source_titles_match"]).sum()
        )

    })


group_df = pd.DataFrame(group_results)

print("Conflict groups:", len(group_df))
print("Pairwise comparisons:", len(result))

print(
    "Groups with IMDb:",
    int(group_df["imdb_available"].sum())
)

print(
    "Groups with TMDb:",
    int(group_df["tmdb_available"].sum())
)

print(
    "Groups with all titles matching:",
    int(group_df["all_source_titles_match"].sum())
)


# ============================================================
# 10. REFERENCE COVERAGE
# ============================================================

print("\n[5] REFERENCE COVERAGE")

conflict_ids = set(pairs["movielens_id_norm"].dropna())

links_ids = set(links["movieId_norm"].dropna())

movies_ids = set(movies["movieId_norm"].dropna())

print("Conflict MovieLens IDs:", len(conflict_ids))
print("IDs in links.csv:", len(conflict_ids & links_ids))
print("IDs in movies.csv:", len(conflict_ids & movies_ids))

print("Missing links:", len(conflict_ids - links_ids))
print("Missing movie metadata:", len(conflict_ids - movies_ids))

missing_ids = sorted(conflict_ids - links_ids)

missing_df = pd.DataFrame({
    "movielens_id": missing_ids
})


# ============================================================
# 11. VALIDATION
# ============================================================

print("\n[6] FINAL VALIDATION")

validation = {

    "pairwise_population_preserved": len(result) == len(pairs),

    "conflict_group_population_preserved": (
        len(group_df) == len(conflict_ids)
    ),

    "external_link_integrity": (
        links["movieId_norm"].is_unique
    ),

    "external_movie_integrity": (
        movies["movieId_norm"].is_unique
    ),

    "all_pairwise_records_accounted_for": (
        result["movielens_id"].notna().all()
    ),

    "no_automatic_identity_merging": (
        result["identity_decision"].eq("UNRESOLVED").all()
    ),

    "group_level_evidence_explicit": (
        result["evidence_scope"].notna().all()
    )

}

for name, status in validation.items():

    print(
        f"{name}: {'PASS' if status else 'FAIL'}"
    )

require(
    all(validation.values()),
    "Final validation failed."
)


# ============================================================
# 12. EXPORT RESULTS
# ============================================================

print("\n[7] EXPORTING RESULTS")

result.to_csv(
    OUTPUT_DIR / "external_identifier_pairwise_evidence.csv",
    index=False,
    encoding="utf-8-sig"
)

group_df.to_csv(
    OUTPUT_DIR / "external_identifier_group_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

missing_df.to_csv(
    OUTPUT_DIR / "missing_external_reference_ids.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 13. MANIFEST
# ============================================================

manifest = {

    "experiment": "21.16F",

    "name": "External Movie Identifier Evidence Validation",

    "purpose": (
        "Cross-reference MovieLens conflict groups against "
        "independent MovieLens IMDb/TMDb metadata."
    ),

    "timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "source_counts": {

        "merged_records": int(len(merged)),

        "movielens_link_records": int(len(links)),

        "movielens_movie_records": int(len(movies)),

        "conflict_groups": int(len(group_df)),

        "pairwise_comparisons": int(len(result))

    },

    "evidence_counts": {

        "groups_with_imdb": int(
            group_df["imdb_available"].sum()
        ),

        "groups_with_tmdb": int(
            group_df["tmdb_available"].sum()
        ),

        "groups_all_titles_match": int(
            group_df["all_source_titles_match"].sum()
        ),

        "groups_with_title_mismatches": int(
            (~group_df["all_source_titles_match"]).sum()
        ),

        "missing_external_reference_ids": int(
            len(missing_df)
        )

    },

    "validation": {
    key: bool(value)
    for key, value in validation.items()
    },

    "source_checksums": {

        "movielens_archive_sha256": file_sha256(ARCHIVE_FILE)

    },

    "identity_policy": {

        "external_identifiers_are_group_level_evidence": True,

        "shared_movielens_id_is_identity_proof": False,

        "title_match_is_identity_proof": False,

        "automatic_entity_merging": False,

        "unresolved_records_preserved": True

    },

    "limitations": [

        "IMDb and TMDb identifiers are inherited from the shared MovieLens ID.",

        "They do not independently verify each conflicting database record.",

        "Title agreement is supporting evidence, not identity proof.",

        "External database records have not yet been independently retrieved.",

        "No automatic identity resolution has been performed."

    ],

    "outputs": [

        "external_identifier_pairwise_evidence.csv",

        "external_identifier_group_summary.csv",

        "missing_external_reference_ids.csv",

        "experiment_21_16F_manifest.json"

    ],

    "completion_status": "COMPLETED_VALIDATION"

}


with open(
    OUTPUT_DIR / "experiment_21_16F_manifest.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        manifest,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 14. COMPLETION
# ============================================================

print("\n" + "=" * 75)
print("EXPERIMENT 21.16F COMPLETED")
print("=" * 75)

print("Conflict groups:", len(group_df))
print("Pairwise comparisons:", len(result))

print("\nOUTPUT DIRECTORY:")
print(OUTPUT_DIR.resolve())

print("\nOriginal files unchanged.")
print("Unified identity mapping unchanged.")
print("Knowledge graph unchanged.")
print("No model training or inference.")
print("No automatic entity merging.")

print("\nEXPERIMENT 21.16F FINISHED.")
