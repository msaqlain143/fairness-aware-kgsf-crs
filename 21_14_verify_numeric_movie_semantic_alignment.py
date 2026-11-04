
"""
EXPERIMENT 21.14
Numeric-to-Movie Semantic Alignment Verification

Purpose:
Independently verify the 812 numeric identifiers against
the ReDial movie metadata and internal movie catalog.

Original dataset files remain unchanged.
No graph modification or model training is performed.
"""

import json
import re
import unicodedata
from pathlib import Path
from collections import Counter

import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE = Path("data/dataset/redial/nltk")

REFERENCE = Path("external_validation/redial_reference")

OUTPUT = Path(
    "results_category_audit/experiment_21_14"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

ENTITY_FILE = BASE / "entity2id.json"
MOVIE_FILE = BASE / "movie_ids.json"

MERGED_FILE = REFERENCE / "movies_merged.csv"
MENTION_FILE = REFERENCE / "movies_with_mentions.csv"


# ============================================================
# 2. VALIDATION UTILITIES
# ============================================================

def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def normalize_title(value):
    """
    Normalize formatting differences without removing
    meaningful title information such as release years.
    """
    if pd.isna(value):
        return ""

    value = unicodedata.normalize("NFKC", str(value))
    value = value.casefold()

    value = re.sub(r"\s+", " ", value).strip()

    # Ignore punctuation and spacing differences.
    value = "".join(
        char for char in value
        if char.isalnum()
    )

    return value


def load_json(path):
    require(path.exists(), f"Missing source file: {path}")

    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_csv(path, required_columns):
    require(path.exists(), f"Missing reference file: {path}")

    df = pd.read_csv(path)

    missing = set(required_columns) - set(df.columns)

    require(
        not missing,
        f"Missing columns in {path}: {sorted(missing)}"
    )

    return df


# ============================================================
# 3. LOAD SOURCE DATA
# ============================================================

print("=" * 75)
print("EXPERIMENT 21.14: NUMERIC-TO-MOVIE SEMANTIC ALIGNMENT")
print("=" * 75)

entity2id = load_json(ENTITY_FILE)
movie_ids = load_json(MOVIE_FILE)

merged = load_csv(
    MERGED_FILE,
    ["databaseId", "movieName", "movielensId"]
)

mentions = load_csv(
    MENTION_FILE,
    ["movieId", "movieName", "nbMentions"]
)

require(
    isinstance(entity2id, dict),
    "entity2id.json must contain a dictionary."
)

require(
    isinstance(movie_ids, list),
    "movie_ids.json must contain a list."
)

# Numeric external identifiers mapped to internal IDs.
numeric_mapping = {}

for key, internal_id in entity2id.items():

    key = str(key).strip()

    if key.isdigit():

        external_id = int(key)
        internal_id = int(internal_id)

        require(
            external_id not in numeric_mapping,
            f"Duplicate numeric identifier: {external_id}"
        )

        numeric_mapping[external_id] = internal_id


movie_catalog = [int(x) for x in movie_ids]
movie_catalog_set = set(movie_catalog)

require(
    len(movie_catalog) == len(movie_catalog_set),
    "Duplicate internal movie IDs detected."
)

print("\n[1] SOURCE DATA")

print("Entity mappings:", len(entity2id))
print("Numeric mappings:", len(numeric_mapping))
print("Movie catalog:", len(movie_catalog))
print("Merged reference:", len(merged))
print("Mention reference:", len(mentions))


# ============================================================
# 4. PREPARE REFERENCE IDENTIFIERS
# ============================================================

merged["databaseId"] = pd.to_numeric(
    merged["databaseId"],
    errors="coerce"
)

mentions["movieId"] = pd.to_numeric(
    mentions["movieId"],
    errors="coerce"
)

target_ids = set(numeric_mapping.keys())

merged_target = merged[
    merged["databaseId"].isin(target_ids)
].copy()

mention_target = mentions[
    mentions["movieId"].isin(target_ids)
].copy()

merged_counts = (
    merged_target.groupby("databaseId")
    .size()
    .to_dict()
)

mention_counts = (
    mention_target.groupby("movieId")
    .size()
    .to_dict()
)

# Build lookup dictionaries only for unique matches.
merged_lookup = {
    int(row.databaseId): row
    for row in merged_target.itertuples()
    if merged_counts[int(row.databaseId)] == 1
}

mention_lookup = {
    int(row.movieId): row
    for row in mention_target.itertuples()
    if mention_counts[int(row.movieId)] == 1
}


# ============================================================
# 5. SEMANTIC ALIGNMENT
# ============================================================

results = []

for external_id, internal_id in sorted(
    numeric_mapping.items()
):

    merged_row = merged_lookup.get(external_id)
    mention_row = mention_lookup.get(external_id)

    merged_title = (
        str(merged_row.movieName)
        if merged_row is not None else ""
    )

    mention_title = (
        str(mention_row.movieName)
        if mention_row is not None else ""
    )

    merged_normalized = normalize_title(merged_title)
    mention_normalized = normalize_title(mention_title)

    title_match = (
        bool(merged_normalized)
        and bool(mention_normalized)
        and merged_normalized == mention_normalized
    )

    in_catalog = internal_id in movie_catalog_set

    merged_unique = merged_counts.get(external_id, 0) == 1
    mention_unique = mention_counts.get(external_id, 0) == 1

    merged_present = merged_counts.get(external_id, 0) > 0
    mention_present = mention_counts.get(external_id, 0) > 0

    if (
        in_catalog
        and merged_unique
        and mention_unique
        and title_match
    ):
        status = "VERIFIED"

    else:
        status = "REVIEW"

    results.append({

        "external_numeric_id": external_id,

        "internal_entity_id": internal_id,

        "merged_movie_title": merged_title,

        "mention_movie_title": mention_title,

        "mention_count": (
            int(mention_row.nbMentions)
            if mention_row is not None else None
        ),

        "in_movie_catalog": in_catalog,

        "merged_reference_present": merged_present,

        "mention_reference_present": mention_present,

        "merged_reference_unique": merged_unique,

        "mention_reference_unique": mention_unique,

        "normalized_merged_title": merged_normalized,

        "normalized_mention_title": mention_normalized,

        "title_match": title_match,

        "mapping_status": status

    })


result_df = pd.DataFrame(results)


# ============================================================
# 6. VALIDATION SUMMARY
# ============================================================

verified = int(
    (result_df["mapping_status"] == "VERIFIED").sum()
)

review = int(
    (result_df["mapping_status"] == "REVIEW").sum()
)

print("\n[2] IDENTIFIER COVERAGE")

print("Target identifiers:", len(target_ids))

print(
    "Merged reference matches:",
    int(result_df["merged_reference_present"].sum())
)

print(
    "Mention reference matches:",
    int(result_df["mention_reference_present"].sum())
)

print(
    "Internal catalog matches:",
    int(result_df["in_movie_catalog"].sum())
)


print("\n[3] SEMANTIC VALIDATION")

print(
    "Unique merged matches:",
    int(result_df["merged_reference_unique"].sum())
)

print(
    "Unique mention matches:",
    int(result_df["mention_reference_unique"].sum())
)

print(
    "Matching normalized titles:",
    int(result_df["title_match"].sum())
)

print("Verified mappings:", verified)
print("Mappings requiring review:", review)


# ============================================================
# 7. DUPLICATE AND CONFLICT AUDIT
# ============================================================

print("\n[4] DUPLICATE AUDIT")

print(
    "Target duplicate database IDs:",
    sum(v > 1 for v in merged_counts.values())
)

print(
    "Target duplicate mention IDs:",
    sum(v > 1 for v in mention_counts.values())
)

# Export title mismatches and missing/ambiguous records.
review_df = result_df[
    result_df["mapping_status"] == "REVIEW"
].copy()


# ============================================================
# 8. EXPORT RESULTS
# ============================================================

result_path = OUTPUT / "numeric_movie_semantic_alignment.csv"

review_path = OUTPUT / "numeric_movie_alignment_review.csv"

manifest_path = OUTPUT / "experiment_21_14_manifest.json"

result_df.to_csv(
    result_path,
    index=False,
    encoding="utf-8-sig"
)

review_df.to_csv(
    review_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 9. FINAL MANIFEST
# ============================================================

manifest = {

    "experiment": "21.14",

    "purpose": (
        "Verify numeric external identifiers against "
        "independent ReDial movie metadata."
    ),

    "counts": {

        "numeric_identifiers": len(target_ids),

        "internal_catalog": len(movie_catalog),

        "merged_reference_records": len(merged),

        "mention_reference_records": len(mentions),

        "merged_target_matches": int(
            result_df["merged_reference_present"].sum()
        ),

        "mention_target_matches": int(
            result_df["mention_reference_present"].sum()
        ),

        "internal_catalog_matches": int(
            result_df["in_movie_catalog"].sum()
        ),

        "title_matches": int(
            result_df["title_match"].sum()
        ),

        "verified": verified,

        "requiring_review": review

    },

    "validation": {

        "numeric_mapping_unique": (
            len(numeric_mapping) == len(target_ids)
        ),

        "internal_movie_catalog_unique": (
            len(movie_catalog) == len(movie_catalog_set)
        ),

        "all_internal_ids_in_catalog": bool(
            result_df["in_movie_catalog"].all()
        ),

        "all_merged_references_unique": bool(
            result_df["merged_reference_unique"].all()
        ),

        "all_mention_references_unique": bool(
            result_df["mention_reference_unique"].all()
        ),

        "all_titles_match": bool(
            result_df["title_match"].all()
        ),

        "all_mappings_verified": (
            verified == len(target_ids)
        )

    },

    "interpretation_status": (
        "ALL_VERIFIED"
        if verified == len(target_ids)
        else "PARTIAL_VERIFICATION_REQUIRES_REVIEW"
    ),

    "limitations": [

        "Title agreement is supporting evidence, not independent proof "
        "of the original identifier-generation process.",

        "Title normalization does not establish identity when "
        "different movies share similar titles.",

        "Only the supplied reference datasets were used."

    ]

}

with manifest_path.open(
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
# 10. FINAL REPORT
# ============================================================

print("\n" + "=" * 75)
print("EXPERIMENT 21.14 COMPLETED")
print("=" * 75)

print("Total numeric mappings:", len(target_ids))
print("Verified:", verified)
print("Requires review:", review)

print("\nVALIDATION")

for name, passed in manifest["validation"].items():

    print(
        f"{name}: {'PASS' if passed else 'FAIL'}"
    )

print("\nOUTPUT FILES")

print(result_path)
print(review_path)
print(manifest_path)

print("\nOriginal files unchanged.")
print("No graph modification.")
print("No model training or inference.")

print("\nEXPERIMENT 21.14 FINISHED.")
