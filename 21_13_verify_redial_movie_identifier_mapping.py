
import json
import pandas as pd
from pathlib import Path
from collections import Counter

BASE = Path("data/dataset/redial/nltk")
REF = Path("external_validation/redial_reference")
OUT = Path("results_category_audit/experiment_21_13")

OUT.mkdir(parents=True, exist_ok=True)

print("=" * 75)
print("EXPERIMENT 21.13: REDIAL MOVIE IDENTIFIER VERIFICATION")
print("=" * 75)

# ---------------------------------------------------------
# 1. LOAD ORIGINAL DATA
# ---------------------------------------------------------

with (BASE / "entity2id.json").open(encoding="utf-8") as f:
    entity2id = json.load(f)

with (BASE / "movie_ids.json").open(encoding="utf-8") as f:
    movie_ids = json.load(f)

# Numeric entity keys: external ID -> internal entity ID
numeric_mapping = {
    int(key): int(value)
    for key, value in entity2id.items()
    if str(key).isdigit()
}

print("\n[1] ORIGINAL DATA")
print("Entity records:", len(entity2id))
print("Movie catalog:", len(movie_ids))
print("Numeric identifiers:", len(numeric_mapping))

# ---------------------------------------------------------
# 2. LOAD INDEPENDENT REFERENCES
# ---------------------------------------------------------

merged = pd.read_csv(REF / "movies_merged.csv")
mentions = pd.read_csv(REF / "movies_with_mentions.csv")

merged["databaseId"] = pd.to_numeric(
    merged["databaseId"], errors="coerce"
)

mentions["movieId"] = pd.to_numeric(
    mentions["movieId"], errors="coerce"
)

merged = merged.dropna(subset=["databaseId"]).copy()
mentions = mentions.dropna(subset=["movieId"]).copy()

merged["databaseId"] = merged["databaseId"].astype(int)
mentions["movieId"] = mentions["movieId"].astype(int)

print("\n[2] REFERENCE DATA")
print("Merged records:", len(merged))
print("Mention records:", len(mentions))
print("Unique database IDs:", merged["databaseId"].nunique())
print("Unique movie IDs:", mentions["movieId"].nunique())

# ---------------------------------------------------------
# 3. DUPLICATE AUDIT
# ---------------------------------------------------------

merged_duplicates = merged[
    merged.duplicated("databaseId", keep=False)
]

mention_duplicates = mentions[
    mentions.duplicated("movieId", keep=False)
]

print("\n[3] DUPLICATE AUDIT")
print("Merged duplicate records:", len(merged_duplicates))
print("Mention duplicate records:", len(mention_duplicates))

# Do not silently resolve duplicate identifiers.
if len(merged_duplicates) or len(mention_duplicates):
    merged_duplicates.to_csv(
        OUT / "merged_duplicate_ids.csv", index=False
    )
    mention_duplicates.to_csv(
        OUT / "mention_duplicate_ids.csv", index=False
    )
    raise RuntimeError(
        "Duplicate reference identifiers detected. "
        "Inspect duplicate reports before continuing."
    )

# ---------------------------------------------------------
# 4. BUILD NUMERIC ID INVENTORY
# ---------------------------------------------------------

inventory = pd.DataFrame([
    {
        "external_movie_id": external_id,
        "internal_entity_id": internal_id
    }
    for external_id, internal_id in numeric_mapping.items()
])

# ---------------------------------------------------------
# 5. CROSS-SOURCE JOINS
# ---------------------------------------------------------

inventory = inventory.merge(
    mentions[
        ["movieId", "movieName", "nbMentions"]
    ],
    left_on="external_movie_id",
    right_on="movieId",
    how="left",
    validate="one_to_one"
)

inventory = inventory.merge(
    merged[
        ["databaseId", "movieName", "movielensId"]
    ].rename(columns={
        "movieName": "merged_movie_name"
    }),
    left_on="external_movie_id",
    right_on="databaseId",
    how="left",
    validate="one_to_one"
)

inventory["mention_match"] = inventory["movieId"].notna()
inventory["merged_match"] = inventory["databaseId"].notna()

# ---------------------------------------------------------
# 6. TITLE CONSISTENCY
# ---------------------------------------------------------

def normalize_title(value):
    if pd.isna(value):
        return None

    return " ".join(str(value).casefold().split())

inventory["title_match"] = inventory.apply(
    lambda row: (
        normalize_title(row["movieName"])
        == normalize_title(row["merged_movie_name"])
        if row["mention_match"] and row["merged_match"]
        else False
    ),
    axis=1
)

# ---------------------------------------------------------
# 7. VALIDATION SUMMARY
# ---------------------------------------------------------

total = len(inventory)

counts = {
    "numeric_identifiers": total,
    "matched_movie_mentions": int(
        inventory["mention_match"].sum()
    ),
    "matched_merged_metadata": int(
        inventory["merged_match"].sum()
    ),
    "matched_both_sources": int(
        (
            inventory["mention_match"]
            & inventory["merged_match"]
        ).sum()
    ),
    "matching_titles": int(
        inventory["title_match"].sum()
    ),
    "missing_from_mentions": int(
        (~inventory["mention_match"]).sum()
    ),
    "missing_from_merged": int(
        (~inventory["merged_match"]).sum()
    )
}

# ---------------------------------------------------------
# 8. EXPORT RESULTS
# ---------------------------------------------------------

inventory.to_csv(
    OUT / "numeric_movie_identifier_correspondence.csv",
    index=False,
    encoding="utf-8-sig"
)

unmatched = inventory[
    ~(
        inventory["mention_match"]
        & inventory["merged_match"]
        & inventory["title_match"]
    )
]

unmatched.to_csv(
    OUT / "unmatched_or_inconsistent_identifiers.csv",
    index=False,
    encoding="utf-8-sig"
)

manifest = {
    "experiment": "21.13",
    "purpose": "Independent verification of numeric ReDial movie identifiers",
    "counts": counts,
    "validation": {
        "numeric_mapping_unique": (
            len(numeric_mapping) == len(set(numeric_mapping))
        ),
        "internal_entity_ids_unique": (
            len(set(numeric_mapping.values())) == total
        ),
        "all_ids_in_mentions": (
            counts["matched_movie_mentions"] == total
        ),
        "all_ids_in_merged_metadata": (
            counts["matched_merged_metadata"] == total
        ),
        "all_titles_match": (
            counts["matching_titles"] == total
        )
    },
    "interpretation": "PENDING_REVIEW"
}

with (OUT / "experiment_21_13_manifest.json").open(
    "w", encoding="utf-8"
) as f:
    json.dump(manifest, f, indent=4)

# ---------------------------------------------------------
# 9. DISPLAY
# ---------------------------------------------------------

print("\n[4] CROSS-SOURCE CORRESPONDENCE")

for key, value in counts.items():
    print(f"{key}: {value}")

print("\n[5] VALIDATION")

for key, value in manifest["validation"].items():
    print(f"{key}: {'PASS' if value else 'FAIL'}")

print("\n[6] SAMPLE CORRESPONDENCE")

print(
    inventory.head(20)[
        [
            "external_movie_id",
            "internal_entity_id",
            "movieName",
            "merged_movie_name",
            "nbMentions",
            "movielensId"
        ]
    ].to_string(index=False)
)

print("\n[7] OUTPUT DIRECTORY")
print(OUT.resolve())

print("\nEXPERIMENT 21.13 COMPLETED")
