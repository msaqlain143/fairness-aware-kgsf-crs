from pathlib import Path
import json
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent

OUT = ROOT / "results_category_audit" / "experiment_21_16C"
OUT.mkdir(parents=True, exist_ok=True)

MERGED = ROOT / "external_validation/redial_reference/movies_merged.csv"
MENTIONS = ROOT / "external_validation/redial_reference/movies_with_mentions.csv"

UNIFIED = ROOT / "results_category_audit/experiment_21_15/unified_movie_entity_mapping.csv"
CROSSWALK = ROOT / "results_category_audit/experiment_21_14/numeric_movie_semantic_alignment.csv"
URI_AUDIT = ROOT / "results_category_audit/experiment_21_16A/uri_title_candidate_audit.csv"


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required input missing: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalize_title(value):
    value = str(value).strip().casefold()
    return re.sub(r"\s+", " ", value)


def save(df, filename):
    df.to_csv(OUT / filename, index=False, encoding="utf-8-sig")


print("=" * 75)
print("EXPERIMENT 21.16C: DATABASE IDENTIFIER MULTIPLICITY AUDIT")
print("=" * 75)

# ---------------------------------------------------------
# 1. LOAD SOURCE DATA
# ---------------------------------------------------------

merged = load_csv(MERGED)
mentions = load_csv(MENTIONS)
unified = load_csv(UNIFIED)
crosswalk = load_csv(CROSSWALK)
uri_audit = load_csv(URI_AUDIT)

required = {
    "merged": {"databaseId", "movieName", "movielensId"},
    "mentions": {"movieId", "movieName", "nbMentions"},
    "unified": {
        "movie_entity_id", "identity_type",
        "canonical_identifier", "identity_status"
    },
    "crosswalk": {
        "external_numeric_id", "internal_entity_id"
    },
}

for name, frame in [
    ("merged", merged),
    ("mentions", mentions),
    ("unified", unified),
    ("crosswalk", crosswalk),
]:
    missing = required[name] - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")

print("\n[1] SOURCE INVENTORY")
print("Merged records:", len(merged))
print("Mention records:", len(mentions))
print("Unified identities:", len(unified))
print("Verified numeric mappings:", len(crosswalk))
print("URI audit records:", len(uri_audit))

# Work on normalized copies; preserve original files.

for df, id_col, title_col in [
    (merged, "databaseId", "movieName"),
    (mentions, "movieId", "movieName"),
]:
    df["_id"] = df[id_col].str.strip()
    df["_title_norm"] = df[title_col].map(normalize_title)

# ---------------------------------------------------------
# 2. EXACT DUPLICATE ROW AUDIT
# ---------------------------------------------------------

exact_duplicates = merged[
    merged.duplicated(
        subset=["databaseId", "movieName", "movielensId"],
        keep=False
    )
].copy()

save(exact_duplicates, "exact_duplicate_reference_rows.csv")

print("\n[2] EXACT DUPLICATE AUDIT")
print("Exact duplicate rows:", len(exact_duplicates))
print("Unique duplicate records:",
      exact_duplicates[["databaseId", "movieName", "movielensId"]]
      .drop_duplicates().shape[0])

# ---------------------------------------------------------
# 3. DATABASE ID MULTIPLICITY
# ---------------------------------------------------------

valid_merged = merged[merged["_id"] != ""].copy()

db_groups = valid_merged.groupby("_id", sort=False)

db_summary = db_groups.agg(
    record_count=("_id", "size"),
    unique_title_count=("_title_norm", "nunique"),
    unique_movielens_count=("movielensId", "nunique"),
    titles=("movieName", lambda x: " || ".join(sorted(set(x)))),
    movielens_ids=("movielensId",
                   lambda x: " || ".join(sorted(set(x))))
).reset_index()

db_summary = db_summary.rename(
    columns={"_id": "databaseId"}
)

db_summary["multiplicity_status"] = "ONE_RECORD"

db_summary.loc[
    db_summary["record_count"] > 1,
    "multiplicity_status"
] = "MULTIPLE_RECORDS"

db_summary.loc[
    db_summary["unique_title_count"] > 1,
    "multiplicity_status"
] = "MULTIPLE_TITLES"

db_summary.loc[
    db_summary["unique_movielens_count"] > 1,
    "multiplicity_status"
] = "MULTIPLE_MOVIELENS_IDS"

save(db_summary, "database_id_multiplicity.csv")

print("\n[3] DATABASE ID MULTIPLICITY")
print("Unique database IDs:", len(db_summary))
print("IDs with multiple records:",
      (db_summary["record_count"] > 1).sum())
print("IDs with multiple titles:",
      (db_summary["unique_title_count"] > 1).sum())
print("IDs with multiple MovieLens IDs:",
      (db_summary["unique_movielens_count"] > 1).sum())

# ---------------------------------------------------------
# 4. DATABASE ID TITLE VARIATIONS
# ---------------------------------------------------------

title_variations = db_summary[
    db_summary["unique_title_count"] > 1
].copy()

save(title_variations, "database_id_title_variations.csv")

# ---------------------------------------------------------
# 5. MOVIELENS ID INTEGRITY
# ---------------------------------------------------------

valid_ml = valid_merged[
    ~valid_merged["movielensId"].isin(["", "-1", "None"])
].copy()

ml_groups = valid_ml.groupby("movielensId", sort=False)

ml_summary = ml_groups.agg(
    record_count=("databaseId", "size"),
    unique_database_ids=("databaseId", "nunique"),
    unique_titles=("_title_norm", "nunique"),
    database_ids=("databaseId",
                  lambda x: " || ".join(sorted(set(x)))),
    titles=("movieName",
            lambda x: " || ".join(sorted(set(x))))
).reset_index()

ml_summary["integrity_status"] = "ONE_DATABASE_ID"

ml_summary.loc[
    ml_summary["unique_database_ids"] > 1,
    "integrity_status"
] = "MULTIPLE_DATABASE_IDS"

save(ml_summary, "movielens_id_integrity.csv")

print("\n[4] MOVIELENS ID INTEGRITY")
print("Valid MovieLens records:", len(valid_ml))
print("Unique MovieLens IDs:", len(ml_summary))
print("IDs linked to multiple database IDs:",
      (ml_summary["unique_database_ids"] > 1).sum())

# ---------------------------------------------------------
# 6. TITLE COLLISION AUDIT
# ---------------------------------------------------------

title_groups = valid_merged.groupby("_title_norm", sort=False)

title_summary = title_groups.agg(
    record_count=("databaseId", "size"),
    unique_database_ids=("databaseId", "nunique"),
    titles=("movieName", lambda x: " || ".join(sorted(set(x)))),
    database_ids=("databaseId",
                  lambda x: " || ".join(sorted(set(x))))
).reset_index()

title_summary = title_summary.rename(
    columns={"_title_norm": "normalized_title"}
)

title_summary = title_summary[
    title_summary["unique_database_ids"] > 1
].copy()

save(title_summary, "title_collision_audit.csv")

# ---------------------------------------------------------
# 7. RECONCILE DATABASE IDs WITH MENTION IDS
# ---------------------------------------------------------

merged_ids = set(valid_merged["_id"])
mention_ids = set(
    mentions.loc[mentions["_id"] != "", "_id"]
)

intersection = merged_ids & mention_ids

print("\n[5] REFERENCE ID RECONCILIATION")
print("Merged database IDs:", len(merged_ids))
print("Mention movie IDs:", len(mention_ids))
print("Intersection:", len(intersection))
print("Merged-only IDs:", len(merged_ids - mention_ids))
print("Mention-only IDs:", len(mention_ids - merged_ids))

# ---------------------------------------------------------
# 8. IDENTITY RESOLUTION CANDIDATES
# ---------------------------------------------------------

merged_lookup = db_summary.set_index("databaseId").to_dict("index")

mention_lookup = (
    mentions.drop_duplicates("_id")
    .set_index("_id")
    .to_dict("index")
)

crosswalk_ids = set(crosswalk["external_numeric_id"].str.strip())

uri_lookup = (
    uri_audit.drop_duplicates("movie_entity_id")
    .set_index("movie_entity_id")
    .to_dict("index")
)

resolution = []

for _, row in unified.iterrows():

    entity_id = row["movie_entity_id"]
    identity_type = row["identity_type"]
    canonical_id = row["canonical_identifier"].strip()

    dbid = ""
    if identity_type == "Numeric_External_ID":
        dbid = canonical_id

    merged_record = merged_lookup.get(dbid, {})
    mention_record = mention_lookup.get(dbid, {})
    uri_record = uri_lookup.get(entity_id, {})

    if identity_type == "Numeric_External_ID":
        if dbid in crosswalk_ids and dbid in merged_ids:
            status = "NUMERIC_REFERENCE_MATCH"
        else:
            status = "NUMERIC_REFERENCE_UNRESOLVED"
    else:
        status = "URI_IDENTITY_PRESERVED_TITLE_UNVERIFIED"

    resolution.append({
        "movie_entity_id": entity_id,
        "identity_type": identity_type,
        "canonical_identifier": canonical_id,
        "reference_database_id": dbid,
        "reference_record_count": merged_record.get(
            "record_count", 0
        ),
        "reference_title_count": merged_record.get(
            "unique_title_count", 0
        ),
        "reference_titles": merged_record.get("titles", ""),
        "movielens_ids": merged_record.get("movielens_ids", ""),
        "mention_movie_id": (
            mention_record.get("movieId", "")
        ),
        "mention_movie_title": (
            mention_record.get("movieName", "")
        ),
        "mention_count": (
            mention_record.get("nbMentions", "")
        ),
        "uri_candidate_status": (
            uri_record.get("title_match_status", "")
        ),
        "uri_candidate_count": (
            uri_record.get("candidate_count", "")
        ),
        "identity_resolution_status": status
    })

resolution_df = pd.DataFrame(resolution)

save(
    resolution_df,
    "movie_identity_resolution_candidates.csv"
)

# ---------------------------------------------------------
# 9. VALIDATION AND MANIFEST
# ---------------------------------------------------------

validation = {
    "unified_population_6924": len(unified) == 6924,
    "unified_entity_id_unique":
        unified["movie_entity_id"].is_unique,
    "numeric_crosswalk_unique":
        crosswalk["external_numeric_id"].is_unique,
    "mention_ids_unique":
        mentions["_id"].is_unique,
    "database_id_multiplicity_audited":
        len(db_summary) > 0,
    "movielens_multiplicity_audited":
        len(ml_summary) > 0,
    "resolution_population_preserved":
        len(resolution_df) == len(unified),
    "all_numeric_candidates_accounted_for":
        (
            resolution_df["identity_type"]
            == "Numeric_External_ID"
        ).sum() == 812
}
# Convert NumPy boolean values to native Python booleans.
validation = {
    key: bool(value)
    for key, value in validation.items()
}

manifest = {
    "experiment": "21.16C",
    "name": "Database Identifier Multiplicity and Movie Identity Resolution",
    "counts": {
        "merged_records": len(merged),
        "mention_records": len(mentions),
        "unified_entities": len(unified),
        "unique_database_ids": len(db_summary),
        "exact_duplicate_rows": len(exact_duplicates),
        "database_ids_multiple_records":
            int((db_summary["record_count"] > 1).sum()),
        "database_ids_multiple_titles":
            int((db_summary["unique_title_count"] > 1).sum()),
        "database_ids_multiple_movielens":
            int((db_summary["unique_movielens_count"] > 1).sum()),
        "movielens_ids_multiple_database_ids":
            int((ml_summary["unique_database_ids"] > 1).sum()),
        "title_collision_groups": len(title_summary),
        "merged_mention_id_intersection": len(intersection),
        "identity_resolution_records": len(resolution_df)
    },
    "validation": validation,
    "interpretation": {
        "title_matching": "Not sufficient as standalone identity proof.",
        "numeric_mapping": "Preserved using Experiment 21.14.",
        "uri_mapping": "Original canonical URI preserved.",
        "unresolved_records": "Must remain explicitly unresolved."
    },
    "original_files_modified": False,
    "graph_modified": False,
    "model_training": False,
    "model_inference": False
}

with open(
    OUT / "experiment_21_16C_manifest.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(manifest, f, indent=4, ensure_ascii=False)

print("\n" + "=" * 75)
print("EXPERIMENT 21.16C SUMMARY")
print("=" * 75)

print(json.dumps(manifest, indent=4, ensure_ascii=False))

print("\nOUTPUT DIRECTORY:")
print(OUT)

print("\nVALIDATION:")
for key, value in validation.items():
    print(f"{key}: {'PASS' if value else 'FAIL'}")

print("\nEXPERIMENT 21.16C FINISHED.")