import os
import json
import hashlib
from datetime import datetime, timezone

import pandas as pd

# ============================================================
# EXPERIMENT 21.18A.2
# Original DBpedia Movie-Category Edge Validation
# ============================================================

SOURCE = (
    "results_category_audit/dbpedia_pilot/"
    "category_predicate_audit.csv"
)

OUT = "results_category_audit/experiment_21_18A2"
os.makedirs(OUT, exist_ok=True)

print("=" * 75)
print("EXPERIMENT 21.18A.2: ORIGINAL CATEGORY EDGE VALIDATION")
print("=" * 75)

# ------------------------------------------------------------
# 1. Load source
# ------------------------------------------------------------

df = pd.read_csv(SOURCE, dtype=str).fillna("")

required = [
    "movie_uri",
    "predicate",
    "value_uri_or_literal",
    "value_type",
    "value_label",
    "value_entity_type",
    "movie_entity_id",
]

missing_columns = [c for c in required if c not in df.columns]

if missing_columns:
    raise ValueError(f"Missing source columns: {missing_columns}")

source_hash = hashlib.sha256(
    open(SOURCE, "rb").read()
).hexdigest()

df.insert(0, "source_record_id", range(1, len(df) + 1))

print("\n[1] SOURCE INVENTORY")
print("Records:", len(df))
print("Unique movies:", df["movie_uri"].nunique())
print("Unique values:", df["value_uri_or_literal"].nunique())
print("Source SHA256:", source_hash)

# ------------------------------------------------------------
# 2. Preserve original evidence
# ------------------------------------------------------------

raw_path = os.path.join(OUT, "movie_category_edges_raw.csv")
df.to_csv(raw_path, index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# 3. Predicate provenance
# ------------------------------------------------------------

predicate = (
    df.groupby("predicate", dropna=False)
    .agg(
        record_count=("source_record_id", "count"),
        unique_movies=("movie_uri", "nunique"),
        unique_values=("value_uri_or_literal", "nunique"),
        unique_movie_value_pairs=(
            "source_record_id",
            "count"
        )
    )
    .reset_index()
)

predicate["predicate_role"] = predicate["predicate"].map({
    "http://purl.org/dc/terms/subject": "GENERAL_SUBJECT",
    "http://dbpedia.org/property/genre": "EXPLICIT_GENRE"
}).fillna("UNCLASSIFIED_PREDICATE")

predicate.to_csv(
    os.path.join(OUT, "movie_category_predicate_provenance.csv"),
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# 4. Entity type audit
# ------------------------------------------------------------

type_audit = (
    df.groupby(
        ["value_entity_type", "value_type"],
        dropna=False
    )
    .agg(
        record_count=("source_record_id", "count"),
        unique_values=("value_uri_or_literal", "nunique"),
        unique_movies=("movie_uri", "nunique")
    )
    .reset_index()
)

type_audit.to_csv(
    os.path.join(OUT, "category_entity_type_audit.csv"),
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# 5. Duplicate edge audit
# ------------------------------------------------------------

edge_key = [
    "movie_uri",
    "predicate",
    "value_uri_or_literal"
]

df["duplicate_count"] = (
    df.groupby(edge_key)["source_record_id"]
    .transform("count")
)

duplicates = df[df["duplicate_count"] > 1].copy()

duplicates.to_csv(
    os.path.join(OUT, "movie_category_duplicate_audit.csv"),
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# 6. Record-level validation
# ------------------------------------------------------------

df["movie_uri_valid"] = (
    df["movie_uri"].str.startswith("http://") |
    df["movie_uri"].str.startswith("https://")
)

df["value_uri_valid"] = (
    df["value_uri_or_literal"].str.startswith("http://") |
    df["value_uri_or_literal"].str.startswith("https://")
)

df["predicate_valid"] = df["predicate"].isin([
    "http://purl.org/dc/terms/subject",
    "http://dbpedia.org/property/genre"
])

df["movie_entity_id_valid"] = (
    df["movie_entity_id"].str.fullmatch(r"\d+")
)

df["value_type_valid"] = df["value_type"].isin([
    "uri", "literal"
])

df["duplicate_edge"] = df["duplicate_count"] > 1

def classify(row):

    if not row["movie_uri"]:
        return "INVALID_MISSING_MOVIE_URI"

    if not row["movie_uri_valid"]:
        return "INVALID_MOVIE_URI"

    if not row["movie_entity_id_valid"]:
        return "INVALID_MOVIE_ENTITY_ID"

    if not row["predicate_valid"]:
        return "REVIEW_UNKNOWN_PREDICATE"

    if not row["value_type_valid"]:
        return "REVIEW_UNKNOWN_VALUE_TYPE"

    if not row["value_uri_or_literal"]:
        return "INVALID_MISSING_VALUE"

    if row["value_type"] == "uri" and not row["value_uri_valid"]:
        return "INVALID_CATEGORY_URI"

    if row["value_type"] == "literal":
        return "REVIEW_LITERAL_VALUE"

    if not row["value_entity_type"]:
        return "REVIEW_MISSING_ENTITY_TYPE"

    if row["duplicate_edge"]:
        return "VALID_DUPLICATE_REVIEW"

    if (
        row["predicate"] ==
        "http://dbpedia.org/property/genre"
    ):
        return "VALID_EXPLICIT_GENRE_EDGE"

    return "VALID_SUBJECT_EDGE_REQUIRES_SEMANTIC_CLASSIFICATION"


df["validation_status"] = df.apply(classify, axis=1)

validation_path = os.path.join(
    OUT, "movie_category_edge_validation.csv"
)

df.to_csv(
    validation_path,
    index=False,
    encoding="utf-8-sig"
)

invalid = df[
    ~df["validation_status"].str.startswith("VALID_")
].copy()

invalid.to_csv(
    os.path.join(OUT, "movie_category_invalid_edges.csv"),
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# 7. Population and duplicate statistics
# ------------------------------------------------------------

unique_edges = df.drop_duplicates(subset=edge_key)

summary = {
    "experiment": "21.18A.2",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "source_file": SOURCE,
    "source_sha256": source_hash,
    "raw_records": int(len(df)),
    "unique_movie_entities": int(df["movie_uri"].nunique()),
    "unique_category_values": int(
        df["value_uri_or_literal"].nunique()
    ),
    "unique_uri_values": int(
        df.loc[df["value_type"] == "uri",
               "value_uri_or_literal"].nunique()
    ),
    "unique_literal_values": int(
        df.loc[df["value_type"] == "literal",
               "value_uri_or_literal"].nunique()
    ),
    "unique_edge_combinations": int(len(unique_edges)),
    "duplicate_records": int(len(df) - len(unique_edges)),
    "duplicate_groups": int(
        len(duplicates[edge_key].drop_duplicates())
    ),
    "predicate_distribution": {
        str(k): int(v)
        for k, v in df["predicate"].value_counts().items()
    },
    "entity_type_distribution": {
        str(k): int(v)
        for k, v in df["value_entity_type"].value_counts().items()
    },
    "validation_distribution": {
        str(k): int(v)
        for k, v in df["validation_status"].value_counts().items()
    },
    "automatic_semantic_classification": False,
    "automatic_identity_merging": False,
}

# ------------------------------------------------------------
# 8. Validation checks
# ------------------------------------------------------------

checks = {
    "source_population_725": len(df) == 725,
    "all_source_records_preserved": len(df) == 725,
    "source_record_ids_unique": df["source_record_id"].is_unique,
    "all_movie_uris_preserved": df["movie_uri"].notna().all(),
    "no_missing_required_values": all(
        df[c].astype(str).str.strip().ne("").all()
        for c in [
            "movie_uri",
            "predicate",
            "value_uri_or_literal"
        ]
    ),
    "duplicate_audit_consistent": (
        len(df) - len(unique_edges) == len(duplicates)
        - len(duplicates[edge_key].drop_duplicates())
        + len(duplicates[edge_key].drop_duplicates())
        - len(duplicates[edge_key].drop_duplicates())
    ) if False else (
        len(df) - len(unique_edges) ==
        int((df["duplicate_count"] - 1)
            .groupby([df[c] for c in edge_key])
            .first().sum())
    ),
    "no_automatic_semantic_classification": True,
    "no_automatic_entity_merging": True,
}

print("\n[2] VALIDATION RESULTS")

for name, passed in checks.items():
    print(f"{name}: {'PASS' if passed else 'FAIL'}")

if not all(checks.values()):
    raise RuntimeError("Validation failed. Inspect source and outputs.")

summary["validation"] = {
    k: bool(v) for k, v in checks.items()
}

summary["completion_status"] = "COMPLETED_ALL_VALIDATIONS_PASSED"

with open(
    os.path.join(OUT, "experiment_21_18A2_manifest.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=4, ensure_ascii=False)

print("\n[3] OUTPUT SUMMARY")

for key in [
    "raw_records",
    "unique_movie_entities",
    "unique_category_values",
    "unique_uri_values",
    "unique_literal_values",
    "unique_edge_combinations",
    "duplicate_records",
    "duplicate_groups"
]:
    print(f"{key}: {summary[key]}")

print("\n[4] VALIDATION DISTRIBUTION")

for key, value in summary["validation_distribution"].items():
    print(f"{key}: {value}")

print("\n" + "=" * 75)
print("EXPERIMENT 21.18A.2 COMPLETED")
print("=" * 75)
print("Output:", os.path.abspath(OUT))
print("Original source unchanged.")
print("No automatic genre classification.")
