
"""
======================================================================
EXPERIMENT 21.15
UNIFIED MOVIE ENTITY IDENTITY RECONSTRUCTION
======================================================================

Research purpose:
Construct a canonical, auditable identity table for all 6,924
movies in the ReDial dataset.

Identity sources:
1. entity2id.json
2. movie_ids.json
3. catalog_metadata_reconciliation.csv
4. Experiment 21.14 semantic alignment results

Identity classes:
- DBpedia URI-based entities
- Verified numeric external identifiers

No fabricated URIs.
No source-data modification.
No graph modification.
No model training or inference.
======================================================================
"""

import json
import re
from pathlib import Path
from collections import Counter
from urllib.parse import unquote

import pandas as pd


# ==============================================================
# 1. CONFIGURATION
# ==============================================================

BASE = Path("data/dataset/redial/nltk")

AUDIT = Path("results_category_audit")

EXP14 = AUDIT / "experiment_21_14"

OUTPUT = AUDIT / "experiment_21_15"

ENTITY_FILE = BASE / "entity2id.json"
MOVIE_FILE = BASE / "movie_ids.json"

CATALOG_FILE = AUDIT / "catalog_metadata_reconciliation.csv"

ALIGNMENT_FILE = EXP14 / "numeric_movie_semantic_alignment.csv"

OUTPUT.mkdir(parents=True, exist_ok=True)

EXPECTED_CATALOG = 6924
EXPECTED_URI = 6112
EXPECTED_NUMERIC = 812


# ==============================================================
# 2. VALIDATION UTILITIES
# ==============================================================

def require(condition, message):

    if not condition:
        raise RuntimeError(message)


def load_json(path):

    require(path.exists(), f"Missing file: {path}")

    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_csv(path, required_columns):

    require(path.exists(), f"Missing file: {path}")

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig"
    )

    missing = set(required_columns) - set(df.columns)

    require(
        not missing,
        f"Missing columns in {path}: {sorted(missing)}"
    )

    return df


def normalize_uri(value):

    value = str(value).strip()

    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1]

    return value.strip()


def normalize_numeric(value):

    value = str(value).strip()

    if not value:
        return None

    require(
        re.fullmatch(r"\d+", value) is not None,
        f"Invalid numeric identifier: {value}"
    )

    return str(int(value))


def uri_local_name(uri):

    """
    Produce a URI-derived label for inspection only.
    This is NOT treated as an authoritative movie title.
    """

    if not uri:
        return ""

    local = uri.rsplit("/", 1)[-1]

    local = unquote(local)

    return local.replace("_", " ")


def integer(value, field):

    try:
        return int(str(value).strip())

    except (ValueError, TypeError):

        raise RuntimeError(
            f"Invalid integer in {field}: {value}"
        )


def unique_dictionary(records, key_field, value_field):

    result = {}

    for record in records:

        key = record[key_field]

        require(
            key not in result,
            f"Duplicate key encountered: {key}"
        )

        result[key] = record[value_field]

    return result


# ==============================================================
# 3. LOAD SOURCE DATA
# ==============================================================

print("=" * 75)
print("EXPERIMENT 21.15: UNIFIED MOVIE ENTITY IDENTITY RECONSTRUCTION")
print("=" * 75)

entity2id = load_json(ENTITY_FILE)

movie_ids = load_json(MOVIE_FILE)

catalog = load_csv(
    CATALOG_FILE,
    [
        "movie_entity_id",
        "movie_uri",
        "mapping_value",
        "mapping_type",
        "resource_uri",
        "external_entity_key"
    ]
)

alignment = load_csv(
    ALIGNMENT_FILE,
    [
        "external_numeric_id",
        "internal_entity_id",
        "merged_movie_title",
        "mention_movie_title",
        "mention_count",
        "in_movie_catalog",
        "title_match",
        "mapping_status"
    ]
)

require(isinstance(entity2id, dict), "Invalid entity2id structure.")

require(isinstance(movie_ids, list), "Invalid movie_ids structure.")

require(
    len(movie_ids) == EXPECTED_CATALOG,
    "Unexpected movie catalog size."
)

movie_ids = [
    integer(x, "movie_ids.json")
    for x in movie_ids
]

movie_set = set(movie_ids)

require(
    len(movie_set) == len(movie_ids),
    "Duplicate internal movie IDs."
)

print("\n[1] SOURCE INVENTORY")

print("Entity mappings:", len(entity2id))
print("Movie catalog:", len(movie_ids))
print("Catalog metadata:", len(catalog))
print("Experiment 21.14 mappings:", len(alignment))


# ==============================================================
# 4. RECONSTRUCT ENTITY INDEX
# ==============================================================

entity_by_internal = {}

for key, value in entity2id.items():

    internal_id = integer(value, "entity2id.json")

    require(
        internal_id not in entity_by_internal,
        f"Duplicate internal entity ID: {internal_id}"
    )

    entity_by_internal[internal_id] = str(key).strip()


print("\n[2] ENTITY INDEX VALIDATION")

print("Unique internal entity IDs:", len(entity_by_internal))

catalog_entities = {
    integer(value, "movie_entity_id")
    for value in catalog["movie_entity_id"]
}

print("Unique catalog metadata IDs:", len(catalog_entities))

require(
    len(catalog_entities) == len(catalog),
    "Duplicate movie_entity_id in catalog metadata."
)

require(
    catalog_entities == movie_set,
    "Catalog metadata IDs do not exactly match movie_ids.json."
)

require(
    movie_set.issubset(set(entity_by_internal)),
    "Some movie IDs are missing from entity2id.json."
)

print("Movie catalog correspondence: PASS")


# ==============================================================
# 5. PREPARE EXPERIMENT 21.14 CROSSWALK
# ==============================================================

numeric_crosswalk = {}

for row in alignment.to_dict("records"):

    external_id = normalize_numeric(
        row["external_numeric_id"]
    )

    internal_id = integer(
        row["internal_entity_id"],
        "alignment.internal_entity_id"
    )

    require(
        external_id not in numeric_crosswalk,
        f"Duplicate numeric alignment: {external_id}"
    )

    numeric_crosswalk[external_id] = row


require(
    len(numeric_crosswalk) == EXPECTED_NUMERIC,
    "Experiment 21.14 does not contain exactly 812 mappings."
)

print("\n[3] NUMERIC CROSSWALK")

print("Verified source records:", len(numeric_crosswalk))


# ==============================================================
# 6. RECONSTRUCT CANONICAL IDENTITIES
# ==============================================================

records = []

for row in catalog.to_dict("records"):

    internal_id = integer(
        row["movie_entity_id"],
        "movie_entity_id"
    )

    entity_key = entity_by_internal[internal_id]

    movie_uri = normalize_uri(row["movie_uri"])

    mapping_value = normalize_uri(row["mapping_value"])

    resource_uri = normalize_uri(row["resource_uri"])

    mapping_type = row["mapping_type"].strip()

    external_key = normalize_numeric(
        row["external_entity_key"]
    )

    require(
        movie_uri,
        f"Missing movie identifier for {internal_id}"
    )

    # ----------------------------------------------------------
    # URI-BASED IDENTITY
    # ----------------------------------------------------------

    if mapping_type == "dbpedia_resource_uri":

        identity_type = "DBpedia_URI"

        canonical_identifier = normalize_uri(entity_key)

        uri_matches = (
            canonical_identifier.startswith(
                "http://dbpedia.org/resource/"
            )
            and movie_uri == canonical_identifier
            and mapping_value == canonical_identifier
            and resource_uri == canonical_identifier
        )

        require(
            external_key is None,
            f"Unexpected numeric key for URI entity {internal_id}"
        )

        identity_verified = uri_matches

        external_id = ""

        movie_title = ""

        mention_count = ""

        title_status = "NOT_AVAILABLE_IN_21_14"

        uri_label = uri_local_name(canonical_identifier)

        provenance = (
            "entity2id_exact_URI_and_catalog_metadata"
        )

    # ----------------------------------------------------------
    # NUMERIC-BASED IDENTITY
    # ----------------------------------------------------------

    elif mapping_type == "numeric_external_identifier":

        identity_type = "Numeric_External_ID"

        external_id = normalize_numeric(entity_key)

        canonical_identifier = external_id

        crosswalk = numeric_crosswalk.get(external_id)

        require(
            crosswalk is not None,
            f"Numeric entity missing from Experiment 21.14: {external_id}"
        )

        title_match = (
            crosswalk["title_match"].strip().lower() == "true"
        )

        catalog_match = (
            crosswalk["in_movie_catalog"].strip().lower() == "true"
        )

        internal_match = (
            integer(
                crosswalk["internal_entity_id"],
                "alignment.internal_entity_id"
            ) == internal_id
        )

        external_consistency = (
            movie_uri == external_id
            and mapping_value == external_id
            and external_key == external_id
            and resource_uri == ""
        )

        identity_verified = all([
            title_match,
            catalog_match,
            internal_match,
            external_consistency
        ])

        movie_title = crosswalk["merged_movie_title"]

        mention_title = crosswalk["mention_movie_title"]

        mention_count = crosswalk["mention_count"]

        title_status = (
            "MATCHED"
            if title_match
            and movie_title
            and mention_title
            else "REVIEW"
        )

        uri_label = ""

        provenance = (
            "entity2id_numeric_key_plus_"
            "experiment_21_14_cross_source_verification"
        )

    else:

        raise RuntimeError(
            f"Unexpected mapping_type: {mapping_type}"
        )

    canonical_id = (
        f"dbpedia:{canonical_identifier}"
        if identity_type == "DBpedia_URI"
        else f"redial_numeric:{canonical_identifier}"
    )

    status = (
        "VERIFIED"
        if identity_verified
        else "REVIEW"
    )

    records.append({

        "movie_entity_id": internal_id,

        "canonical_entity_id": canonical_id,

        "identity_type": identity_type,

        "canonical_identifier": canonical_identifier,

        "original_entity2id_key": entity_key,

        "original_movie_uri": movie_uri,

        "mapping_value": mapping_value,

        "resource_uri": resource_uri,

        "external_numeric_id": external_id,

        "uri_local_name": uri_label,

        "movie_title": movie_title,

        "mention_count": mention_count,

        "title_status": title_status,

        "identity_provenance": provenance,

        "identity_status": status

    })


identity_df = pd.DataFrame(records)


# ==============================================================
# 7. IDENTITY DISTRIBUTION
# ==============================================================

print("\n[4] IDENTITY DISTRIBUTION")

distribution = Counter(identity_df["identity_type"])

for identity_type, count in distribution.items():

    percentage = count / len(identity_df) * 100

    print(
        f"{identity_type}: {count} "
        f"({percentage:.2f}%)"
    )

print("\nIdentity status:")

print(
    identity_df["identity_status"]
    .value_counts()
    .to_string()
)


# ==============================================================
# 8. INTEGRITY AUDIT
# ==============================================================

print("\n[5] CANONICAL IDENTITY INTEGRITY")

duplicate_internal = int(
    identity_df["movie_entity_id"].duplicated().sum()
)

duplicate_canonical = int(
    identity_df["canonical_entity_id"].duplicated().sum()
)

missing_canonical = int(
    identity_df["canonical_identifier"].eq("").sum()
)

missing_provenance = int(
    identity_df["identity_provenance"].eq("").sum()
)

uri_df = identity_df[
    identity_df["identity_type"] == "DBpedia_URI"
]

numeric_df = identity_df[
    identity_df["identity_type"] == "Numeric_External_ID"
]

uri_integrity = bool(
    (uri_df["identity_status"] == "VERIFIED").all()
)

numeric_integrity = bool(
    (numeric_df["identity_status"] == "VERIFIED").all()
)

print("Duplicate internal IDs:", duplicate_internal)

print("Duplicate canonical IDs:", duplicate_canonical)

print("Missing canonical identifiers:", missing_canonical)

print("Missing provenance:", missing_provenance)

print("URI identity integrity:", uri_integrity)

print("Numeric identity integrity:", numeric_integrity)


# ==============================================================
# 9. STRICT VALIDATION
# ==============================================================

validation = {

    "catalog_size": len(identity_df) == EXPECTED_CATALOG,

    "catalog_internal_id_uniqueness": duplicate_internal == 0,

    "catalog_internal_id_completeness": (
        set(identity_df["movie_entity_id"]) == movie_set
    ),

    "canonical_identifier_uniqueness": duplicate_canonical == 0,

    "canonical_identifier_completeness": missing_canonical == 0,

    "provenance_completeness": missing_provenance == 0,

    "uri_population": len(uri_df) == EXPECTED_URI,

    "numeric_population": len(numeric_df) == EXPECTED_NUMERIC,

    "uri_identity_integrity": uri_integrity,

    "numeric_identity_integrity": numeric_integrity,

    "all_entities_verified": bool(
        (identity_df["identity_status"] == "VERIFIED").all()
    )

}

print("\n[6] STRICT VALIDATION")

for name, passed in validation.items():

    print(
        f"{name}: {'PASS' if passed else 'FAIL'}"
    )

all_passed = all(validation.values())


# ==============================================================
# 10. EXPORT COMPLETE IDENTITY TABLE
# ==============================================================

identity_path = (
    OUTPUT / "unified_movie_entity_mapping.csv"
)

identity_df.to_csv(
    identity_path,
    index=False,
    encoding="utf-8-sig"
)


# ==============================================================
# 11. EXPORT IDENTITY AUDIT
# ==============================================================

audit_df = identity_df[
    [
        "movie_entity_id",
        "canonical_entity_id",
        "identity_type",
        "canonical_identifier",
        "identity_provenance",
        "identity_status"
    ]
].copy()

audit_path = (
    OUTPUT / "unified_movie_identity_audit.csv"
)

audit_df.to_csv(
    audit_path,
    index=False,
    encoding="utf-8-sig"
)


# ==============================================================
# 12. GENERATE MANIFEST
# ==============================================================

manifest = {

    "experiment": "21.15",

    "name": "Unified Movie Entity Identity Reconstruction",

    "purpose": (
        "Construct a canonical identity table for all "
        "6,924 ReDial movie entities."
    ),

    "counts": {

        "total_movies": len(identity_df),

        "dbpedia_uri_entities": len(uri_df),

        "numeric_external_id_entities": len(numeric_df),

        "verified_entities": int(
            (identity_df["identity_status"] == "VERIFIED").sum()
        ),

        "entities_requiring_review": int(
            (identity_df["identity_status"] == "REVIEW").sum()
        ),

        "unique_internal_ids": int(
            identity_df["movie_entity_id"].nunique()
        ),

        "unique_canonical_ids": int(
            identity_df["canonical_entity_id"].nunique()
        )

    },

    "validation": validation,

    "completion_status": (
        "COMPLETE_ALL_VALIDATIONS_PASSED"
        if all_passed
        else "INCOMPLETE_REQUIRES_INVESTIGATION"
    ),

    "identity_policy": {

        "uri_entities": (
            "Preserve original DBpedia resource URIs."
        ),

        "numeric_entities": (
            "Preserve verified numeric external identifiers."
        ),

        "fabricated_uris": False,

        "original_internal_ids_preserved": True,

        "uri_local_names_are_authoritative_titles": False,

        "numeric_movie_titles_source": "Experiment 21.14"

    },

    "limitations": [

        "URI identity validation establishes correspondence "
        "with the existing DBpedia URI metadata, not independent "
        "verification of every movie title.",

        "Movie titles are populated from Experiment 21.14 "
        "for numeric entities only.",

        "URI-derived labels are not treated as authoritative titles.",

        "No new external metadata has been retrieved."

    ],

    "outputs": [

        "unified_movie_entity_mapping.csv",

        "unified_movie_identity_audit.csv",

        "unified_movie_identity_manifest.json"

    ]

}

manifest_path = (
    OUTPUT / "unified_movie_identity_manifest.json"
)

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


# ==============================================================
# 13. FINAL REPORT
# ==============================================================

print("\n" + "=" * 75)

print("EXPERIMENT 21.15 COMPLETED")

print("=" * 75)

print("Total movie entities:", len(identity_df))

print("DBpedia URI entities:", len(uri_df))

print("Numeric external entities:", len(numeric_df))

print(
    "Verified entities:",
    manifest["counts"]["verified_entities"]
)

print(
    "Entities requiring review:",
    manifest["counts"]["entities_requiring_review"]
)

print("\nOUTPUT FILES")

print(identity_path)

print(audit_path)

print(manifest_path)

print("\nOriginal source files unchanged.")

print("No graph modification.")

print("No model training or inference.")

print("\nFinal status:", manifest["completion_status"])
print("\nEXPERIMENT 21.15 FINISHED.")
