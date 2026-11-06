
"""
EXPERIMENT 21.17A
DBpedia URI Extraction and Structural Validation

Purpose:
1. Extract all DBpedia URI-based movie identities.
2. Preserve original canonical identifiers.
3. Parse and normalize URI components.
4. Identify malformed or unexpected URI structures.
5. Generate a reproducible record-level inventory.

Important:
Structural validation is NOT movie identity verification.
No automatic entity merging, graph modification, or model training.
"""

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, unquote

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

SOURCE = (
    ROOT
    / "results_category_audit"
    / "experiment_21_15"
    / "unified_movie_entity_mapping.csv"
)

OUTPUT = (
    ROOT
    / "results_category_audit"
    / "experiment_21_17A"
)

EXPECTED_URI_COUNT = 6112
EXPECTED_TOTAL = 6924

EXPERIMENT = "21.17A"


# ============================================================
# UTILITIES
# ============================================================

def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def find_column(df, candidates):
    lookup = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise KeyError(
        f"Required column not found. "
        f"Candidates: {candidates}. "
        f"Available: {df.columns.tolist()}"
    )


def parse_dbpedia_uri(value):

    result = {
        "uri_scheme": "",
        "uri_host": "",
        "uri_path": "",
        "uri_resource_segment": "",
        "uri_local_label": "",
        "normalized_uri": "",
        "uri_syntax_valid": False,
        "dbpedia_domain_valid": False,
        "resource_path_valid": False,
        "uri_status": "INVALID"
    }

    if not isinstance(value, str) or not value.strip():
        result["uri_status"] = "MISSING_URI"
        return result

    original = value.strip()

    try:
        parsed = urlsplit(original)

        scheme = parsed.scheme.lower()
        host = (parsed.hostname or "").lower()
        path = parsed.path

        segments = [
            segment for segment in path.split("/")
            if segment
        ]

        resource_segment = ""

        if len(segments) >= 2 and segments[0] == "resource":
            resource_segment = segments[1]

        local_label = unquote(resource_segment)

        normalized = (
            f"{scheme}://{host}{path}"
            if scheme and host
            else ""
        )

        syntax_valid = (
            scheme in {"http", "https"}
            and bool(host)
            and not parsed.query
            and not parsed.fragment
        )

        domain_valid = host in {
            "dbpedia.org",
            "www.dbpedia.org"
        }

        resource_valid = (
            len(segments) >= 2
            and segments[0] == "resource"
            and bool(resource_segment)
        )

        result.update({
            "uri_scheme": scheme,
            "uri_host": host,
            "uri_path": path,
            "uri_resource_segment": resource_segment,
            "uri_local_label": local_label,
            "normalized_uri": normalized,
            "uri_syntax_valid": syntax_valid,
            "dbpedia_domain_valid": domain_valid,
            "resource_path_valid": resource_valid
        })

        if not syntax_valid:
            status = "INVALID_URI_SYNTAX"

        elif not domain_valid:
            status = "NON_DBPEDIA_DOMAIN"

        elif not resource_valid:
            status = "INVALID_RESOURCE_PATH"

        else:
            status = "STRUCTURALLY_VALID_DBPEDIA_URI"

        result["uri_status"] = status

        return result

    except Exception as exc:
        result["uri_status"] = (
            "PARSING_ERROR: " + str(exc)
        )
        return result


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 75)
    print("EXPERIMENT 21.17A: DBPEDIA URI EXTRACTION")
    print("AND STRUCTURAL VALIDATION")
    print("=" * 75)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Required source file not found: {SOURCE}"
        )

    print("\n[1] SOURCE INVENTORY")

    df = pd.read_csv(
        SOURCE,
        dtype=str,
        keep_default_na=False
    )

    print("Total identity records:", len(df))
    print("Source columns:", df.columns.tolist())

    entity_col = find_column(
        df, ["movie_entity_id"]
    )

    type_col = find_column(
        df, ["identity_type"]
    )

    canonical_col = find_column(
        df, ["canonical_identifier"]
    )

    # Validate the complete source population.
    if len(df) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL} identities, "
            f"found {len(df)}."
        )

    if df[entity_col].duplicated().any():
        raise RuntimeError(
            "Duplicate internal entity IDs detected."
        )

    print("\n[2] URI POPULATION")

    uri_df = df[
        df[type_col].str.strip().eq("DBpedia_URI")
    ].copy()

    numeric_df = df[
        df[type_col].str.strip().eq("Numeric_External_ID")
    ].copy()

    print("DBpedia URI identities:", len(uri_df))
    print("Numeric identities:", len(numeric_df))

    if len(uri_df) != EXPECTED_URI_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_URI_COUNT} URI identities, "
            f"found {len(uri_df)}."
        )

    if len(numeric_df) != 812:
        raise RuntimeError(
            f"Expected 812 numeric identities, "
            f"found {len(numeric_df)}."
        )

    print("\n[3] URI EXTRACTION")

    records = []

    for _, row in uri_df.iterrows():

        original_uri = row[canonical_col]

        parsed = parse_dbpedia_uri(original_uri)

        record = row.to_dict()
        record.update(parsed)

        record["original_canonical_identifier"] = (
            original_uri
        )

        # No title is inferred from the URI.
        record["identity_verification_status"] = (
            "NOT_INDEPENDENTLY_VERIFIED"
        )

        records.append(record)

    inventory = pd.DataFrame(records)

    print("Extracted URI records:", len(inventory))

    print("\n[4] STRUCTURAL VALIDATION")

    status_counts = (
        inventory["uri_status"]
        .value_counts()
        .to_dict()
    )

    for status, count in status_counts.items():
        print(f"{status}: {count}")

    print("\n[5] URI UNIQUENESS AUDIT")

    print(
        "Unique original URIs:",
        inventory[
            "original_canonical_identifier"
        ].nunique()
    )

    print(
        "Duplicate original URIs:",
        inventory[
            "original_canonical_identifier"
        ].duplicated().sum()
    )

    print(
        "Unique normalized URIs:",
        inventory["normalized_uri"].nunique()
    )

    print("\n[6] DOMAIN AND RESOURCE DISTRIBUTION")

    print(
        inventory["uri_host"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nResource path validity:")
    print(
        inventory["resource_path_valid"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\n[7] STRICT VALIDATION")

    checks = {
        "source_population_6924":
            len(df) == EXPECTED_TOTAL,

        "source_internal_ids_unique":
            not df[entity_col].duplicated().any(),

        "uri_population_6112":
            len(uri_df) == EXPECTED_URI_COUNT,

        "numeric_population_812":
            len(numeric_df) == 812,

        "all_uri_identifiers_nonblank":
            inventory[
                "original_canonical_identifier"
            ].str.strip().ne("").all(),

        "original_uri_identifiers_unique":
            not inventory[
                "original_canonical_identifier"
            ].duplicated().any(),

        "all_uri_records_accounted_for":
            len(inventory) == EXPECTED_URI_COUNT
    }

    for name, passed in checks.items():
        print(
            f"{name}: {'PASS' if passed else 'FAIL'}"
        )

    if not all(checks.values()):
        raise RuntimeError(
            "One or more mandatory validations failed."
        )

    print("\n[8] EXPORTING RESULTS")

    inventory_path = (
        OUTPUT / "dbpedia_uri_record_inventory.csv"
    )

    issues_path = (
        OUTPUT / "dbpedia_uri_structural_issues.csv"
    )

    summary_path = (
        OUTPUT / "dbpedia_uri_structural_summary.csv"
    )

    inventory.to_csv(
        inventory_path,
        index=False,
        encoding="utf-8-sig"
    )

    issues = inventory[
        inventory["uri_status"] !=
        "STRUCTURALLY_VALID_DBPEDIA_URI"
    ].copy()

    issues.to_csv(
        issues_path,
        index=False,
        encoding="utf-8-sig"
    )

    summary = (
        inventory.groupby(
            "uri_status",
            dropna=False
        )
        .size()
        .reset_index(name="record_count")
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )

    source_hash = sha256_file(SOURCE)

    manifest = {
        "experiment": EXPERIMENT,
        "name": "DBpedia URI Extraction and Structural Validation",
        "execution_time_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "source_file": str(
            SOURCE.relative_to(ROOT)
        ),

        "source_sha256": source_hash,

        "counts": {
            "total_movie_entities": int(len(df)),
            "uri_entities": int(len(uri_df)),
            "numeric_entities": int(len(numeric_df)),
            "uri_records_exported": int(len(inventory)),
            "structural_issues": int(len(issues)),
            "structurally_valid": int(
                (
                    inventory["uri_status"]
                    == "STRUCTURALLY_VALID_DBPEDIA_URI"
                ).sum()
            )
        },

        "validation": {
            key: bool(value)
            for key, value in checks.items()
        },

        "methodology": {
            "original_uris_preserved": True,
            "normalized_uris_are_derived_fields": True,
            "uri_local_labels_are_not_authoritative_titles": True,
            "structural_validity_is_not_identity_verification": True,
            "automatic_merging": False,
            "graph_modified": False,
            "model_training": False,
            "model_inference": False,
            "external_metadata_retrieval": False
        },

        "outputs": [
            inventory_path.name,
            issues_path.name,
            summary_path.name
        ],

        "completion_status": "COMPLETED"
    }

    manifest_path = (
        OUTPUT / "experiment_21_17A_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=4,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 75)
    print("EXPERIMENT 21.17A COMPLETED")
    print("=" * 75)

    print("URI identities:", len(inventory))
    print("Structural issues:", len(issues))

    print("\nOUTPUT DIRECTORY:")
    print(OUTPUT.resolve())

    print("\nOriginal files unchanged.")
    print("Knowledge graph unchanged.")
    print("No automatic entity merging.")
    print("No identity verification claims.")

    print("\nEXPERIMENT 21.17A FINISHED.")


if __name__ == "__main__":

    try:
        main()

    except Exception as exc:

        print("\nEXPERIMENT 21.17A FAILED")
        print(type(exc).__name__ + ":", str(exc))

        sys.exit(1)
