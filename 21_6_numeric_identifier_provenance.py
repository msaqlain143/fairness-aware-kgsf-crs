
import csv
import json
import pickle
import re
from pathlib import Path
from collections import Counter

# ============================================================
# EXPERIMENT 21.6
# NUMERIC EXTERNAL-IDENTIFIER PROVENANCE INVESTIGATION
#
# Purpose:
# 1. Validate all numeric external identifiers.
# 2. Test their overlap with internal graph namespaces.
# 3. Trace identifier-generation logic in project source code.
# 4. Inventory candidate original source files.
# 5. Preserve reproducible audit outputs.
#
# No graph modification, training, or inference.
# ============================================================

ROOT = Path(".")

BASE = ROOT / "data/dataset/redial/nltk"

METADATA_FILE = (
    ROOT / "results_category_audit/"
    "catalog_metadata_reconciliation.csv"
)

CATALOG_FILE = ROOT / "catalog_audit_corrected.pkl"

ENTITY_FILE = BASE / "entity2id.json"
KG_FILE = BASE / "dbpedia_subkg.json"

OUTPUT = (
    ROOT / "results_category_audit/"
    "numeric_identifier_provenance_v1"
)

EXPECTED_NUMERIC_RECORDS = 812
EXPECTED_CATALOG = 6924

OUTPUT.mkdir(parents=True, exist_ok=True)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def parse_numeric_id(value):
    """Parse only non-negative integer-valued numeric identifiers."""

    value = str(value).strip()

    if not value:
        return None

    if not re.fullmatch(r"\d+(?:\.0+)?", value):
        return None

    return int(float(value))


def save_csv(path, rows, fields=None):

    if fields is None:
        fields = list(rows[0].keys()) if rows else []

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 1. LOAD ORIGINAL DATA
# ============================================================

print("=" * 75)
print("EXPERIMENT 21.6: NUMERIC IDENTIFIER PROVENANCE")
print("=" * 75)

with ENTITY_FILE.open(encoding="utf-8") as f:
    entity2id = json.load(f)

with KG_FILE.open(encoding="utf-8") as f:
    kg = json.load(f)

with CATALOG_FILE.open("rb") as f:
    catalog = pickle.load(f)

with METADATA_FILE.open(
    encoding="utf-8-sig",
    newline=""
) as f:
    metadata = list(csv.DictReader(f))

movie_ids = set(map(int, catalog["movie_ids"]))

internal_entity_ids = set(
    map(int, entity2id.values())
)

kg_node_ids = set(map(int, kg.keys()))

kg_tail_ids = {
    int(tail)
    for edges in kg.values()
    for relation, tail in edges
}

kg_all_node_ids = kg_node_ids | kg_tail_ids

# Reconstruct relation-87 entity set.
relation87_ids = set()

for head, edges in kg.items():

    for relation, tail in edges:

        if int(relation) == 87:
            relation87_ids.add(int(head))


print("\n[1] SOURCE DATA")

print("Catalog:", len(movie_ids))
print("Entity index:", len(entity2id))
print("Internal entity IDs:", len(internal_entity_ids))
print("KG head IDs:", len(kg_node_ids))
print("KG node IDs:", len(kg_all_node_ids))
print("Relation 87 entities:", len(relation87_ids))

require(
    len(movie_ids) == EXPECTED_CATALOG,
    "Unexpected catalog size."
)

# ============================================================
# 2. IDENTIFY NUMERIC EXTERNAL IDENTIFIERS
# ============================================================

numeric_rows = []

for row in metadata:

    mapping_type = str(
        row["mapping_type"]
    ).strip()

    if mapping_type != "numeric_external_identifier":
        continue

    mapping_id = parse_numeric_id(
        row["mapping_value"]
    )

    external_key = parse_numeric_id(
        row["external_entity_key"]
    )

    require(
        mapping_id is not None,
        f"Invalid numeric mapping: {row}"
    )

    movie_id = int(row["movie_entity_id"])

    numeric_rows.append({
        "movie_entity_id": movie_id,
        "movie_uri": row["movie_uri"],
        "mapping_value": mapping_id,
        "external_entity_key": external_key,
        "mapping_type": mapping_type,
        "resource_uri": row["resource_uri"],
        "mapping_equals_external_key": (
            mapping_id == external_key
        ),
        "in_internal_entity_index": (
            mapping_id in internal_entity_ids
        ),
        "in_kg_head_namespace": (
            mapping_id in kg_node_ids
        ),
        "in_kg_node_namespace": (
            mapping_id in kg_all_node_ids
        ),
        "in_movie_catalog_namespace": (
            mapping_id in movie_ids
        ),
        "in_relation87_namespace": (
            mapping_id in relation87_ids
        ),
    })


numeric_ids = [
    row["mapping_value"]
    for row in numeric_rows
]

require(
    len(numeric_rows) == EXPECTED_NUMERIC_RECORDS,
    (
        f"Expected {EXPECTED_NUMERIC_RECORDS} numeric records; "
        f"found {len(numeric_rows)}."
    )
)

require(
    len(set(numeric_ids)) == len(numeric_ids),
    "Duplicate numeric external identifiers detected."
)

require(
    len({r["movie_entity_id"] for r in numeric_rows})
    == len(numeric_rows),
    "Duplicate movie records detected."
)

require(
    all(
        r["mapping_equals_external_key"]
        for r in numeric_rows
    ),
    "Mapping value and external key mismatch."
)

require(
    all(
        r["movie_entity_id"] in movie_ids
        for r in numeric_rows
    ),
    "Numeric mapping references an unknown catalog movie."
)

print("\n[2] NUMERIC IDENTIFIER INTEGRITY")

print("Numeric records:", len(numeric_rows))
print("Unique identifiers:", len(set(numeric_ids)))
print(
    "Mapping/external-key equality:",
    all(r["mapping_equals_external_key"] for r in numeric_rows)
)

# ============================================================
# 3. TEST IDENTIFIER NAMESPACE OVERLAP
# ============================================================

overlap_fields = [
    "in_internal_entity_index",
    "in_kg_head_namespace",
    "in_kg_node_namespace",
    "in_movie_catalog_namespace",
    "in_relation87_namespace",
]

namespace_results = {}

print("\n[3] IDENTIFIER NAMESPACE OVERLAP")

for field in overlap_fields:

    count = sum(
        bool(row[field])
        for row in numeric_rows
    )

    namespace_results[field] = {
        "matching_records": count,
        "total_records": len(numeric_rows),
        "overlap_ratio": count / len(numeric_rows),
    }

    print(
        f"{field}: {count}/{len(numeric_rows)} "
        f"({count / len(numeric_rows):.4%})"
    )

# ============================================================
# 4. IDENTIFIER DISTRIBUTION
# ============================================================

identifier_distribution = {
    "minimum": min(numeric_ids),
    "maximum": max(numeric_ids),
    "unique": len(set(numeric_ids)),
    "mean": sum(numeric_ids) / len(numeric_ids),
}

print("\n[4] IDENTIFIER DISTRIBUTION")

for key, value in identifier_distribution.items():
    print(f"{key}: {value}")

# ============================================================
# 5. SOURCE CODE PROVENANCE SCAN
# ============================================================

print("\n[5] SOURCE CODE PROVENANCE SCAN")

search_terms = [
    "numeric_external_identifier",
    "external_entity_key",
    "mapping_value",
    "mapping_type",
    "catalog_metadata_reconciliation",
]

source_hits = []

excluded_dirs = {
    ".venv",
    ".git",
    "__pycache__",
    "node_modules",
}

for path in ROOT.rglob("*.py"):

    if any(part in excluded_dirs for part in path.parts):
        continue

    try:
        lines = path.read_text(
            encoding="utf-8",
            errors="ignore"
        ).splitlines()

    except OSError:
        continue

    for line_number, line in enumerate(lines, start=1):

        for term in search_terms:

            if term.lower() in line.lower():

                source_hits.append({
                    "file": str(path),
                    "line_number": line_number,
                    "matched_term": term,
                    "source_line": line.strip(),
                })

                break


print("Source-code matches:", len(source_hits))

# ============================================================
# 6. CANDIDATE SOURCE FILE INVENTORY
# ============================================================

print("\n[6] CANDIDATE SOURCE FILE INVENTORY")

filename_pattern = re.compile(
    r"mapping|entity|movie|dbpedia|catalog|redial",
    re.IGNORECASE
)

candidate_files = []

for path in ROOT.rglob("*"):

    if not path.is_file():
        continue

    if any(part in excluded_dirs for part in path.parts):
        continue

    if filename_pattern.search(path.name):

        candidate_files.append({
            "file": str(path),
            "extension": path.suffix,
            "size_bytes": path.stat().st_size,
        })


candidate_files.sort(
    key=lambda x: x["file"].lower()
)

print("Candidate files:", len(candidate_files))

# ============================================================
# 7. EXPORT DETAILED RESULTS
# ============================================================

save_csv(
    OUTPUT / "numeric_identifier_inventory.csv",
    numeric_rows
)

save_csv(
    OUTPUT / "identifier_source_code_hits.csv",
    source_hits
)

save_csv(
    OUTPUT / "candidate_source_files.csv",
    candidate_files
)

# ============================================================
# 8. MANIFEST
# ============================================================

summary = {
    "experiment": "21.6",
    "purpose": (
        "Investigate the provenance and namespace of "
        "numeric external identifiers."
    ),
    "counts": {
        "catalog": len(movie_ids),
        "numeric_records": len(numeric_rows),
        "unique_numeric_identifiers": len(set(numeric_ids)),
        "source_code_hits": len(source_hits),
        "candidate_source_files": len(candidate_files),
    },
    "identifier_distribution": identifier_distribution,
    "namespace_overlap": namespace_results,
    "validation": {
        "catalog_size": True,
        "numeric_record_count": True,
        "numeric_identifier_uniqueness": True,
        "movie_record_uniqueness": True,
        "mapping_external_key_consistency": True,
        "catalog_membership": True,
    },
    "interpretation_status": (
        "PENDING_SOURCE_PROVENANCE_VERIFICATION"
    ),
    "limitations": [
        "Namespace overlap does not establish identifier semantics.",
        "Source-code matches do not independently prove provenance.",
        "Numeric identifiers require confirmation against their "
        "original generating source."
    ],
}

with (
    OUTPUT / "numeric_identifier_provenance_manifest.json"
).open("w", encoding="utf-8") as f:

    json.dump(summary, f, indent=2)

# ============================================================
# 9. FINAL REPORT
# ============================================================

print("\n" + "=" * 75)
print("EXPERIMENT 21.6 COMPLETED")
print("=" * 75)

print("Numeric records:", len(numeric_rows))
print("Unique numeric identifiers:", len(set(numeric_ids)))

print("\nNamespace overlap:")

for key, value in namespace_results.items():
    print(
        key,
        value["matching_records"],
        "/",
        value["total_records"]
    )

print("\nSource-code hits:", len(source_hits))
print("Candidate source files:", len(candidate_files))

print("\nOUTPUT DIRECTORY:")
print(OUTPUT.resolve())

print("\nNo source data modified.")
print("No graph modified.")
print("No model training or inference performed.")
print("Identifier semantic provenance remains unverified.")
