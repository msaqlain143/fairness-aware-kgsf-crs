
import csv
import json
from pathlib import Path

ROOT = Path(".")

METADATA = (
    ROOT / "results_category_audit"
    / "catalog_metadata_reconciliation.csv"
)

PROVENANCE = (
    ROOT / "results_category_audit"
    / "entity_uri_provenance_v1"
    / "entity_uri_provenance.csv"
)

NUMERIC_INVENTORY = (
    ROOT / "results_category_audit"
    / "numeric_identifier_provenance_v1"
    / "numeric_identifier_inventory.csv"
)

OUTPUT = (
    ROOT / "results_category_audit"
    / "metadata_consistency_v1"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

TARGET_MOVIES = {18405, 18537, 18637}
TARGET_NUMERIC_IDS = {75962, 76488, 76845}


def load_csv(path):
    with path.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def print_rows(title, rows):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)

    for row in rows:
        for key, value in row.items():
            print(f"{key}: {value}")
        print("-" * 50)


# ============================================================
# 1. LOAD FILES
# ============================================================

metadata_fields, metadata_rows = load_csv(METADATA)
inventory_fields, inventory_rows = load_csv(NUMERIC_INVENTORY)
provenance_fields, provenance_rows = load_csv(PROVENANCE)

print("=" * 75)
print("EXPERIMENT 21.7: METADATA CONSISTENCY AUDIT")
print("=" * 75)

print("\n[1] METADATA SCHEMAS")

for name, fields in [
    ("Catalog metadata", metadata_fields),
    ("Numeric inventory", inventory_fields),
    ("URI provenance", provenance_fields),
]:
    print(f"\n{name}:")
    print(fields)


# ============================================================
# 2. EXAMINE ORIGINAL METADATA
# ============================================================

selected_metadata = [
    row for row in metadata_rows
    if row.get("movie_entity_id", "").strip()
    in {str(x) for x in TARGET_MOVIES}
]

print_rows(
    "[2] ORIGINAL CATALOG METADATA",
    selected_metadata
)


# ============================================================
# 3. EXAMINE NUMERIC INVENTORY
# ============================================================

selected_inventory = [
    row for row in inventory_rows
    if row.get("movie_entity_id", "").strip()
    in {str(x) for x in TARGET_MOVIES}
]

print_rows(
    "[3] NUMERIC INVENTORY",
    selected_inventory
)


# ============================================================
# 4. EXAMINE PREVIOUS URI PROVENANCE
# ============================================================

selected_provenance = []

for row in provenance_rows:

    values = set(row.values())

    if (
        values & {str(x) for x in TARGET_MOVIES}
        or any(
            value and any(
                value.rstrip("/").endswith(str(x))
                for x in TARGET_MOVIES
            )
            for value in values
        )
    ):
        selected_provenance.append(row)

print_rows(
    "[4] PREVIOUS URI PROVENANCE RECORDS",
    selected_provenance
)


# ============================================================
# 5. FIELD EQUALITY AUDIT
# ============================================================

print("\n[5] FIELD EQUALITY AUDIT")

for column in [
    "movie_uri",
    "mapping_value",
    "external_entity_key",
    "resource_uri",
]:

    available = [
        row.get(column, "").strip()
        for row in metadata_rows
        if row.get(column, "").strip()
    ]

    print(f"\n{column}")
    print("Non-null:", len(available))
    print("Unique:", len(set(available)))

    numeric_count = sum(
        value.isdigit()
        for value in available
    )

    print("Numeric values:", numeric_count)
    print("URI-like values:", sum(
        value.startswith(("http://", "https://"))
        for value in available
    ))


# ============================================================
# 6. NUMERIC FIELD RELATIONSHIPS
# ============================================================

numeric_rows = [
    row for row in metadata_rows
    if row.get("mapping_type", "").strip()
    == "numeric_external_identifier"
]

comparisons = [
    ("movie_uri", "mapping_value"),
    ("movie_uri", "external_entity_key"),
    ("mapping_value", "external_entity_key"),
]

print("\n[6] NUMERIC FIELD RELATIONSHIPS")

for first, second in comparisons:

    matches = sum(
        row.get(first, "").strip()
        == row.get(second, "").strip()
        for row in numeric_rows
    )

    print(
        f"{first} == {second}: "
        f"{matches}/{len(numeric_rows)}"
    )


# ============================================================
# 7. SAVE AUDIT SUMMARY
# ============================================================

summary = {
    "experiment": "21.7",
    "metadata_columns": metadata_fields,
    "inventory_columns": inventory_fields,
    "provenance_columns": provenance_fields,
    "numeric_records": len(numeric_rows),
    "selected_metadata_records": len(selected_metadata),
    "selected_inventory_records": len(selected_inventory),
    "selected_provenance_records": len(selected_provenance),
    "interpretation": "PENDING",
}

with (
    OUTPUT / "metadata_consistency_summary.json"
).open("w", encoding="utf-8") as f:

    json.dump(summary, f, indent=2)

print("\n" + "=" * 75)
print("EXPERIMENT 21.7 COMPLETED")
print("=" * 75)

print("Output:", OUTPUT.resolve())
print("Original files unchanged.")
print("No graph modification.")
print("No model training.")
