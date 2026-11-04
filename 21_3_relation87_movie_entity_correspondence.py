
import json
import csv
import pickle
from pathlib import Path
from collections import Counter

# ============================================================
# EXPERIMENT 21.3
# RELATION 87 AND MOVIE ENTITY CORRESPONDENCE AUDIT
# ============================================================

BASE = Path("data/dataset/redial/nltk")

KG_FILE = BASE / "dbpedia_subkg.json"
ENTITY_FILE = BASE / "entity2id.json"

CATALOG_FILE = Path("catalog_audit_corrected.pkl")

METADATA_FILE = Path(
    "results_category_audit/catalog_metadata_reconciliation.csv"
)

OUT = Path(
    "results_category_audit/relation87_correspondence_v1"
)

OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. LOAD SOURCE FILES
# ============================================================

with KG_FILE.open(encoding="utf-8") as f:
    kg = json.load(f)

with ENTITY_FILE.open(encoding="utf-8") as f:
    entity2id = json.load(f)

with CATALOG_FILE.open("rb") as f:
    catalog = pickle.load(f)

movie_ids = set(map(int, catalog["movie_ids"]))

# ============================================================
# 2. RECONSTRUCT RELATION 87 ENTITY SET
# ============================================================

relation87_heads = set()
relation87_records = 0
relation87_nonself = 0

for head, edges in kg.items():

    head_id = int(head)

    for relation, tail in edges:

        if int(relation) == 87:

            tail_id = int(tail)

            relation87_records += 1
            relation87_heads.add(head_id)

            if head_id != tail_id:
                relation87_nonself += 1

print("=" * 75)
print("EXPERIMENT 21.3: RELATION 87 CORRESPONDENCE")
print("=" * 75)

print("\n[1] RELATION 87")

print("Records:", relation87_records)
print("Unique heads:", len(relation87_heads))
print("Non-self-loop records:", relation87_nonself)

# ============================================================
# 3. ENTITY INDEX VALIDATION
# ============================================================

entity_values = set(map(int, entity2id.values()))

print("\n[2] ENTITY INDEX")

print("Entity mapping records:", len(entity2id))
print("Unique entity indices:", len(entity_values))

print(
    "Relation 87 entities inside entity index:",
    len(relation87_heads & entity_values)
)

print(
    "Relation 87 entities outside entity index:",
    len(relation87_heads - entity_values)
)

# ============================================================
# 4. MOVIE METADATA MAPPING
# ============================================================

metadata_rows = []

with METADATA_FILE.open(
    encoding="utf-8-sig",
    newline=""
) as f:

    metadata_rows = list(csv.DictReader(f))

print("\n[3] MOVIE METADATA")

print("Metadata records:", len(metadata_rows))

print(
    "Metadata columns:",
    list(metadata_rows[0].keys())
    if metadata_rows else []
)

# Examine available entity mappings.
mapping_fields = [
    "mapping_value",
    "external_entity_key",
    "resource_uri",
]

for field in mapping_fields:

    if not metadata_rows or field not in metadata_rows[0]:
        continue

    mapped = set()

    for row in metadata_rows:

        value = str(row.get(field, "")).strip()

        if value.isdigit():
            mapped.add(int(value))

    if mapped:

        overlap = relation87_heads & mapped

        print(f"\nMapping field: {field}")
        print("Unique mapped values:", len(mapped))
        print("Relation 87 overlap:", len(overlap))
        print(
            "Relation 87 overlap ratio:",
            round(
                len(overlap) / len(relation87_heads),
                6
            )
        )

# ============================================================
# 5. DIRECT MOVIE ID COMPARISON
# ============================================================

print("\n[4] DIRECT ID COMPARISON")

direct_overlap = movie_ids & relation87_heads

print("Catalog movie IDs:", len(movie_ids))
print("Relation 87 entities:", len(relation87_heads))
print("Direct numeric overlap:", len(direct_overlap))

print(
    "Catalog IDs in relation 87:",
    len(movie_ids - relation87_heads)
)

print(
    "Relation 87 IDs outside catalog:",
    len(relation87_heads - movie_ids)
)

# ============================================================
# 6. SAVE ENTITY SET AUDIT
# ============================================================

output_file = OUT / "relation87_entity_inventory.csv"

with output_file.open(
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "entity_id",
        "is_in_entity_index",
        "is_direct_catalog_id"
    ])

    for entity in sorted(relation87_heads):

        writer.writerow([
            entity,
            entity in entity_values,
            entity in movie_ids
        ])

# ============================================================
# 7. SUMMARY
# ============================================================

summary = {
    "experiment": "21.3",
    "relation_id": 87,
    "relation_records": relation87_records,
    "unique_entities": len(relation87_heads),
    "nonself_records": relation87_nonself,
    "entity_index_overlap": len(
        relation87_heads & entity_values
    ),
    "entity_index_nonoverlap": len(
        relation87_heads - entity_values
    ),
    "direct_catalog_overlap": len(direct_overlap),
    "direct_catalog_overlap_ratio": (
        len(direct_overlap) / len(movie_ids)
        if movie_ids else 0
    )
}

with (
    OUT / "relation87_correspondence_summary.json"
).open("w", encoding="utf-8") as f:

    json.dump(summary, f, indent=2)

print("\n[5] OUTPUT")

print("Entity inventory:", output_file)
print("Summary:", OUT / "relation87_correspondence_summary.json")

print("\nEXPERIMENT 21.3 COMPLETED")
