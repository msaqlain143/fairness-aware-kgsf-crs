
import json
import csv
import pickle
from pathlib import Path
from collections import Counter

# ============================================================
# EXPERIMENT 21.4
# ENTITY URI AND CATALOG PROVENANCE AUDIT
# ============================================================

BASE = Path("data/dataset/redial/nltk")

ENTITY_FILE = BASE / "entity2id.json"
KG_FILE = BASE / "dbpedia_subkg.json"

CATALOG_FILE = Path("catalog_audit_corrected.pkl")

METADATA_FILE = Path(
    "results_category_audit/catalog_metadata_reconciliation.csv"
)

OUTPUT = Path(
    "results_category_audit/entity_uri_provenance_v1"
)

OUTPUT.mkdir(parents=True, exist_ok=True)


def normalize(value):
    if value is None:
        return ""

    return str(value).strip()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


# ============================================================
# 1. LOAD DATA
# ============================================================

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

# Reverse entity mapping.
id2entity = {
    int(entity_id): normalize(entity)
    for entity, entity_id in entity2id.items()
}

# ============================================================
# 2. RELATION 87 RECONSTRUCTION
# ============================================================

relation87 = Counter()

for head, edges in kg.items():

    head = int(head)

    for relation, tail in edges:

        if int(relation) == 87:
            relation87[(head, int(tail))] += 1

relation87_entities = {
    head
    for head, tail in relation87
}

require(
    relation87_entities == movie_ids,
    "Catalog and relation-87 entity sets differ."
)

require(
    all(head == tail for head, tail in relation87),
    "Non-self-loop relation-87 record detected."
)

# ============================================================
# 3. METADATA INDEX
# ============================================================

metadata_by_id = {}

for row in metadata:

    movie_id = int(row["movie_entity_id"])

    require(
        movie_id not in metadata_by_id,
        f"Duplicate metadata record: {movie_id}"
    )

    metadata_by_id[movie_id] = row

require(
    set(metadata_by_id) == movie_ids,
    "Metadata and catalog IDs do not match."
)

# ============================================================
# 4. URI PROVENANCE COMPARISON
# ============================================================

results = []

statistics = Counter()

for movie_id in sorted(movie_ids):

    row = metadata_by_id[movie_id]

    entity_uri = id2entity.get(movie_id, "")

    movie_uri = normalize(row.get("movie_uri"))
    resource_uri = normalize(row.get("resource_uri"))

    mapping_value = normalize(row.get("mapping_value"))
    external_key = normalize(row.get("external_entity_key"))

    entity_movie_match = (
        entity_uri == movie_uri
        if entity_uri and movie_uri
        else False
    )

    entity_resource_match = (
        entity_uri == resource_uri
        if entity_uri and resource_uri
        else False
    )

    movie_resource_match = (
        movie_uri == resource_uri
        if movie_uri and resource_uri
        else False
    )

    relation87_count = sum(
        count
        for (head, tail), count in relation87.items()
        if head == movie_id
    )

    result = {
        "movie_entity_id": movie_id,
        "entity2id_uri": entity_uri,
        "movie_uri": movie_uri,
        "resource_uri": resource_uri,
        "mapping_value": mapping_value,
        "external_entity_key": external_key,
        "relation87_record_count": relation87_count,
        "entity_movie_uri_match": entity_movie_match,
        "entity_resource_uri_match": entity_resource_match,
        "movie_resource_uri_match": movie_resource_match,
        "has_mapping_value": bool(mapping_value),
        "has_external_entity_key": bool(external_key),
    }

    results.append(result)

    statistics["catalog_entities"] += 1

    statistics["entity_uri_available"] += bool(entity_uri)
    statistics["movie_uri_available"] += bool(movie_uri)
    statistics["resource_uri_available"] += bool(resource_uri)

    statistics["entity_movie_uri_match"] += entity_movie_match
    statistics["entity_resource_uri_match"] += entity_resource_match
    statistics["movie_resource_uri_match"] += movie_resource_match

    statistics["mapping_value_available"] += bool(mapping_value)
    statistics["external_key_available"] += bool(external_key)

    statistics["one_relation87_record"] += (
        relation87_count == 1
    )

# ============================================================
# 5. SAVE DETAILED AUDIT
# ============================================================

output_csv = OUTPUT / "entity_uri_provenance.csv"

with output_csv.open(
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)

# ============================================================
# 6. SAVE SUMMARY
# ============================================================

summary = {
    "experiment": "21.4",
    "catalog_entities": len(movie_ids),
    "relation87_entities": len(relation87_entities),
    "entity_index_size": len(entity2id),
    "statistics": dict(statistics),
    "validation": {
        "catalog_relation87_exact_equality": True,
        "relation87_self_loop_integrity": True,
        "metadata_catalog_integrity": True,
        "entity_index_integrity": (
            movie_ids.issubset(set(id2entity))
        ),
    }
}

with (
    OUTPUT / "entity_uri_provenance_summary.json"
).open("w", encoding="utf-8") as f:

    json.dump(summary, f, indent=2)

# ============================================================
# 7. REPORT
# ============================================================

print("=" * 75)
print("EXPERIMENT 21.4: ENTITY URI PROVENANCE AUDIT")
print("=" * 75)

print("\n[1] ENTITY COVERAGE")

for key, value in statistics.items():
    print(f"{key}: {value}")

print("\n[2] URI CORRESPONDENCE")

for key in [
    "entity_movie_uri_match",
    "entity_resource_uri_match",
    "movie_resource_uri_match",
]:
    print(
        key,
        statistics[key],
        "/",
        len(movie_ids)
    )

print("\n[3] SAMPLE RECORDS")

for row in results[:10]:

    print(
        row["movie_entity_id"],
        "| entity2id:",
        row["entity2id_uri"],
        "| movie_uri:",
        row["movie_uri"],
        "| resource_uri:",
        row["resource_uri"]
    )

print("\n[4] VALIDATION")

for key, value in summary["validation"].items():
    print(f"{key}: {'PASS' if value else 'FAIL'}")

print("\n[5] OUTPUT")

print(output_csv)
print(OUTPUT / "entity_uri_provenance_summary.json")

print("\nEXPERIMENT 21.4 COMPLETED")
