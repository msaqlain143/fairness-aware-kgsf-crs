
from pathlib import Path
import json
import pandas as pd

BASE = Path("results_category_audit")

ENTITY_FILE = Path("data/dataset/redial/nltk/entity2id.json")

TAXONOMY_FILE = (
    BASE / "dbpedia_pilot" /
    "canonical_category_inventory.csv"
)

NEIGHBOR_FILE = (
    BASE / "experiment_21_18A5" /
    "movie_neighbor_entity_inventory.csv"
)

def normalize_uri(value):
    value = str(value).strip()
    if value.startswith("<") and value.endswith(">"):
        return value[1:-1]
    return value

print("=" * 70)
print("EXPERIMENT 21.19A.1: NUMERIC-TO-URI RECONCILIATION")
print("=" * 70)

# --------------------------------------------------
# 1. LOAD ORIGINAL ENTITY MAPPING
# --------------------------------------------------

with open(ENTITY_FILE, "r", encoding="utf-8") as f:
    entity2id = json.load(f)

id2entity = {
    int(entity_id): normalize_uri(uri)
    for uri, entity_id in entity2id.items()
}

print("\n[1] ORIGINAL ENTITY MAPPING")
print("Entity mappings:", len(entity2id))
print("Unique numeric IDs:", len(id2entity))

# --------------------------------------------------
# 2. LOAD TAXONOMY
# --------------------------------------------------

taxonomy = pd.read_csv(TAXONOMY_FILE)

category_uris = set(
    taxonomy["category_uri"].dropna().map(normalize_uri)
)

category_uri_to_id = {
    normalize_uri(uri): int(entity_id)
    for uri, entity_id in entity2id.items()
    if normalize_uri(uri) in category_uris
}

print("\n[2] TAXONOMY-TO-ENTITY MAPPING")
print("Taxonomy categories:", len(category_uris))
print("Categories found in entity2id:", len(category_uri_to_id))
print("Categories absent from entity2id:",
      len(category_uris - set(category_uri_to_id)))

print("\nFirst mapped categories:")
for uri, entity_id in list(category_uri_to_id.items())[:10]:
    print(entity_id, uri)

# --------------------------------------------------
# 3. INSPECT NEIGHBOR INVENTORY
# --------------------------------------------------

neighbors = pd.read_csv(NEIGHBOR_FILE)

print("\n[3] NEIGHBOR INVENTORY SCHEMA")
print("Rows:", len(neighbors))
print("Columns:", neighbors.columns.tolist())

print("\nFirst five records:")
print(neighbors.head().to_string(index=False))

# Identify the numeric neighbor ID column.
possible_columns = [
    "neighbor_entity_id",
    "entity_id",
    "neighbor_id"
]

id_column = next(
    (c for c in possible_columns if c in neighbors.columns),
    None
)

if id_column is None:
    raise RuntimeError(
        "Could not identify neighbor entity ID column. "
        "Inspect the printed schema before proceeding."
    )

neighbor_ids = set(
    pd.to_numeric(
        neighbors[id_column], errors="coerce"
    ).dropna().astype(int)
)

print("\n[4] NUMERIC ID RECONCILIATION")
print("Unique neighbor IDs:", len(neighbor_ids))

mapped_neighbor_uris = {
    id2entity[x]
    for x in neighbor_ids
    if x in id2entity
}

print("Neighbor IDs mapped to URIs:",
      len(mapped_neighbor_uris))

# --------------------------------------------------
# 5. EXACT CATEGORY ENTITY MATCH
# --------------------------------------------------

category_ids = set(category_uri_to_id.values())

category_ids_in_neighbors = category_ids & neighbor_ids

category_uris_in_neighbors = {
    id2entity[x]
    for x in category_ids_in_neighbors
}

print("\n[5] CATEGORY MATCH RESULTS")
print("Category IDs in entity2id:", len(category_ids))
print("Category IDs in KG neighbors:",
      len(category_ids_in_neighbors))
print("Category URIs in KG neighbors:",
      len(category_uris_in_neighbors))

print("\nMatched categories:")
for uri in sorted(category_uris_in_neighbors)[:30]:
    print(uri)

# --------------------------------------------------
# 6. EXPORT RECONCILIATION
# --------------------------------------------------

result = taxonomy.copy()

result["entity_id"] = result["category_uri"].map(
    lambda x: category_uri_to_id.get(normalize_uri(x))
)

result["present_in_neighbor_inventory"] = (
    result["entity_id"].isin(neighbor_ids)
)

result.to_csv(
    BASE / "experiment_21_19A_category_id_reconciliation.csv",
    index=False
)

print("\n[6] FINAL")
print("Categories mapped:", result["entity_id"].notna().sum())
print("Categories present in neighbors:",
      result["present_in_neighbor_inventory"].sum())

print("\nDiagnostic exported successfully.")
