
import pickle
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE = Path("results_category_audit")
PILOT = BASE / "dbpedia_pilot"

with open("catalog_audit_corrected.pkl", "rb") as f:
    catalog = pickle.load(f)

mapping = pd.read_csv(
    BASE / "movie_entity_mapping.csv"
)

uri_audit = pd.read_csv(
    BASE / "catalog_dbpedia_uri_audit.csv"
)

reconciliation = pd.read_csv(
    BASE / "catalog_metadata_reconciliation.csv"
)

genre = pd.read_csv(
    PILOT / "genre_manual_validation_cleaned.csv"
)

inventory = pd.read_csv(
    PILOT / "canonical_category_inventory.csv"
)

# --------------------------------------------------
# 1. CHECK TAU DEFINITION
# --------------------------------------------------

print("\n========== TAU VALIDATION ==========")

print("Tau:", catalog["tau"])
print("G0 maximum:", catalog["g0_max_frequency"])
print("Frequency definition:",
      catalog["frequency_definition"])
print("Quintile definition:",
      catalog["quintile_definition"])

# --------------------------------------------------
# 2. PREPARE ITEM GROUPS
# --------------------------------------------------

zero_ids = set(catalog["zero_shot_ids"])
one_ids = set(catalog["one_shot_ids"])

mapping["movie_entity_id"] = pd.to_numeric(
    mapping["movie_entity_id"], errors="coerce"
)

mapping["has_entity_mapping"] = pd.to_numeric(
    mapping["has_entity_mapping"], errors="coerce"
).fillna(0)

mapping["direct_kg_degree_records"] = pd.to_numeric(
    mapping["direct_kg_degree_records"], errors="coerce"
).fillna(0)

# --------------------------------------------------
# 3. MAPPING AND KG COVERAGE
# --------------------------------------------------

def coverage_report(name, ids):

    subset = mapping[
        mapping["movie_entity_id"].isin(ids)
    ].copy()

    total = len(ids)

    mapped = (
        subset["has_entity_mapping"] == 1
    ).sum()

    kg_records = (
        subset["direct_kg_degree_records"] > 0
    ).sum()

    no_records = (
        subset["direct_kg_degree_records"] == 0
    ).sum()

    print(f"\n{name}")
    print("-" * 45)
    print("Total items:", total)
    print("Matched mapping rows:", len(subset))
    print("Mapped entities:", mapped)
    print("Mapping coverage:",
          round(mapped / total * 100, 2) if total else 0, "%")
    print("Items with direct KG degree > 0:", kg_records)
    print("Items with zero direct KG degree:", no_records)

    if len(subset):
        print("Mean KG degree:",
              round(subset["direct_kg_degree_records"].mean(), 2))
        print("Median KG degree:",
              subset["direct_kg_degree_records"].median())

    return subset


zero_df = coverage_report("ZERO-SHOT", zero_ids)
one_df = coverage_report("ONE-SHOT", one_ids)

# --------------------------------------------------
# 4. URI AUDIT
# --------------------------------------------------

print("\n========== DBPEDIA URI AUDIT ==========")

print(uri_audit["status"].value_counts(dropna=False))

# --------------------------------------------------
# 5. METADATA RECONCILIATION
# --------------------------------------------------

print("\n========== METADATA RECONCILIATION ==========")

print(reconciliation["metadata_status"]
      .value_counts(dropna=False))

# --------------------------------------------------
# 6. GENRE ANNOTATION STATUS
# --------------------------------------------------

print("\n========== GENRE VALIDATION ==========")

print("Manual validation rows:", len(genre))
print(genre["manual_decision"].value_counts(dropna=False))

print("\nValidated genre types:")
print(genre["validated_type"].value_counts(dropna=False))

print("\n========== CATEGORY INVENTORY ==========")

print("Inventory rows:", len(inventory))
print(inventory["annotation_status"].value_counts(dropna=False))
print(inventory["manual_decision"].value_counts(dropna=False))

# --------------------------------------------------
# 7. DUPLICATES AND MISSING IDS
# --------------------------------------------------

print("\n========== DATA INTEGRITY ==========")

print("Mapping duplicate movie IDs:",
      mapping["movie_entity_id"].duplicated().sum())

print("URI audit duplicate movie IDs:",
      uri_audit["movie_entity_id"].duplicated().sum())

print("Zero-shot overlap with one-shot:",
      len(zero_ids & one_ids))

print("Catalog size:", len(catalog["sorted_items"]))

print("\nDIAGNOSTIC FINISHED")
