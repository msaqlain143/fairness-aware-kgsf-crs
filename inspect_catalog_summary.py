import pickle
import pandas as pd
from pathlib import Path

BASE = Path("results_category_audit")

print("\n========== 1. CATALOG SUMMARY ==========")

with open("catalog_audit_corrected.pkl", "rb") as f:
    catalog = pickle.load(f)

for key in [
    "sorted_items",
    "zero_shot_ids",
    "one_shot_ids",
    "low_interaction_ids",
]:
    values = catalog[key]
    print(f"{key}: {len(values)}")

print("\n========== 2. FREQUENCY GROUPS ==========")

print("Tau:", catalog["tau"])
print("G0 maximum frequency:", catalog["g0_max_frequency"])
print("Training recommender turns:",
      catalog["train_recommender_turns"])
print("Training target occurrences:",
      catalog["train_target_occurrences"])

for group, metadata in catalog["group_meta"].items():
    print(f"\nGroup {group}:")
    print(metadata)

print("\n========== 3. MAPPING FILE SUMMARIES ==========")

files = [
    "movie_entity_mapping.csv",
    "catalog_dbpedia_uri_audit.csv",
    "catalog_metadata_reconciliation.csv",
    "genre_manual_validation_cleaned.csv",
    "canonical_category_inventory.csv",
]

for filename in files:
    matches = list(BASE.rglob(filename))

    if not matches:
        print(f"\nNOT FOUND: {filename}")
        continue

    df = pd.read_csv(matches[0], dtype=str)

    print(f"\nFILE: {filename}")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))
    print("\nSample:")
    print(df.head(3).to_string(index=False))

print("\n========== DIAGNOSTIC COMPLETE ==========")
