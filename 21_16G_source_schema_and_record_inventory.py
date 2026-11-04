
from pathlib import Path
import pandas as pd
import json

BASE = Path(".")

SOURCES = {
    "merged_reference":
        BASE / "external_validation/redial_reference/movies_merged.csv",

    "mention_reference":
        BASE / "external_validation/redial_reference/movies_with_mentions.csv",

    "movielens_links":
        BASE / "external_validation/movielens_32m/links.csv",

    "movielens_movies":
        BASE / "external_validation/movielens_32m/movies.csv",

    "unified_identity":
        BASE / "results_category_audit/experiment_21_15/unified_movie_entity_mapping.csv",

    "pairwise_evidence":
        BASE / "results_category_audit/experiment_21_16F/external_identifier_pairwise_evidence.csv",

    "conflict_summary":
        BASE / "results_category_audit/experiment_21_16F/external_identifier_group_summary.csv",
}

OUT = BASE / "results_category_audit/experiment_21_16G"
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 75)
print("EXPERIMENT 21.16G: SOURCE SCHEMA AND RECORD INVENTORY")
print("=" * 75)

inventory = {}

for name, path in SOURCES.items():

    print("\n" + "=" * 75)
    print(name.upper())
    print("=" * 75)

    if not path.exists():
        print("MISSING:", path)
        inventory[name] = {"status": "MISSING"}
        continue

    df = pd.read_csv(path, low_memory=False)

    print("File:", path)
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values:")
    print(df.isna().sum().to_string())

    print("\nFirst five records:")
    print(df.head().to_string(index=False))

    inventory[name] = {
        "status": "LOADED",
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "missing_values": {
            str(k): int(v)
            for k, v in df.isna().sum().items()
        }
    }

# Specific investigation: The Foreigner

print("\n" + "=" * 75)
print("FOREIGNER RECORD RECONSTRUCTION")
print("=" * 75)

merged = pd.read_csv(
    SOURCES["merged_reference"],
    low_memory=False
)

mentions = pd.read_csv(
    SOURCES["mention_reference"],
    low_memory=False
)

database_ids = ["176399", "195484"]

for database_id in database_ids:

    print(f"\nDatabase ID: {database_id}")

    record = merged[
        merged["databaseId"].astype(str) == database_id
    ]

    mention = mentions[
        mentions["movieId"].astype(str) == database_id
    ]

    print("\nMerged reference:")
    print(record.to_string(index=False))

    print("\nMention reference:")
    print(mention.to_string(index=False))

# Save inventory

manifest = {
    "experiment": "21.16G",
    "name": "Source Schema and Record Inventory",
    "sources": inventory,
    "foreigners_investigated": database_ids
}

manifest_path = OUT / "source_schema_inventory.json"

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=4, ensure_ascii=False)

print("\n" + "=" * 75)
print("SOURCE INVENTORY COMPLETED")
print("=" * 75)

print("Output:", manifest_path)
