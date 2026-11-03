
import pandas as pd
from pathlib import Path

base = Path("results_category_audit")

df = pd.read_csv(
    base / "catalog_metadata_reconciliation.csv",
    dtype=str
)

non_uri = df[
    df["mapping_type"] != "dbpedia_resource_uri"
].copy()

print("========== NON-URI MAPPING AUDIT ==========")

print("Total non-URI records:", len(non_uri))

print("\nMapping types:")
print(non_uri["mapping_type"].value_counts(dropna=False))

print("\nMetadata status:")
print(non_uri["metadata_status"].value_counts(dropna=False))

print("\nSample records:")
print(
    non_uri[
        [
            "movie_entity_id",
            "movie_uri",
            "mapping_value",
            "mapping_type",
            "resource_uri",
            "metadata_status"
        ]
    ].head(30).to_string(index=False)
)

out = base / "non_uri_mapping_audit.csv"

non_uri.to_csv(out, index=False, encoding="utf-8-sig")

print("\nSaved:", out.resolve())
