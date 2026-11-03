
import pandas as pd
from pathlib import Path

root = Path("results_category_audit")

target_uri = (
    "http://dbpedia.org/resource/"
    "Category:DreamWorks_Animation_animated_films"
)

print("=== DUPLICATE URI PROVENANCE INVESTIGATION ===")
print("Target:", target_uri)

for file in root.rglob("*.csv"):

    try:
        df = pd.read_csv(file, dtype=str).fillna("")
    except Exception:
        continue

    uri_columns = [
        c for c in df.columns
        if "uri" in c.lower()
    ]

    if not uri_columns:
        continue

    for col in uri_columns:

        mask = (
            df[col].astype(str)
            .str.replace("\\", "", regex=False)
            .str.strip()
            .eq(target_uri)
        )

        if mask.any():

            print("\nFILE:", file)
            print("MATCHES:", mask.sum())

            print(
                df.loc[mask].to_string(index=False)
            )

print("\n=== INVESTIGATION COMPLETE ===")
print("No files modified.")