
import pandas as pd
from pathlib import Path

BASE = Path("external_validation/redial_reference")

merged = pd.read_csv(BASE / "movies_merged.csv")
mentions = pd.read_csv(BASE / "movies_with_mentions.csv")

print("=" * 75)
print("EXPERIMENT 21.16B: MERGED REFERENCE DIAGNOSTIC")
print("=" * 75)

# 1. Schema
print("\n[1] DATASET STRUCTURE")
print("Merged shape:", merged.shape)
print("Mention shape:", mentions.shape)
print("Merged columns:", merged.columns.tolist())

# 2. Database ID integrity
print("\n[2] DATABASE ID DISTRIBUTION")

print("Merged database IDs:", merged["databaseId"].nunique())
print("Mention movie IDs:", mentions["movieId"].nunique())

merged_counts = merged.groupby("databaseId").size()

print("Database IDs occurring once:", (merged_counts == 1).sum())
print("Database IDs occurring multiple times:", (merged_counts > 1).sum())
print("Maximum records per database ID:", merged_counts.max())

# 3. Reference coverage
print("\n[3] IDENTIFIER COVERAGE")

merged_ids = set(merged["databaseId"].dropna().astype(int))
mention_ids = set(mentions["movieId"].dropna().astype(int))

print("Mention IDs in merged reference:", len(mention_ids & merged_ids))
print("Mention IDs absent from merged reference:", len(mention_ids - merged_ids))
print("Merged IDs outside mention catalog:", len(merged_ids - mention_ids))

# 4. Alternative title analysis
print("\n[4] ALTERNATIVE TITLE STATISTICS")

title_counts = (
    merged.groupby("databaseId")["movieName"]
    .nunique()
)

print("IDs with multiple distinct titles:", (title_counts > 1).sum())
print("Maximum distinct titles per ID:", title_counts.max())

# 5. Inspect examples
print("\n[5] EXAMPLE MULTI-TITLE RECORDS")

multi_ids = title_counts[title_counts > 1].head(20).index

for database_id in multi_ids:

    subset = merged[
        merged["databaseId"] == database_id
    ]

    print("\nDatabase ID:", database_id)

    for title in subset["movieName"].dropna().unique():
        print("  -", title)

# 6. Check repeated title across identifiers
print("\n[6] TITLE COLLISION AUDIT")

title_id_counts = (
    merged.groupby("movieName")["databaseId"]
    .nunique()
)

print(
    "Titles linked to multiple database IDs:",
    (title_id_counts > 1).sum()
)

print("\nTop 20 repeated titles:")

print(
    title_id_counts.sort_values(ascending=False)
    .head(20)
    .to_string()
)

print("\n" + "=" * 75)
print("21.16B DIAGNOSTIC COMPLETED")
print("=" * 75)
