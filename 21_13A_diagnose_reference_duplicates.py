
import json
import pandas as pd
from pathlib import Path

BASE = Path("data/dataset/redial/nltk")
REF = Path("external_validation/redial_reference")
OUT = Path("results_category_audit/experiment_21_13A")

OUT.mkdir(parents=True, exist_ok=True)

# Load numeric IDs from original entity mapping
with (BASE / "entity2id.json").open(encoding="utf-8") as f:
    entity2id = json.load(f)

numeric_ids = {
    int(key)
    for key in entity2id
    if str(key).isdigit()
}

merged = pd.read_csv(REF / "movies_merged.csv")
mentions = pd.read_csv(REF / "movies_with_mentions.csv")

merged["databaseId"] = pd.to_numeric(
    merged["databaseId"], errors="coerce"
)

print("=" * 75)
print("EXPERIMENT 21.13A: DUPLICATE RELATIONSHIP DIAGNOSTIC")
print("=" * 75)

# 1. Duplicate distribution
grouped = merged.groupby("databaseId", dropna=False)

summary = grouped.agg(
    record_count=("databaseId", "size"),
    unique_titles=("movieName", "nunique"),
    unique_movielens_ids=("movielensId", "nunique")
).reset_index()

duplicates = summary[
    summary["record_count"] > 1
].sort_values("record_count", ascending=False)

print("\n[1] DUPLICATE STRUCTURE")
print("Total records:", len(merged))
print("Unique database IDs:", merged["databaseId"].nunique())
print("Duplicate database ID groups:", len(duplicates))
print("Records belonging to duplicate groups:",
      int(duplicates["record_count"].sum()))

print("\n[2] DUPLICATE GROUP DISTRIBUTION")
print(duplicates["record_count"].value_counts().sort_index().to_string())

# 2. Inspect representative duplicate IDs
print("\n[3] REPRESENTATIVE DUPLICATES")

sample_ids = duplicates["databaseId"].head(10).tolist()

for identifier in sample_ids:
    print("\nDatabase ID:", identifier)

    print(
        merged.loc[
            merged["databaseId"] == identifier,
            ["movieName", "databaseId", "movielensId"]
        ].to_string(index=False)
    )

# 3. Investigate the 812 target identifiers
print("\n[4] NUMERIC TARGET IDENTIFIERS")

target = merged[
    merged["databaseId"].isin(numeric_ids)
].copy()

target_summary = target.groupby("databaseId").agg(
    record_count=("databaseId", "size"),
    title_count=("movieName", "nunique"),
    movielens_count=("movielensId", "nunique"),
    titles=("movieName", lambda x: " | ".join(
        sorted(set(x.dropna().astype(str)))
    ))
).reset_index()

target_summary["numeric_id"] = target_summary["databaseId"].isin(
    numeric_ids
)

print("Target IDs:", len(numeric_ids))
print("Matched database IDs:", target_summary["databaseId"].nunique())
print("Target records:", len(target))

target_summary.to_csv(
    OUT / "numeric_target_duplicate_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

# 4. Check exact duplicate rows
exact_duplicates = merged[
    merged.duplicated(keep=False)
]

print("\n[5] EXACT DUPLICATE ROWS")
print("Count:", len(exact_duplicates))

exact_duplicates.to_csv(
    OUT / "exact_duplicate_rows.csv",
    index=False,
    encoding="utf-8-sig"
)

# 5. Compare identifier populations
merged_ids = set(
    merged["databaseId"].dropna().astype(int)
)

mention_ids = set(
    mentions["movieId"].dropna().astype(int)
)

print("\n[6] CROSS-SOURCE ID DIFFERENCES")
print("Merged IDs:", len(merged_ids))
print("Mention IDs:", len(mention_ids))
print("Common IDs:", len(merged_ids & mention_ids))
print("Merged-only IDs:", len(merged_ids - mention_ids))
print("Mention-only IDs:", len(mention_ids - merged_ids))

print("\nMerged-only examples:",
      sorted(merged_ids - mention_ids)[:30])

print("\n[7] TARGET ID COVERAGE")
print("Target IDs in merged:", len(numeric_ids & merged_ids))
print("Target IDs in mentions:", len(numeric_ids & mention_ids))

print("\nOUTPUT DIRECTORY:")
print(OUT.resolve())

print("\nDIAGNOSTIC COMPLETED")
