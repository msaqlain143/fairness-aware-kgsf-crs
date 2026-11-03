
import pandas as pd
import html
import re
from pathlib import Path

# Original file
source = Path(
    "results_category_audit/dbpedia_pilot/"
    "genre_manual_validation.csv"
)

# New output file
output = Path(
    "results_category_audit/dbpedia_pilot/"
    "genre_manual_validation_cleaned.csv"
)

if not source.exists():
    raise FileNotFoundError(source)

# Load everything as strings
df = pd.read_csv(source, dtype=str).fillna("")

print("=== CLEANING GENRE VALIDATION FILE ===")
print("Original rows:", len(df))

# 1. Clean column names
clean_columns = []

for col in df.columns:
    col = html.unescape(str(col))
    col = re.sub(r"^\*+", "", col)
    col = col.strip()
    clean_columns.append(col)

df.columns = clean_columns

# 2. Normalize DBpedia URI values
uri_col = "value_uri_or_literal"

if uri_col not in df.columns:
    raise ValueError("URI column not found.")

df[uri_col] = (
    df[uri_col]
    .astype(str)
    .str.replace("\\", "", regex=False)
    .str.strip()
)

# 3. Check required columns
required = [
    "original_category",
    "value_uri_or_literal",
    "preliminary_class",
    "validated_type",
    "canonical_genre",
    "parent_genre",
    "temporal_modifier",
    "geographical_modifier",
    "manual_decision",
    "review_notes"
]

missing = set(required) - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

# 4. Check URI validity
uri = df[uri_col]

invalid = ~uri.str.startswith(
    ("http://", "https://")
)

# 5. Identify duplicate URIs
duplicates = df[
    df.duplicated(uri_col, keep=False)
].sort_values(uri_col)

# 6. Annotation statistics
print("\n=== ANNOTATION STATUS ===")

print(
    df["manual_decision"]
    .value_counts(dropna=False)
)

# 7. Save cleaned copy
df.to_csv(
    output,
    index=False,
    encoding="utf-8-sig"
)

print("\n=== CLEANING SUMMARY ===")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Invalid URI prefixes:", invalid.sum())
print("Duplicate URI records:", len(duplicates))

print("\n=== DUPLICATE URI DETAILS ===")

if len(duplicates):
    print(
        duplicates[
            ["original_category", uri_col]
        ].to_string(index=False)
    )
else:
    print("No duplicate URIs.")

print("\n=== CLEANED HEADERS ===")
print(df.columns.tolist())

print("\nSaved cleaned file:")
print(output)

print("\nOriginal file was NOT modified.")
