
import pandas as pd
from pathlib import Path

path = Path(
    "results_category_audit/dbpedia_pilot/"
    "genre_manual_validation.csv"
)

if not path.exists():
    raise FileNotFoundError(f"File not found: {path}")

df = pd.read_csv(path, dtype=str).fillna("")

print("\n=== CSV QUALITY AUDIT ===")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())

print("\n=== MARKDOWN ARTIFACTS ===")

for col in df.columns:
    if df[col].str.contains(
        r"\*\*|&#x[0-9a-fA-F]+;",
        regex=True
    ).any():
        print("Problem in column:", repr(col))

print("\n=== URI FORMAT ===")

uri_col = "value_uri_or_literal"

if uri_col in df.columns:
    uri = df[uri_col]

    print("Backslashes:", uri.str.contains(
        r"\\", regex=True
    ).sum())

    print("Invalid HTTP prefix:", (
        ~uri.str.startswith("http://") &
        ~uri.str.startswith("https://")
    ).sum())

    print("Duplicate URIs:", uri.duplicated().sum())

print("\n=== ANNOTATION STATUS ===")

decision_col = "manual_decision"

if decision_col in df.columns:
    print(df[decision_col].value_counts(
        dropna=False
    ))

print("\n=== DUPLICATE CATEGORY NAMES ===")

print(
    df[df.duplicated("original_category", keep=False)]
    [["original_category", uri_col]]
    .to_string(index=False)
)

print("\n=== AUDIT COMPLETE ===")
print("Original file was NOT modified.")
