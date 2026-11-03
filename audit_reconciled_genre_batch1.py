
import pandas as pd
from pathlib import Path

path = Path(
    "results_category_audit/dbpedia_pilot/"
    "genre_manual_validation_reconciled_batch1.csv"
)

df = pd.read_csv(path, dtype=str).fillna("")

completed = df[df["manual_decision"] != "PENDING"].copy()

print("=== BATCH 1 ANNOTATION QUALITY AUDIT ===")

print("\nDecision distribution:")
print(completed["manual_decision"].value_counts())

print("\nValidated type distribution:")
print(completed["validated_type"].value_counts())

print("\n=== MISSING ANNOTATION FIELDS ===")

for field in [
    "validated_type",
    "canonical_genre",
    "review_notes"
]:
    missing = (
        completed[field].astype(str).str.strip() == ""
    ).sum()

    print(f"{field}: {missing} missing")

print("\n=== ACCEPTED GENRE CHECK ===")

accepted = completed[
    completed["manual_decision"].isin(
        ["ACCEPT_GENRE", "ACCEPT_HYBRID"]
    )
]

missing_genre = accepted[
    accepted["canonical_genre"].str.strip() == ""
]

print("Accepted genre records:", len(accepted))
print("Missing canonical genre:", len(missing_genre))

if len(missing_genre):
    print(missing_genre[
        ["original_category", "manual_decision"]
    ].to_string(index=False))

print("\n=== ATTRIBUTE-ONLY CHECK ===")

attributes = completed[
    completed["manual_decision"] == "ATTRIBUTE_ONLY"
]

print("Attribute-only records:", len(attributes))

print(attributes[
    ["original_category", "validated_type",
     "canonical_genre", "review_notes"]
].to_string(index=False))

print("\n=== DUPLICATE URI CHECK ===")

duplicates = df[
    df.duplicated("value_uri_or_literal", keep=False)
]

print("Rows involved in duplicate URIs:", len(duplicates))

if len(duplicates):
    print(duplicates[
        ["original_category", "value_uri_or_literal"]
    ].to_string(index=False))

print("\n=== AUDIT COMPLETE ===")
print("No source data modified.")
