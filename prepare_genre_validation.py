
import pandas as pd
from pathlib import Path

BASE = Path("results_category_audit/dbpedia_pilot")

INPUT = BASE / "preliminary_genre_taxonomy.csv"

OUTPUT = BASE / "genre_manual_validation.csv"

df = pd.read_csv(INPUT).fillna("")

# Select potential genres for detailed review
candidates = df[
    df["preliminary_class"].isin([
        "potential_genre",
        "ambiguous_or_other"
    ])
].copy()

# Preserve original metadata
candidates["original_category"] = candidates["label"]

# Add manual review fields
candidates["validated_type"] = ""
candidates["canonical_genre"] = ""
candidates["parent_genre"] = ""
candidates["temporal_modifier"] = ""
candidates["geographical_modifier"] = ""
candidates["manual_decision"] = "PENDING"
candidates["review_notes"] = ""

# Suggested validation types:
# primary_genre
# subgenre
# content_descriptor
# temporal_category
# geographical_category
# production_category
# franchise_category
# unrelated
# unresolved

# Put the most important columns first
columns = [
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

candidates = candidates[columns]

candidates.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("=== GENRE VALIDATION FILE ===")
print("Total categories for review:", len(candidates))
print("Output:", OUTPUT)

print("\nOriginal heuristic classification:")
print(candidates["preliminary_class"].value_counts())

print("\nManual validation fields created.")
print("No original research data modified.")
