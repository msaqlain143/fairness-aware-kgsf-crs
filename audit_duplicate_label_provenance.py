
import pandas as pd
from pathlib import Path

ROOT = Path("results_category_audit/dbpedia_pilot")

TARGET_URI = (
    "http://dbpedia.org/resource/"
    "Category:DreamWorks_Animation_animated_films"
)

FILES = [
    ROOT / "category_predicate_audit.csv",
    ROOT / "unique_subject_categories.csv",
    ROOT / "preliminary_genre_taxonomy.csv",
]

def normalize_uri(series):
    return (
        series.astype(str)
        .str.replace("\\", "", regex=False)
        .str.strip()
    )

print("=== CORRECTED DUPLICATE URI AUDIT ===")

for file in FILES:

    print("\nFILE:", file.name)

    if not file.exists():
        print("File not found.")
        continue

    df = pd.read_csv(file, dtype=str).fillna("")

    print("Columns:", df.columns.tolist())

    if "value_uri_or_literal" not in df.columns:
        print("Required category URI column not found.")
        continue

    normalized = normalize_uri(
        df["value_uri_or_literal"]
    )

    matched = df[normalized == TARGET_URI]

    print("Matching records:", len(matched))

    if len(matched) > 0:

        columns = [
            c for c in [
                "movie_uri",
                "predicate",
                "value_uri_or_literal",
                "value_label",
                "value_type",
                "value_entity_type",
                "movie_entity_id",
            ]
            if c in matched.columns
        ]

        print(
            matched[columns].to_string(index=False)
        )

        if "value_label" in matched.columns:
            print(
                "Unique labels:",
                matched["value_label"].unique().tolist()
            )

print("\n=== AUDIT COMPLETE ===")
print("No source files modified.")
