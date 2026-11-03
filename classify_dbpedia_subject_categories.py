
import pandas as pd
import re
from pathlib import Path

BASE = Path("results_category_audit/dbpedia_pilot")

INPUT = BASE / "unique_subject_categories.csv"
OUTPUT = BASE / "preliminary_genre_taxonomy.csv"

df = pd.read_csv(INPUT).fillna("")

# Normalize labels
df["label"] = df["value_label"].astype(str).str.strip()

df.loc[df["label"] == "", "label"] = (
    df["value_uri_or_literal"]
    .astype(str)
    .str.split("/")
    .str[-1]
    .str.replace("Category:", "", regex=False)
    .str.replace("_", " ", regex=False)
)

def classify(label):

    text = label.lower()

    # Clearly non-genre categories first
    exclusions = {
        "production_or_industry": [
            "pictures", "cinema", "film studio",
            "entertainment", "distributor",
            "production company", "films by",
            "award-winning", "award winning",
            "award", "national film registry"
        ],

        "temporal_geographical": [
            " films of ", " films set in ",
            "films shot in", "films from",
            "film by country", "american films",
            "british films", "french films",
            "2010 films", "2000 films",
            "1990 films", "1980 films",
            "silent films"
        ],

        "franchise_or_source_material": [
            "franchise", "novel", "book",
            "based on", "sequels", "remakes",
            "film series", "film adaptations"
        ]
    }

    for group, terms in exclusions.items():
        if any(term in text for term in terms):
            return group

    genre_terms = [
        "comedy", "drama", "horror", "thriller",
        "romance", "western", "science fiction",
        "fantasy", "action", "adventure",
        "crime", "mystery", "musical",
        "war film", "documentary", "animation",
        "animated", "martial arts", "noir",
        "coming-of-age", "disaster",
        "psychological", "historical",
        "superhero", "satire", "biographical",
        "sports film", "erotic", "parody",
        "road movie", "melodrama"
    ]

    if any(term in text for term in genre_terms):
        return "potential_genre"

    return "ambiguous_or_other"


df["preliminary_class"] = df["label"].apply(classify)

df["manual_validation"] = "required"

# Save complete taxonomy
df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("=== TAXONOMY SUMMARY ===")

print(
    df["preliminary_class"]
    .value_counts()
    .to_string()
)

print("\n=== POTENTIAL GENRES ===")

print(
    df[df["preliminary_class"] == "potential_genre"]
    [["label", "value_uri_or_literal"]]
    .to_string(index=False)
)

print("\n=== AMBIGUOUS CATEGORIES ===")

print(
    df[df["preliminary_class"] == "ambiguous_or_other"]
    ["label"]
    .to_string(index=False)
)

print("\nTotal unique categories:", len(df))
print("Saved:", OUTPUT)

print(
    "\nNOTE: This is a heuristic screening, "
    "not a verified genre taxonomy."
)
