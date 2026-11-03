
from pathlib import Path
import pandas as pd
import re

ROOT = Path("results_category_audit/dbpedia_pilot")

SOURCE = ROOT / "canonical_category_inventory.csv"

OUT_ALL = ROOT / "genre_validation_priority_queue.csv"
OUT_BATCH = ROOT / "genre_manual_validation_batch2.csv"

BATCH_SIZE = 100

GENRE_TERMS = [
    "action", "adventure", "comedy", "drama", "crime",
    "fantasy", "horror", "mystery", "romance", "thriller",
    "western", "war", "musical", "documentary", "science fiction",
    "sci-fi", "noir", "biographical", "biopic", "satire",
    "parody", "superhero", "martial arts", "coming-of-age",
    "black comedy", "disaster", "apocalyptic", "fairy tale",
    "psychological", "historical drama"
]

ATTRIBUTE_PATTERNS = {
    "temporal": r"\b(19|20)\d{2}s?\b",
    "geographical": r"\b(American|British|Canadian|Chinese|French|"
                    r"Italian|Japanese|Hong Kong|Australian|"
                    r"German|Indian|English-language)\b",
    "production": r"\b(studios?|productions?|films produced by|"
                  r"films directed by|films scored by)\b",
    "location": r"\b(set in|shot at|filmed in)\b",
    "format": r"\b(animated|animation|computer-animated|"
              r"3D films|IMAX|black-and-white|silent films)\b",
    "distribution": r"\b(distributor|distribution|sequel|"
                    r"franchise|remake|lost films)\b",
    "awards": r"\b(winners|award|nominated|registry)\b",
    "subject": r"\b(about|adapted into|based on|"
               r"containing|involved in)\b",
}

def classify(label):

    label = str(label).lower()

    genre_matches = [
        term for term in GENRE_TERMS
        if term in label
    ]

    attribute_matches = [
        name for name, pattern in ATTRIBUTE_PATTERNS.items()
        if re.search(pattern, label, flags=re.I)
    ]

    if genre_matches and attribute_matches:
        group = "MIXED_REVIEW"

    elif genre_matches:
        group = "GENRE_CANDIDATE"

    elif attribute_matches:
        group = "ATTRIBUTE_CANDIDATE"

    else:
        group = "AMBIGUOUS_REVIEW"

    return pd.Series({
        "recommended_review_group": group,
        "detected_genre_terms": "; ".join(genre_matches),
        "detected_attribute_types": "; ".join(attribute_matches)
    })


print("=== PREPARING GENRE VALIDATION BATCH 2 ===")

df = pd.read_csv(SOURCE, dtype=str).fillna("")

required = {
    "category_uri",
    "preferred_label_provisional",
    "annotation_status"
}

if not required.issubset(df.columns):
    raise ValueError(
        f"Missing columns: {required - set(df.columns)}"
    )

pending = df[
    df["annotation_status"] == "PENDING"
].copy()

if pending["category_uri"].duplicated().any():
    raise ValueError("Duplicate category URIs found.")

classification = pending[
    "preferred_label_provisional"
].apply(classify)

pending = pd.concat(
    [pending.reset_index(drop=True),
     classification.reset_index(drop=True)],
    axis=1
)

priority = {
    "GENRE_CANDIDATE": 0,
    "MIXED_REVIEW": 1,
    "AMBIGUOUS_REVIEW": 2,
    "ATTRIBUTE_CANDIDATE": 3
}

pending["priority"] = pending[
    "recommended_review_group"
].map(priority)

pending = pending.sort_values(
    ["priority", "preferred_label_provisional"]
)

pending.to_csv(OUT_ALL, index=False)

# Batch 2 is a review queue, not automatic annotation.
batch = pending.head(BATCH_SIZE).copy()

batch["validated_type"] = ""
batch["canonical_genre"] = ""
batch["parent_genre"] = ""
batch["temporal_modifier"] = ""
batch["geographical_modifier"] = ""
batch["manual_decision"] = "PENDING"
batch["review_notes"] = ""

batch.to_csv(OUT_BATCH, index=False)

print("\n=== CLASSIFICATION SUMMARY ===")
print(pending["recommended_review_group"].value_counts())

print("\nTotal pending:", len(pending))
print("Batch 2 size:", len(batch))

print("\nBatch 2 distribution:")
print(batch["recommended_review_group"].value_counts())

print("\nSaved:")
print(OUT_ALL)
print(OUT_BATCH)

print("\nOriginal inventory was NOT modified.")
