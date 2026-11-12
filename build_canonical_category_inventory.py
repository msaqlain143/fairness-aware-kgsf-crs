
from pathlib import Path
import pandas as pd

ROOT = Path("results_category_audit/dbpedia_pilot")

SOURCE = ROOT / "unique_subject_categories.csv"
ANNOTATIONS = ROOT / "genre_manual_validation_reconciled_batch1.csv"

OUT_INVENTORY = ROOT / "canonical_category_inventory.csv"
OUT_ALIASES = ROOT / "category_label_aliases.csv"

def normalize_uri(series):
    return (
        series.astype(str)
        .str.replace("\\", "", regex=False)
        .str.strip()
    )

def combine(values):
    values = sorted({
        str(v).strip()
        for v in values
        if pd.notna(v) and str(v).strip()
    })
    return " || ".join(values)

print("=== CANONICAL CATEGORY INVENTORY ===")

categories = pd.read_csv(SOURCE, dtype=str).fillna("")

required = {"value_uri_or_literal", "value_label"}

if not required.issubset(categories.columns):
    raise ValueError(f"Missing columns: {required - set(categories.columns)}")

categories["category_uri"] = normalize_uri(
    categories["value_uri_or_literal"]
)

categories = categories[
    categories["category_uri"].str.startswith("http", na=False)
].copy()

categories["category_label"] = (
    categories["value_label"].astype(str).str.strip()
)

categories = categories[
    categories["category_label"] != ""
]

# Preserve every original label and its URI.
aliases = categories[
    ["category_uri", "category_label"]
].drop_duplicates()

# Restore annotations, if the reconciled file exists.
if ANNOTATIONS.exists():

    ann = pd.read_csv(ANNOTATIONS, dtype=str).fillna("")

    ann["category_uri"] = normalize_uri(
        ann["value_uri_or_literal"]
    )

    annotation_fields = [
        "preliminary_class",
        "validated_type",
        "canonical_genre",
        "parent_genre",
        "temporal_modifier",
        "geographical_modifier",
        "manual_decision",
        "review_notes",
    ]

    for col in annotation_fields:
        if col not in ann.columns:
            ann[col] = ""

    annotation_groups = (
        ann.groupby("category_uri")
        .agg({
            col: combine for col in annotation_fields
        })
        .reset_index()
    )

else:
    annotation_groups = pd.DataFrame(
        columns=["category_uri"] + annotation_fields
    )

# One record per unique semantic URI.
records = []

for uri, group in categories.groupby("category_uri"):

    labels = sorted(group["category_label"].unique())

    # Longest label is only a provisional display label.
    preferred = sorted(
        labels,
        key=lambda x: (-len(x), x)
    )[0]

    records.append({
        "category_uri": uri,
        "preferred_label_provisional": preferred,
        "alternative_labels": combine(labels),
        "label_count": len(labels),
        "source_row_count": len(group),
    })

inventory = pd.DataFrame(records)

inventory = inventory.merge(
    annotation_groups,
    on="category_uri",
    how="left"
)

inventory = inventory.fillna("")

# Detect incomplete and conflicting annotations.
def annotation_status(row):

    decision = row["manual_decision"].split(" || ")
    decision = [x for x in decision if x]

    if not decision or all(x == "PENDING" for x in decision):
        return "PENDING"

    if len(set(decision)) > 1:
        return "CONFLICT"

    return "ANNOTATED"

inventory["annotation_status"] = inventory.apply(
    annotation_status,
    axis=1
)

inventory = inventory.sort_values(
    "preferred_label_provisional"
)

aliases.to_csv(OUT_ALIASES, index=False)
inventory.to_csv(OUT_INVENTORY, index=False)

print("\n=== RESULTS ===")
print("Original category rows:", len(categories))
print("Unique category URIs:", len(inventory))
print("Duplicate label records:", len(aliases) - len(inventory))

print("\nAnnotation status:")
print(inventory["annotation_status"].value_counts())

print("\nRepeated labels:")
print(inventory[inventory["label_count"] > 1][
    ["preferred_label_provisional",
     "alternative_labels",
     "category_uri"]
].to_string(index=False))

print("\nSaved:")
print(OUT_INVENTORY)
print(OUT_ALIASES)

print("\nOriginal files were not modified.")
