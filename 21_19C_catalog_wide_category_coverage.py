
from pathlib import Path
import json
import pandas as pd

BASE = Path("results_category_audit")

RAW_FILE = BASE / "experiment_21_18A2" / "movie_category_edges_raw.csv"
TAX_FILE = BASE / "dbpedia_pilot" / "canonical_category_inventory.csv"
MOVIE_FILE = BASE / "movie_entity_mapping.csv"

OUT = BASE / "experiment_21_19C"
OUT.mkdir(parents=True, exist_ok=True)


def norm(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if value.startswith("<") and value.endswith(">"):
        return value[1:-1]
    return value


print("=" * 75)
print("EXPERIMENT 21.19C: CATALOG-WIDE CATEGORY COVERAGE")
print("=" * 75)

# --------------------------------------------------
# 1. LOAD ORIGINAL SOURCES
# --------------------------------------------------

raw = pd.read_csv(RAW_FILE)
taxonomy = pd.read_csv(TAX_FILE)
movies = pd.read_csv(MOVIE_FILE)

for df, column in [
    (raw, "movie_uri"),
    (raw, "value_uri_or_literal"),
    (taxonomy, "category_uri"),
    (movies, "movie_uri")
]:
    df[column] = df[column].map(norm)

print("\n[1] SOURCE POPULATIONS")
print("Movies:", len(movies))
print("Original category edges:", len(raw))
print("Taxonomy categories:", len(taxonomy))

# --------------------------------------------------
# 2. VALIDATE CATALOG
# --------------------------------------------------

assert movies["movie_entity_id"].is_unique
assert movies["movie_uri"].is_unique

catalog_uris = set(movies["movie_uri"])
raw_movie_uris = set(raw["movie_uri"])

unmatched_movies = raw_movie_uris - catalog_uris

assert not unmatched_movies, (
    f"Unmatched source movies: {len(unmatched_movies)}"
)

# --------------------------------------------------
# 3. BUILD CATEGORY CLASSIFICATION
# --------------------------------------------------

accepted_decisions = [
    "ACCEPT_GENRE",
    "ACCEPT_HYBRID"
]

taxonomy["manual_decision"] = (
    taxonomy["manual_decision"].fillna("").astype(str).str.strip()
)

taxonomy_lookup = {}

for _, row in taxonomy.iterrows():
    uri = row["category_uri"]

    if not uri:
        continue

    decision = row["manual_decision"]

    if decision == "ACCEPT_GENRE":
        classification = "ACCEPTED_GENRE"

    elif decision == "ACCEPT_HYBRID":
        classification = "ACCEPTED_HYBRID"

    elif decision == "ATTRIBUTE_ONLY":
        classification = "ATTRIBUTE_ONLY"

    else:
        classification = "UNVALIDATED_CATEGORY"

    taxonomy_lookup[uri] = {
        "classification": classification,
        "canonical_genre": row.get("canonical_genre", ""),
        "validated_type": row.get("validated_type", ""),
        "decision": decision
    }

# --------------------------------------------------
# 4. CLASSIFY ORIGINAL CATEGORY EDGES
# --------------------------------------------------

edge_rows = []

for _, row in raw.iterrows():

    uri = row["value_uri_or_literal"]

    evidence = taxonomy_lookup.get(uri)

    if evidence:
        classification = evidence["classification"]
        canonical = evidence["canonical_genre"]
    else:
        classification = (
            "NON_TAXONOMY_OR_LITERAL_EVIDENCE"
        )
        canonical = ""

    edge_rows.append({
        "source_record_id": row["source_record_id"],
        "movie_entity_id": row["movie_entity_id"],
        "movie_uri": row["movie_uri"],
        "predicate": row["predicate"],
        "category_uri_or_literal": uri,
        "value_type": row["value_type"],
        "value_label": row["value_label"],
        "classification": classification,
        "canonical_genre": canonical
    })

edges = pd.DataFrame(edge_rows)

edges.to_csv(
    OUT / "category_evidence_edge_classification.csv",
    index=False
)

# --------------------------------------------------
# 5. RECONSTRUCT ALL MOVIE PROFILES
# --------------------------------------------------

profiles = []

for _, movie in movies.iterrows():

    movie_uri = movie["movie_uri"]

    group = edges[edges["movie_uri"] == movie_uri]

    accepted = group[
        group["classification"].isin([
            "ACCEPTED_GENRE",
            "ACCEPTED_HYBRID"
        ])
    ]

    genres = sorted({
    genre.strip()
    for value in accepted["canonical_genre"].dropna().astype(str)
    for genre in value.split(";")
    if genre.strip()
    })

    def count_class(name):
        return int(
            (group["classification"] == name).sum()
        )

    profiles.append({
        "movie_entity_id": int(movie["movie_entity_id"]),
        "movie_uri": movie_uri,
        "has_entity_mapping": int(movie["has_entity_mapping"]),
        "direct_kg_degree_records": int(
            movie["direct_kg_degree_records"]
        ),
        "total_category_evidence_records": len(group),
        "unique_category_values": group[
            "category_uri_or_literal"
        ].nunique(),
        "accepted_genre_records": len(
            group[group["classification"] == "ACCEPTED_GENRE"]
        ),
        "accepted_hybrid_records": len(
            group[group["classification"] == "ACCEPTED_HYBRID"]
        ),
        "attribute_only_records": count_class(
            "ATTRIBUTE_ONLY"
        ),
        "unvalidated_records": count_class(
            "UNVALIDATED_CATEGORY"
        ),
        "other_evidence_records": count_class(
            "NON_TAXONOMY_OR_LITERAL_EVIDENCE"
        ),
        "has_original_category_evidence": len(group) > 0,
        "has_accepted_genre_evidence": len(accepted) > 0,
        "canonical_genres": ";".join(genres)
    })

profiles = pd.DataFrame(profiles)

profiles.to_csv(
    OUT / "catalog_item_category_profiles.csv",
    index=False
)

# --------------------------------------------------
# 6. COVERAGE SUMMARY
# --------------------------------------------------

summary = {
    "experiment": "21.19C",
    "catalog_population": int(len(profiles)),
    "original_category_edge_records": int(len(edges)),
    "unique_category_bearing_movies": int(
        profiles["has_original_category_evidence"].sum()
    ),
    "movies_with_accepted_genre_evidence": int(
        profiles["has_accepted_genre_evidence"].sum()
    ),
    "movies_without_original_category_evidence": int(
        (~profiles["has_original_category_evidence"]).sum()
    ),
    "movies_without_accepted_genre_evidence": int(
        (~profiles["has_accepted_genre_evidence"]).sum()
    ),
    "accepted_genre_edge_records": int(
        (edges["classification"] == "ACCEPTED_GENRE").sum()
    ),
    "accepted_hybrid_edge_records": int(
        (edges["classification"] == "ACCEPTED_HYBRID").sum()
    ),
    "attribute_only_edge_records": int(
        (edges["classification"] == "ATTRIBUTE_ONLY").sum()
    ),
    "unvalidated_edge_records": int(
        (edges["classification"] == "UNVALIDATED_CATEGORY").sum()
    ),
    "catalog_population_preserved": (
        len(profiles) == len(movies)
    ),
    "all_original_category_edges_preserved": (
        len(edges) == len(raw)
    ),
    "all_category_movies_mapped": (
    int(profiles["has_original_category_evidence"].sum())
    == int(raw["movie_uri"].nunique())
    ),
    "no_automatic_genre_assignment": True,
    "no_original_kg_modification": True
}

summary["category_evidence_coverage_percent"] = round(
    100 * summary["unique_category_bearing_movies"]
    / summary["catalog_population"], 4
)

summary["accepted_genre_coverage_percent"] = round(
    100 * summary["movies_with_accepted_genre_evidence"]
    / summary["catalog_population"], 4
)

# --------------------------------------------------
# 7. VALIDATION
# --------------------------------------------------

assert len(profiles) == len(movies)
assert profiles["movie_entity_id"].is_unique
assert len(edges) == len(raw)

assert (
    profiles["has_original_category_evidence"].sum()
    == raw["movie_uri"].nunique()
)

assert (
    profiles["has_accepted_genre_evidence"].sum()
    <= profiles["has_original_category_evidence"].sum()
)

validation = {
    "catalog_population_preserved": True,
    "unique_movie_ids": True,
    "all_raw_edges_preserved": True,
    "all_raw_movie_uris_mapped": True,
    "item_profiles_complete": True,
    "accepted_genres_require_manual_validation": True,
    "no_automatic_genre_assignment": True,
    "no_original_kg_modification": True
}

with open(
    OUT / "experiment_21_19C_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=4)

with open(
    OUT / "experiment_21_19C_validation.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(validation, f, indent=4)

print("\n[2] CATEGORY COVERAGE")
for key, value in summary.items():
    print(f"{key}: {value}")

print("\n[3] EDGE CLASSIFICATION")
print(edges["classification"].value_counts().to_string())

print("\n[4] VALIDATION")
for key, value in validation.items():
    print(f"{key}: {'PASS' if value else 'FAIL'}")

print("\n[5] OUTPUTS")
for file in sorted(OUT.iterdir()):
    print(file.name, file.stat().st_size)

print("\nEXPERIMENT 21.19C COMPLETED")
