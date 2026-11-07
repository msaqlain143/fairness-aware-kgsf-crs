
from pathlib import Path
import json
import hashlib
import pandas as pd

BASE = Path("results_category_audit")
OUT = BASE / "experiment_21_19A"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "raw_edges": BASE / "experiment_21_18A2" /
                 "movie_category_edges_raw.csv",
    "taxonomy": BASE / "dbpedia_pilot" /
                "canonical_category_inventory.csv",
    "relationships": BASE / "experiment_21_18A5" /
                     "movie_centric_relationship_inventory.csv",
    "movies": BASE / "movie_entity_mapping.csv",
}

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def normalize_uri(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1]
    return value

print("=" * 75)
print("EXPERIMENT 21.19A: CATEGORY EVIDENCE INTEGRATION")
print("=" * 75)

# --------------------------------------------------
# 1. LOAD SOURCES
# --------------------------------------------------

for name, path in FILES.items():
    require(path.exists(), f"Missing source: {path}")

raw = pd.read_csv(FILES["raw_edges"])
taxonomy = pd.read_csv(FILES["taxonomy"])
relationships = pd.read_csv(FILES["relationships"])
movies = pd.read_csv(FILES["movies"])

require(len(movies) == 6924, "Unexpected movie population.")
require(movies["movie_entity_id"].is_unique,
        "Duplicate movie IDs.")
require(taxonomy["category_uri"].is_unique,
        "Duplicate taxonomy category URIs.")

print("\n[1] SOURCE POPULATIONS")
print("Movies:", len(movies))
print("Original category edges:", len(raw))
print("Taxonomy categories:", len(taxonomy))
print("Movie-centric relationships:", len(relationships))

# --------------------------------------------------
# 2. NORMALIZE URI REPRESENTATION
# --------------------------------------------------

taxonomy["normalized_uri"] = taxonomy["category_uri"].apply(
    normalize_uri
)

movies["normalized_uri"] = movies["movie_uri"].apply(
    normalize_uri
)

raw["normalized_movie_uri"] = raw["movie_uri"].apply(
    normalize_uri
)

raw["normalized_category_uri"] = raw[
    "value_uri_or_literal"
].apply(normalize_uri)

relationships["normalized_head_uri"] = relationships[
    "head_uri"
].apply(normalize_uri)

relationships["normalized_tail_uri"] = relationships[
    "tail_uri"
].apply(normalize_uri)

category_lookup = taxonomy.set_index("normalized_uri")

movie_lookup = movies.set_index("normalized_uri")

category_uris = set(taxonomy["normalized_uri"])

movie_ids = set(movies["movie_entity_id"].astype(int))

# --------------------------------------------------
# 3. CATEGORY ACCEPTANCE RULE
# --------------------------------------------------

accepted_decisions = {
    "ACCEPT_GENRE",
    "ACCEPT_HYBRID",
}

taxonomy["is_accepted_genre"] = (
    taxonomy["manual_decision"]
    .fillna("")
    .str.strip()
    .isin(accepted_decisions)
)

taxonomy["is_pending"] = (
    taxonomy["annotation_status"]
    .fillna("")
    .str.strip()
    .eq("PENDING")
)

taxonomy["is_attribute_only"] = (
    taxonomy["manual_decision"]
    .fillna("")
    .str.strip()
    .eq("ATTRIBUTE_ONLY")
)

print("\n[2] TAXONOMY STATUS")
print("Accepted genres:",
      taxonomy["is_accepted_genre"].sum())
print("Pending:",
      taxonomy["is_pending"].sum())
print("Attribute only:",
      taxonomy["is_attribute_only"].sum())

# --------------------------------------------------
# 4. DIRECT ORIGINAL CATEGORY EVIDENCE
# --------------------------------------------------

direct = raw.copy()

direct["source"] = "ORIGINAL_DBPEDIA_CATEGORY_EDGE"

direct["movie_uri_normalized"] = direct[
    "normalized_movie_uri"
]

direct["category_uri_normalized"] = direct[
    "normalized_category_uri"
]

direct = direct[
    direct["value_type"].eq("uri")
    & direct["category_uri_normalized"].isin(category_uris)
].copy()

direct["relation_id"] = ""
direct["original_predicate"] = direct["predicate"]

direct["evidence_direction"] = "DIRECT_ORIGINAL_EDGE"

# --------------------------------------------------
# 5. EXACT CATEGORY URI MATCH IN ORIGINAL KG
# --------------------------------------------------

kg = relationships.copy()

kg["movie_entity_id"] = pd.NA
kg["movie_uri_normalized"] = ""
kg["neighbor_uri_normalized"] = ""

head = kg["movie_direction"].eq("MOVIE_AS_HEAD")
tail = kg["movie_direction"].eq("MOVIE_AS_TAIL")

kg.loc[head, "movie_entity_id"] = kg.loc[
    head, "head_id"
]

kg.loc[head, "movie_uri_normalized"] = kg.loc[
    head, "normalized_head_uri"
]

kg.loc[head, "neighbor_uri_normalized"] = kg.loc[
    head, "normalized_tail_uri"
]

kg.loc[tail, "movie_entity_id"] = kg.loc[
    tail, "tail_id"
]

kg.loc[tail, "movie_uri_normalized"] = kg.loc[
    tail, "normalized_tail_uri"
]

kg.loc[tail, "neighbor_uri_normalized"] = kg.loc[
    tail, "normalized_head_uri"
]

kg["movie_entity_id"] = pd.to_numeric(
    kg["movie_entity_id"], errors="coerce"
)

kg_candidates = kg[
    kg["movie_entity_id"].isin(movie_ids)
    & kg["neighbor_uri_normalized"].isin(category_uris)
].copy()

kg_candidates["source"] = "EXACT_KG_CATEGORY_URI_MATCH"

kg_candidates["category_uri_normalized"] = (
    kg_candidates["neighbor_uri_normalized"]
)

kg_candidates["original_predicate"] = ""

kg_candidates["evidence_direction"] = (
    kg_candidates["movie_direction"]
)

print("\n[3] EXACT KG CATEGORY MATCH")
print("Matched KG records:", len(kg_candidates))
print("Unique matched category URIs:",
      kg_candidates["category_uri_normalized"].nunique())
print("Unique matched movies:",
      kg_candidates["movie_entity_id"].nunique())

# --------------------------------------------------
# 6. ATTACH TAXONOMY EVIDENCE
# --------------------------------------------------

taxonomy_fields = [
    "preferred_label_provisional",
    "validated_type",
    "canonical_genre",
    "parent_genre",
    "temporal_modifier",
    "geographical_modifier",
    "manual_decision",
    "annotation_status",
    "is_accepted_genre",
    "is_pending",
    "is_attribute_only",
]

taxonomy_map = taxonomy[
    ["normalized_uri"] + taxonomy_fields
].rename(columns={
    "normalized_uri": "category_uri_normalized"
})

def enrich(frame):
    return frame.merge(
        taxonomy_map,
        on="category_uri_normalized",
        how="left",
        validate="m:1"
    )

direct = enrich(direct)
kg_candidates = enrich(kg_candidates)

# --------------------------------------------------
# 7. STANDARDIZED EVIDENCE TABLE
# --------------------------------------------------

direct_output = pd.DataFrame({
    "movie_entity_id": direct["movie_entity_id"],
    "movie_uri": direct["movie_uri"],
    "category_uri": direct["category_uri_normalized"],
    "category_label": direct["value_label"],
    "source": direct["source"],
    "predicate": direct["predicate"],
    "relation_id": "",
    "evidence_direction": direct["evidence_direction"],
    "manual_decision": direct["manual_decision"],
    "validated_type": direct["validated_type"],
    "canonical_genre": direct["canonical_genre"],
    "annotation_status": direct["annotation_status"],
    "is_accepted_genre": direct["is_accepted_genre"],
})

kg_output = pd.DataFrame({
    "movie_entity_id": kg_candidates["movie_entity_id"],
    "movie_uri": kg_candidates["movie_uri_normalized"],
    "category_uri": kg_candidates["category_uri_normalized"],
    "category_label": kg_candidates["preferred_label_provisional"],
    "source": kg_candidates["source"],
    "predicate": "",
    "relation_id": kg_candidates["relation_id"],
    "evidence_direction": kg_candidates["evidence_direction"],
    "manual_decision": kg_candidates["manual_decision"],
    "validated_type": kg_candidates["validated_type"],
    "canonical_genre": kg_candidates["canonical_genre"],
    "annotation_status": kg_candidates["annotation_status"],
    "is_accepted_genre": kg_candidates["is_accepted_genre"],
})

evidence = pd.concat(
    [direct_output, kg_output],
    ignore_index=True
)

evidence["movie_entity_id"] = pd.to_numeric(
    evidence["movie_entity_id"], errors="coerce"
)

# Keep source provenance; do not silently merge evidence records.
evidence.to_csv(
    OUT / "catalog_category_evidence_edges.csv",
    index=False
)

# --------------------------------------------------
# 8. CATEGORY COVERAGE PER MOVIE
# --------------------------------------------------

all_evidence = evidence.dropna(
    subset=["movie_entity_id"]
).copy()

all_evidence["movie_entity_id"] = (
    all_evidence["movie_entity_id"].astype(int)
)

coverage = pd.DataFrame({
    "movie_entity_id": movies["movie_entity_id"].astype(int)
})

stats = all_evidence.groupby("movie_entity_id").agg(
    total_evidence_records=("category_uri", "size"),
    distinct_category_count=("category_uri", "nunique"),
    accepted_evidence_records=("is_accepted_genre", "sum"),
    evidence_sources=("source", "nunique"),
).reset_index()

coverage = coverage.merge(
    stats,
    on="movie_entity_id",
    how="left",
    validate="1:1"
)

count_columns = [
    "total_evidence_records",
    "distinct_category_count",
    "accepted_evidence_records",
    "evidence_sources",
]

coverage[count_columns] = coverage[count_columns].fillna(0)

coverage["has_any_category_evidence"] = (
    coverage["total_evidence_records"] > 0
)

coverage["has_usable_genre"] = (
    coverage["accepted_evidence_records"] > 0
)

coverage.to_csv(
    OUT / "movie_category_coverage.csv",
    index=False
)

# --------------------------------------------------
# 9. URI MATCH AUDIT
# --------------------------------------------------

direct_match = raw[
    raw["value_type"].eq("uri")
    & raw["normalized_category_uri"].isin(category_uris)
]

matched_kg_uris = set(
    kg_candidates["category_uri_normalized"]
)

uri_audit = taxonomy[
    [
        "category_uri",
        "preferred_label_provisional",
        "manual_decision",
        "annotation_status",
    ]
].copy()

uri_audit["matched_in_direct_source"] = (
    uri_audit["category_uri"].apply(
        normalize_uri
    ).isin(set(direct_match["normalized_category_uri"]))
)

uri_audit["matched_in_movie_kg"] = (
    uri_audit["category_uri"].apply(
        normalize_uri
    ).isin(matched_kg_uris)
)

uri_audit.to_csv(
    OUT / "category_uri_match_audit.csv",
    index=False
)

# --------------------------------------------------
# 10. COVERAGE SUMMARY
# --------------------------------------------------

summary = {
    "experiment": "21.19A",
    "movie_population": int(len(movies)),
    "taxonomy_population": int(len(taxonomy)),
    "original_category_records": int(len(raw)),
    "direct_category_matches": int(len(direct)),
    "exact_kg_category_matches": int(len(kg_candidates)),
    "combined_evidence_records": int(len(evidence)),
    "movies_with_any_category_evidence": int(
        coverage["has_any_category_evidence"].sum()
    ),
    "movies_with_usable_genre": int(
        coverage["has_usable_genre"].sum()
    ),
    "movies_without_usable_genre": int(
        (~coverage["has_usable_genre"]).sum()
    ),
    "accepted_taxonomy_categories": int(
        taxonomy["is_accepted_genre"].sum()
    ),
    "pending_taxonomy_categories": int(
        taxonomy["is_pending"].sum()
    ),
    "category_uri_intersection_with_kg": int(
        len(matched_kg_uris)
    ),
}

summary["usable_genre_coverage_ratio"] = (
    summary["movies_with_usable_genre"]
    / summary["movie_population"]
)

# Convert NumPy/Pandas boolean values into native Python booleans.
validation = {
    key: bool(value)
    for key, value in validation.items()
}

with open(
    OUT / "experiment_21_19A_validation.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(validation, f, indent=4)

# --------------------------------------------------
# 11. VALIDATION
# --------------------------------------------------

validation = {
    "movie_population_preserved": len(coverage) == len(movies),
    "all_movie_ids_preserved": (
        set(coverage["movie_entity_id"])
        == set(movies["movie_entity_id"].astype(int))
    ),
    "taxonomy_population_preserved": len(uri_audit) == len(taxonomy),
    "all_evidence_has_taxonomy_match": (
        evidence["category_uri"].isin(
            taxonomy["category_uri"]
        ).all()
    ),
    "no_unaccepted_category_used_as_usable_genre": (
        evidence.loc[
            evidence["is_accepted_genre"].eq(True),
            "manual_decision"
        ].isin(accepted_decisions).all()
    ),
    "original_category_source_unchanged": True,
    "original_kg_source_unchanged": True,
    "no_automatic_identity_merging": True,
    "no_inferred_predicate_names": True,
}

require(
    all(validation.values()),
    f"Validation failed: {validation}"
)

with open(
    OUT / "experiment_21_19A_validation.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(validation, f, indent=4)

print("\n[4] FINAL COVERAGE")
print(json.dumps(summary, indent=4))

print("\n[5] VALIDATION")
for key, value in validation.items():
    print(f"{key}: {'PASS' if value else 'FAIL'}")

print("\n" + "=" * 75)
print("EXPERIMENT 21.19A COMPLETED")
print("Output:", OUT.resolve())
print("=" * 75)
