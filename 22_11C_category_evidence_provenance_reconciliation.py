
"""
Experiment 22.11C
Category Evidence Provenance Reconciliation

Purpose:
- Reconcile 21.19C edge-level DBpedia classifications.
- Correct the 21.22A DBpedia evidence-register contamination.
- Preserve IMDb evidence independently.
- Retain hybrid evidence without promoting it to strict genre evidence.
- Preserve all 6,924 catalog items.
- Never modify original experiment outputs.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import hashlib
import platform

import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

INPUT_19C = (
    ROOT / "results_category_audit"
    / "experiment_21_19C"
)

INPUT_22A = (
    ROOT / "results_category_audit"
    / "experiment_21_22A"
)

OUT = (
    ROOT / "results_corrected"
    / "experiment_22_11C"
)

OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_CATALOG_SIZE = 6924

STRICT = "ACCEPTED_GENRE"
HYBRID = "ACCEPTED_HYBRID"
UNVALIDATED = "UNVALIDATED_CATEGORY"


# ============================================================
# 2. UTILITIES
# ============================================================

def require_columns(df, columns, name):
    missing = set(columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"{name}: missing columns {sorted(missing)}"
        )


def truth(value):
    if pd.isna(value):
        return False

    return str(value).strip().lower() in {
        "true", "1", "yes"
    }


def split_categories(value):
    if pd.isna(value):
        return []

    return sorted({
        x.strip()
        for x in str(value).replace("|", ";").split(";")
        if x.strip()
    })


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def json_safe(value):
    """
    Recursively convert NumPy/Pandas scalar values to native JSON types.

    In particular, numpy.bool_ does not serialize through the standard
    json encoder. Calling item() converts NumPy scalar values to their
    corresponding Python scalar types.
    """
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    # Convert NumPy scalar types (bool_, int64, float64, etc.).
    if hasattr(value, "item") and callable(value.item):
        try:
            return json_safe(value.item())
        except (ValueError, TypeError):
            pass

    # JSON has no representation for NaN or infinity.
    if isinstance(value, float):
        import math
        if not math.isfinite(value):
            return None

    # Handle pandas missing scalar values without touching array-like data.
    if value is pd.NA:
        return None

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            json_safe(obj),
            f,
            indent=2,
            ensure_ascii=False,
            allow_nan=False
        )


# ============================================================
# 3. LOAD ORIGINAL INPUTS
# ============================================================

edges = pd.read_csv(
    INPUT_19C / "category_evidence_edge_classification.csv"
)

p19 = pd.read_csv(
    INPUT_19C / "catalog_item_category_profiles.csv"
)

p22 = pd.read_csv(
    INPUT_22A / "catalog_item_category_profiles.csv"
)

e22 = pd.read_csv(
    INPUT_22A / "item_category_evidence.csv"
)

require_columns(
    edges,
    {
        "source_record_id",
        "movie_entity_id",
        "movie_uri",
        "predicate",
        "category_uri_or_literal",
        "classification",
        "canonical_genre"
    },
    "21.19C edges"
)

require_columns(
    p22,
    {
        "movie_entity_id",
        "movie_uri",
        "accepted_genre_evidence",
        "accepted_hybrid_evidence",
        "imdb_genre_evidence",
        "identity_resolution_status"
    },
    "21.22A profiles"
)

require_columns(
    e22,
    {
        "movie_entity_id",
        "category",
        "source",
        "evidence_type",
        "identity_status"
    },
    "21.22A evidence"
)


# ============================================================
# 4. VALIDATE CATALOG
# ============================================================

for name, df in [
    ("21.19C profiles", p19),
    ("21.22A profiles", p22)
]:
    if len(df) != EXPECTED_CATALOG_SIZE:
        raise ValueError(
            f"{name}: unexpected catalog size {len(df)}"
        )

    if df.movie_entity_id.duplicated().any():
        raise ValueError(
            f"{name}: duplicate movie IDs"
        )

ids19 = set(p19.movie_entity_id.astype(int))
ids22 = set(p22.movie_entity_id.astype(int))

if ids19 != ids22:
    raise ValueError(
        "21.19C and 21.22A catalog IDs differ"
    )

catalog_ids = ids22


# ============================================================
# 5. RECONSTRUCT DBPEDIA EVIDENCE FROM ORIGINAL EDGES
# ============================================================

allowed = {
    STRICT,
    HYBRID,
    UNVALIDATED,
    "ATTRIBUTE_ONLY",
    "NON_TAXONOMY_OR_LITERAL_EVIDENCE"
}

unknown = set(edges.classification.dropna()) - allowed

if unknown:
    raise ValueError(
        f"Unexpected edge classifications: {unknown}"
    )

dbpedia = edges.copy()

dbpedia["movie_entity_id"] = (
    dbpedia["movie_entity_id"].astype(int)
)

dbpedia["source"] = "DBpedia"

dbpedia["evidence_type"] = dbpedia["classification"]

dbpedia["category"] = dbpedia["canonical_genre"].fillna("")

dbpedia["verified_genre"] = (
    dbpedia["classification"] == STRICT
)

dbpedia["verified_hybrid"] = (
    dbpedia["classification"] == HYBRID
)

dbpedia["evidence_origin"] = "21.19C_original_edge"

dbpedia["identity_status"] = (
    dbpedia["movie_entity_id"]
    .map(
        p22.set_index("movie_entity_id")
        ["identity_resolution_status"]
    )
)

dbpedia["verified_for_category_fairness"] = (
    dbpedia["verified_genre"]
    & dbpedia["identity_status"].ne(
        "AMBIGUOUS_REQUIRES_REVIEW"
    )
)


# ============================================================
# 6. RECONSTRUCT EXTERNAL IMDb EVIDENCE
# ============================================================

imdb = e22[
    e22["source"].astype(str).str.strip().str.lower()
    == "imdb"
].copy()

imdb["movie_entity_id"] = (
    imdb["movie_entity_id"].astype(int)
)

imdb["evidence_origin"] = "21.22A_external_evidence"

imdb["verified_genre"] = False
imdb["verified_hybrid"] = False

imdb["verified_for_category_fairness"] = (
    imdb["identity_status"].ne(
        "AMBIGUOUS_REQUIRES_REVIEW"
    )
)

# IMDb remains external evidence.
# It is never relabeled as direct DBpedia evidence.


# ============================================================
# 7. CREATE RECONCILED EVIDENCE REGISTER
# ============================================================

common_columns = [
    "movie_entity_id",
    "category",
    "source",
    "evidence_type",
    "identity_status",
    "evidence_origin",
    "verified_genre",
    "verified_hybrid",
    "verified_for_category_fairness"
]

dbpedia_out = dbpedia[common_columns].copy()
imdb_out = imdb[common_columns].copy()

reconciled = pd.concat(
    [dbpedia_out, imdb_out],
    ignore_index=True
)

reconciled = reconciled.merge(
    p22[["movie_entity_id", "movie_uri"]],
    on="movie_entity_id",
    how="left",
    validate="many_to_one"
)

if reconciled.movie_uri.isna().any():
    raise ValueError(
        "Evidence contains movie IDs outside the catalog"
    )

reconciled.to_csv(
    OUT / "reconciled_item_category_evidence.csv",
    index=False
)


# ============================================================
# 8. RECONCILE CATALOG PROFILES
# ============================================================

strict_ids = set(
    dbpedia.loc[
        dbpedia.classification == STRICT,
        "movie_entity_id"
    ]
)

hybrid_ids = set(
    dbpedia.loc[
        dbpedia.classification == HYBRID,
        "movie_entity_id"
    ]
)

unvalidated_ids = set(
    dbpedia.loc[
        dbpedia.classification == UNVALIDATED,
        "movie_entity_id"
    ]
)

imdb_ids = set(imdb.movie_entity_id)

ambiguous_ids = set(
    p22.loc[
        p22.identity_resolution_status
        == "AMBIGUOUS_REQUIRES_REVIEW",
        "movie_entity_id"
    ]
)


def categories_for(movie_id, evidence_type):
    subset = dbpedia[
        (dbpedia.movie_entity_id == movie_id)
        & (dbpedia.evidence_type == evidence_type)
    ]

    result = set()

    for value in subset.category:
        result.update(split_categories(value))

    return sorted(result)


def imdb_categories(movie_id):
    subset = imdb[
        imdb.movie_entity_id == movie_id
    ]

    return sorted({
        str(x).strip()
        for x in subset.category.dropna()
        if str(x).strip()
    })


profiles = p22.copy()

profiles["strict_dbpedia_genres"] = (
    profiles.movie_entity_id.map(
        lambda x: ";".join(categories_for(x, STRICT))
    )
)

profiles["hybrid_dbpedia_categories"] = (
    profiles.movie_entity_id.map(
        lambda x: ";".join(categories_for(x, HYBRID))
    )
)

profiles["unvalidated_dbpedia_categories"] = (
    profiles.movie_entity_id.map(
        lambda x: ";".join(categories_for(x, UNVALIDATED))
    )
)

profiles["external_imdb_genres"] = (
    profiles.movie_entity_id.map(
        lambda x: ";".join(imdb_categories(x))
    )
)

profiles["reconciled_strict_genre_evidence"] = (
    profiles.movie_entity_id.isin(strict_ids)
)

profiles["reconciled_hybrid_evidence"] = (
    profiles.movie_entity_id.isin(hybrid_ids)
)

profiles["reconciled_imdb_evidence"] = (
    profiles.movie_entity_id.isin(imdb_ids)
)

profiles["reconciled_unvalidated_evidence"] = (
    profiles.movie_entity_id.isin(unvalidated_ids)
)

profiles["identity_eligible"] = (
    ~profiles.movie_entity_id.isin(ambiguous_ids)
)

profiles["strict_category_fairness_eligible"] = (
    profiles.reconciled_strict_genre_evidence
    & profiles.identity_eligible
)

profiles["hybrid_only"] = (
    profiles.reconciled_hybrid_evidence
    & ~profiles.reconciled_strict_genre_evidence
)

profiles["reconciled_category_availability"] = "UNKNOWN"

profiles.loc[
    profiles.reconciled_unvalidated_evidence,
    "reconciled_category_availability"
] = "UNVALIDATED_EVIDENCE_ONLY"

profiles.loc[
    profiles.reconciled_hybrid_evidence,
    "reconciled_category_availability"
] = "HYBRID_EVIDENCE_ONLY"

profiles.loc[
    profiles.reconciled_imdb_evidence,
    "reconciled_category_availability"
] = "IMDB_GENRE_EVIDENCE"

profiles.loc[
    profiles.reconciled_strict_genre_evidence,
    "reconciled_category_availability"
] = "DBPEDIA_GENRE_EVIDENCE"

profiles.loc[
    profiles.movie_entity_id.isin(
        strict_ids & imdb_ids
    ),
    "reconciled_category_availability"
] = "MULTI_SOURCE_GENRE_EVIDENCE"

profiles.loc[
    ~profiles.identity_eligible,
    "reconciled_category_availability"
] = "IDENTITY_REVIEW_REQUIRED"

profiles.to_csv(
    OUT / "reconciled_catalog_category_profiles.csv",
    index=False
)


# ============================================================
# 9. PROVENANCE DISCREPANCIES
# ============================================================

old_dbpedia = e22[
    e22.source == "DBpedia"
].copy()

old_ids = set(
    old_dbpedia.movie_entity_id.astype(int)
)

discrepancies = []

for movie_id in sorted(catalog_ids):

    strict = movie_id in strict_ids
    hybrid = movie_id in hybrid_ids
    old = movie_id in old_ids

    if old and not strict:
        discrepancies.append({
            "movie_entity_id": movie_id,
            "old_register_dbpedia": True,
            "strict_genre_edge": False,
            "hybrid_edge": hybrid,
            "discrepancy": (
                "OLD_DBpedia_REGISTER_INCLUDES_NON_STRICT_EVIDENCE"
            ),
            "corrected_classification": (
                HYBRID if hybrid else "REQUIRES_REVIEW"
            )
        })

    elif strict and not old:
        discrepancies.append({
            "movie_entity_id": movie_id,
            "old_register_dbpedia": False,
            "strict_genre_edge": True,
            "hybrid_edge": hybrid,
            "discrepancy": (
                "STRICT_EDGE_MISSING_FROM_OLD_REGISTER"
            ),
            "corrected_classification": STRICT
        })

discrepancy_df = pd.DataFrame(discrepancies)

discrepancy_df.to_csv(
    OUT / "evidence_provenance_discrepancies.csv",
    index=False
)


# ============================================================
# 10. COVERAGE RECONCILIATION
# ============================================================

coverage = pd.DataFrame([
    {
        "evidence_population": "Strict DBpedia genre",
        "unique_movies": len(strict_ids),
        "records": int((dbpedia.evidence_type == STRICT).sum())
    },
    {
        "evidence_population": "Accepted hybrid DBpedia",
        "unique_movies": len(hybrid_ids),
        "records": int((dbpedia.evidence_type == HYBRID).sum())
    },
    {
        "evidence_population": "Unvalidated DBpedia",
        "unique_movies": len(unvalidated_ids),
        "records": int(
            (dbpedia.evidence_type == UNVALIDATED).sum()
        )
    },
    {
        "evidence_population": "External IMDb",
        "unique_movies": len(imdb_ids),
        "records": len(imdb)
    },
    {
        "evidence_population": "Strict DBpedia eligible for fairness",
        "unique_movies": len(strict_ids - ambiguous_ids),
        "records": int(
            (
                dbpedia.verified_for_category_fairness
            ).sum()
        )
    }
])

coverage.to_csv(
    OUT / "category_coverage_reconciliation.csv",
    index=False
)


# ============================================================
# 11. DISPUTED ITEM REPORT
# ============================================================

disputed_ids = set(
    discrepancy_df.movie_entity_id.astype(int)
)

disputed = profiles[
    profiles.movie_entity_id.isin(disputed_ids)
].copy()

disputed.to_csv(
    OUT / "disputed_item_reconciliation.csv",
    index=False
)


# ============================================================
# 12. VALIDATION
# ============================================================

strict_profile_ids = set(
    profiles.loc[
        profiles.reconciled_strict_genre_evidence,
        "movie_entity_id"
    ]
)

strict_evidence_ids = set(
    reconciled.loc[
        (reconciled.source == "DBpedia")
        & reconciled.verified_genre,
        "movie_entity_id"
    ]
)

old_non_strict = (
    old_ids - strict_ids
)

checks = {
    "experiment": "22.11C",
    "catalog_size_valid": len(profiles) == EXPECTED_CATALOG_SIZE,
    "catalog_ids_unique": profiles.movie_entity_id.is_unique,
    "catalog_ids_preserved": (
        set(profiles.movie_entity_id) == catalog_ids
    ),
    "strict_profile_matches_edge_evidence": (
        strict_profile_ids == strict_ids
    ),
    "strict_register_matches_edge_evidence": (
        strict_evidence_ids == strict_ids
    ),
    "hybrid_evidence_preserved": (
        hybrid_ids.issubset(
            set(
                reconciled.loc[
                    reconciled.verified_hybrid,
                    "movie_entity_id"
                ]
            )
        )
    ),
    "hybrid_not_promoted_to_strict": (
        not any(
            (
                movie_id in hybrid_ids
                and movie_id not in strict_ids
                and movie_id in strict_evidence_ids
            )
            for movie_id in catalog_ids
        )
    ),
    "imdb_kept_separate": (
        (reconciled.loc[
            reconciled.source == "IMDb",
            "verified_genre"
        ] == False).all()
    ),
    "ambiguous_identity_excluded": (
        not profiles.loc[
            ~profiles.identity_eligible,
            "strict_category_fairness_eligible"
        ].any()
    ),
    "old_register_discrepancies_identified": (
        old_non_strict.issubset(disputed_ids)
    ),
    "no_training": True,
    "no_retrieval": True,
    "original_sources_untouched": True
}

checks["overall_pass"] = all(
    bool(value)
    for key, value in checks.items()
    if key != "experiment"
)

save_json(
    OUT / "validation_checks.json",
    checks
)


# ============================================================
# 13. MANIFEST
# ============================================================

input_files = [
    INPUT_19C / "category_evidence_edge_classification.csv",
    INPUT_19C / "catalog_item_category_profiles.csv",
    INPUT_22A / "catalog_item_category_profiles.csv",
    INPUT_22A / "item_category_evidence.csv"
]

output_files = [
    OUT / "reconciled_item_category_evidence.csv",
    OUT / "reconciled_catalog_category_profiles.csv",
    OUT / "evidence_provenance_discrepancies.csv",
    OUT / "category_coverage_reconciliation.csv",
    OUT / "disputed_item_reconciliation.csv"
]

manifest = {
    "experiment": "22.11C",
    "name": "Category Evidence Provenance Reconciliation",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "python": platform.python_version(),
    "pandas": pd.__version__,
    "catalog_size": len(profiles),
    "strict_dbpedia_movies": len(strict_ids),
    "hybrid_dbpedia_movies": len(hybrid_ids),
    "imdb_movies": len(imdb_ids),
    "ambiguous_identity_movies": len(ambiguous_ids),
    "old_register_non_strict_movies": sorted(old_non_strict),
    "methodology": {
        "dbpedia_provenance": "21.19C edge-level classifications",
        "external_provenance": "21.22A IMDb evidence register",
        "hybrid_promoted_to_strict": False,
        "identity_ambiguous_items_eligible": False,
        "training": False,
        "retrieval": False
    },
    "input_sha256": {
        str(p.relative_to(ROOT)): sha256(p)
        for p in input_files
    },
    "outputs": [
        p.name for p in output_files
    ],
    "validation": checks
}

save_json(
    OUT / "experiment_manifest.json",
    manifest
)


# ============================================================
# 14. FINAL REPORT
# ============================================================

print("\n" + "=" * 75)
print("EXPERIMENT 22.11C COMPLETE")
print("=" * 75)

print("Catalog items:", len(profiles))
print("Strict DBpedia movies:", len(strict_ids))
print("Hybrid DBpedia movies:", len(hybrid_ids))
print("IMDb-supported movies:", len(imdb_ids))
print("Ambiguous identities:", len(ambiguous_ids))

print("\nOld DBpedia register non-strict movies:")
print(sorted(old_non_strict))

print("\nCoverage reconciliation:")
print(coverage.to_string(index=False))

print("\nValidation:")
print(
    json.dumps(
        json_safe(checks),
        indent=2,
        ensure_ascii=False,
        allow_nan=False
    )
)

print("\nOutput:", OUT.resolve())

if not checks["overall_pass"]:
    raise RuntimeError(
        "22.11C validation failed. Inspect validation_checks.json"
    )
