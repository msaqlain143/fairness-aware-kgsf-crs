"""
EXPERIMENT 21.16E
MovieLens-Database ID Cross-Reference Validation

Purpose
-------
1. Audit MovieLens IDs associated with multiple database IDs.
2. Compare movie titles and explicitly stated release years.
3. Reconcile movies_merged.csv with movies_with_mentions.csv.
4. Audit database-ID multiplicity.
5. Reconcile current conflicts with the previous Experiment 21.16C audit
   when its schema can be recognized.
6. Preserve unresolved identity conflicts and export reproducible evidence.

Safety / research policy
------------------------
- Original source files are read only.
- The Experiment 21.15 identity mapping is read only.
- No entity is automatically merged.
- Title similarity and shared MovieLens IDs are candidate evidence only.
- No identity is labelled independently verified by this experiment.
"""

from __future__ import annotations

import json
import re
import unicodedata
from itertools import combinations
from pathlib import Path
from typing import Optional

import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MERGED_PATH = (
    ROOT / "external_validation" / "redial_reference" / "movies_merged.csv"
)

MENTION_PATH = (
    ROOT / "external_validation" / "redial_reference"
    / "movies_with_mentions.csv"
)

IDENTITY_PATH = (
    ROOT / "results_category_audit" / "experiment_21_15"
    / "unified_movie_entity_mapping.csv"
)

PREVIOUS_AUDIT_PATH = (
    ROOT / "results_category_audit" / "experiment_21_16C"
    / "movielens_id_integrity.csv"
)

OUTPUT = (
    ROOT / "results_category_audit" / "experiment_21_16E"
)

EXPECTED_IDENTITY_POPULATION = 6924
MISSING_MOVIELENS_SENTINELS = {"", "-1", "-1.0", "nan", "none", "null"}

OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. INPUT VALIDATION AND LOADING
# ============================================================

required_files = [MERGED_PATH, MENTION_PATH, IDENTITY_PATH]

for path in required_files:
    if not path.is_file():
        raise FileNotFoundError(f"Required source file not found: {path}")

merged = pd.read_csv(
    MERGED_PATH,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
)

mentions = pd.read_csv(
    MENTION_PATH,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
)

identity = pd.read_csv(
    IDENTITY_PATH,
    dtype=str,
    keep_default_na=False,
    encoding="utf-8-sig",
)

# Retain the loaded source population as a baseline for integrity checks.
SOURCE_MERGED_ROW_COUNT = len(merged)

previous_audit = None
previous_audit_load_status = "NOT_AVAILABLE"

if PREVIOUS_AUDIT_PATH.is_file():
    try:
        previous_audit = pd.read_csv(
            PREVIOUS_AUDIT_PATH,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        previous_audit_load_status = "LOADED"
    except Exception as exc:
        previous_audit_load_status = (
            f"LOAD_FAILED: {type(exc).__name__}: {exc}"
        )


# ============================================================
# 3. SCHEMA VALIDATION
# ============================================================

required_merged = {
    "index",
    "movieName",
    "databaseId",
    "movielensId",
}

required_mentions = {
    "movieId",
    "movieName",
    "nbMentions",
}

required_identity = {
    "movie_entity_id",
    "identity_type",
    "canonical_identifier",
    "identity_status",
}


def require_columns(
    dataframe: pd.DataFrame,
    required: set[str],
    source_name: str,
) -> None:
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {missing}. "
            f"Available columns: {list(dataframe.columns)}"
        )


require_columns(merged, required_merged, "movies_merged.csv")
require_columns(mentions, required_mentions, "movies_with_mentions.csv")
require_columns(identity, required_identity, "Experiment 21.15 identity mapping")


# ============================================================
# 4. NORMALIZATION HELPERS
# ============================================================

def clean_text(value: object) -> str:
    """Normalize Unicode representation and trim surrounding whitespace."""
    if value is None:
        return ""
    return unicodedata.normalize("NFKC", str(value)).strip()


def normalize_title(value: object) -> str:
    """
    Produce a conservative normalized title key.

    This key supports comparison only; it is not an identity proof.
    """
    value = clean_text(value).casefold()
    return re.sub(r"[^\w]", "", value, flags=re.UNICODE)


def extract_year(value: object) -> Optional[int]:
    """Extract a four-digit year explicitly present in the source title."""
    match = re.search(r"\b(18\d{2}|19\d{2}|20\d{2})\b", clean_text(value))
    return int(match.group(1)) if match else None


def normalize_title_core(value: object) -> str:
    """Remove a trailing parenthesized year, then normalize the title."""
    value = clean_text(value)
    value = re.sub(r"\s*\((?:18|19|20)\d{2}\)\s*$", "", value)
    return normalize_title(value)


def is_missing_movielens_id(value: object) -> bool:
    return clean_text(value).casefold() in MISSING_MOVIELENS_SENTINELS


def as_python_bool(value: object) -> bool:
    """Convert numpy/Pandas booleans and scalar truth values to Python bool."""
    return bool(value)


# ============================================================
# 5. PREPARE SOURCE DATA
# ============================================================

# Preserve source values in their original columns; use trimmed string
# values in the working columns used for matching.
merged["databaseId"] = merged["databaseId"].map(clean_text)
merged["movielensId"] = merged["movielensId"].map(clean_text)
merged["movieName"] = merged["movieName"].map(clean_text)

mentions["movieId"] = mentions["movieId"].map(clean_text)
mentions["movieName"] = mentions["movieName"].map(clean_text)
mentions["nbMentions"] = mentions["nbMentions"].map(clean_text)

identity["movie_entity_id"] = identity["movie_entity_id"].map(clean_text)
identity["canonical_identifier"] = (
    identity["canonical_identifier"].map(clean_text)
)

merged["normalized_title"] = merged["movieName"].map(normalize_title)
merged["title_core"] = merged["movieName"].map(normalize_title_core)
merged["release_year"] = merged["movieName"].map(extract_year)

# A stable row number refers to the row order in the supplied CSV.
merged["source_row"] = range(len(merged))

# Empty database identifiers are not silently grouped together.
valid_database = merged[merged["databaseId"] != ""].copy()

# Exclude only the known missing/unmatched MovieLens sentinels.
valid_ml = valid_database[
    ~valid_database["movielensId"].map(is_missing_movielens_id)
].copy()


# ============================================================
# 6. MENTION DATASET RECONCILIATION
# ============================================================

if mentions["movieId"].eq("").any():
    raise RuntimeError(
        "movies_with_mentions.csv contains blank movieId values."
    )

if mentions["movieId"].duplicated().any():
    duplicate_ids = (
        mentions.loc[mentions["movieId"].duplicated(keep=False), "movieId"]
        .drop_duplicates()
        .tolist()
    )
    raise RuntimeError(
        "Mention reference contains duplicate movieId values. "
        f"Examples: {duplicate_ids[:20]}"
    )

mention_lookup = mentions.set_index("movieId")

merged["mention_reference_found"] = merged["databaseId"].isin(
    mention_lookup.index
)

merged["mention_title"] = (
    merged["databaseId"].map(mention_lookup["movieName"]).fillna("")
)

merged["mention_title_normalized"] = (
    merged["mention_title"].map(normalize_title)
)

# Empty title strings must not be counted as successful title matches.
merged["mention_title_match"] = (
    merged["normalized_title"].ne("")
    & merged["mention_title_normalized"].ne("")
    & (
        merged["normalized_title"]
        == merged["mention_title_normalized"]
    )
)

merged["mention_count"] = (
    merged["databaseId"].map(mention_lookup["nbMentions"]).fillna("")
)

# Rebuild working datasets after mention reconciliation
valid_database = merged[
    merged["databaseId"] != ""
].copy()

valid_ml = valid_database[
    ~valid_database["movielensId"].map(
        is_missing_movielens_id
    )
].copy()


# ============================================================
# 7. MOVIELENS MULTIPLICITY INVENTORY
# ============================================================

group_stats = (
    valid_ml.groupby("movielensId", dropna=False)
    .agg(
        record_count=("databaseId", "size"),
        unique_database_ids=("databaseId", "nunique"),
        unique_titles=("movieName", "nunique"),
        database_ids=(
            "databaseId",
            lambda values: " || ".join(sorted(set(values))),
        ),
        titles=(
            "movieName",
            lambda values: " || ".join(sorted(set(values))),
        ),
    )
    .reset_index()
)

conflicts = group_stats[
    group_stats["unique_database_ids"] > 1
].copy()

print("=" * 75)
print("EXPERIMENT 21.16E: MOVIELENS-DATABASE CROSS-REFERENCE VALIDATION")
print("=" * 75)

print("\n[1] SOURCE INVENTORY")
print("Merged records:", len(merged))
print("Rows with nonblank database IDs:", len(valid_database))
print("Mention records:", len(mentions))
print("Unified identities:", len(identity))
print("Valid MovieLens records:", len(valid_ml))
print("Unique valid MovieLens IDs:", valid_ml["movielensId"].nunique())
print("Conflicting MovieLens IDs:", len(conflicts))
print("Previous 21.16C audit:", previous_audit_load_status)


# ============================================================
# 8. PAIRWISE CONFLICT CLASSIFICATION
# ============================================================

def classify_pair(a: dict, b: dict) -> str:
    """
    Assign a descriptive candidate class.

    These classes are not identity decisions and must not be used to
    merge records automatically.
    """
    if a["databaseId"] == b["databaseId"]:
        return "SAME_DATABASE_ID_MULTIPLE_SOURCE_RECORDS"

    title_a = a["normalized_title"]
    title_b = b["normalized_title"]
    core_a = a["title_core"]
    core_b = b["title_core"]

    year_a = a["release_year"]
    year_b = b["release_year"]

    same_title = bool(title_a and title_b and title_a == title_b)
    same_core = bool(core_a and core_b and core_a == core_b)

    year_a_known = pd.notna(year_a)
    year_b_known = pd.notna(year_b)
    both_years_known = year_a_known and year_b_known

    if same_title:
        if both_years_known and int(year_a) == int(year_b):
            return "POTENTIAL_DUPLICATE_SAME_TITLE_AND_YEAR"
        return "POTENTIAL_DUPLICATE_NEEDS_YEAR_REVIEW"

    if same_core:
        if both_years_known and int(year_a) != int(year_b):
            return "POTENTIAL_REMAKE_OR_DIFFERENT_VERSION"
        return "POTENTIAL_SAME_TITLE_CORE_NEEDS_REVIEW"

    if both_years_known and int(year_a) != int(year_b):
        return "UNRESOLVED_DIFFERENT_TITLES_AND_YEARS"

    return "UNRESOLVED"


pair_records: list[dict] = []
expected_pairwise_count = 0

for movielens_id, group in valid_ml.groupby("movielensId", sort=True):
    # Remove exact (database ID, title) repetitions for pairwise review,
    # while retaining different titles attached to the same database ID.
    unique_records = group.drop_duplicates(
        subset=["databaseId", "movieName"]
    )

    if unique_records["databaseId"].nunique() < 2:
        continue

    rows = unique_records.to_dict("records")
    expected_pairwise_count += len(rows) * (len(rows) - 1) // 2

    for a, b in combinations(rows, 2):
        category = classify_pair(a, b)

        pair_records.append(
            {
                "movielens_id": movielens_id,
                "database_id_a": a["databaseId"],
                "database_id_b": b["databaseId"],
                "title_a": a["movieName"],
                "title_b": b["movieName"],
                "year_a": a["release_year"],
                "year_b": b["release_year"],
                "normalized_title_a": a["normalized_title"],
                "normalized_title_b": b["normalized_title"],
                "same_normalized_title": (
                    bool(a["normalized_title"])
                    and bool(b["normalized_title"])
                    and a["normalized_title"] == b["normalized_title"]
                ),
                "same_title_core": (
                    bool(a["title_core"])
                    and bool(b["title_core"])
                    and a["title_core"] == b["title_core"]
                ),
                "mention_a_found": as_python_bool(
                    a["mention_reference_found"]
                ),
                "mention_b_found": as_python_bool(
                    b["mention_reference_found"]
                ),
                "mention_a_title": a["mention_title"],
                "mention_b_title": b["mention_title"],
                "mention_a_title_match": as_python_bool(
                    a["mention_title_match"]
                ),
                "mention_b_title_match": as_python_bool(
                    b["mention_title_match"]
                ),
                "candidate_classification": category,
                "identity_verification": "NOT_VERIFIED",
                "automatic_merge": False,
            }
        )

pair_columns = [
    "movielens_id",
    "database_id_a",
    "database_id_b",
    "title_a",
    "title_b",
    "year_a",
    "year_b",
    "normalized_title_a",
    "normalized_title_b",
    "same_normalized_title",
    "same_title_core",
    "mention_a_found",
    "mention_b_found",
    "mention_a_title",
    "mention_b_title",
    "mention_a_title_match",
    "mention_b_title_match",
    "candidate_classification",
    "identity_verification",
    "automatic_merge",
]

pair_df = pd.DataFrame(pair_records, columns=pair_columns)


# ============================================================
# 9. DATABASE-ID MULTIPLICITY AUDIT
# ============================================================

database_audit = (
    valid_database.groupby("databaseId", sort=True)
    .agg(
        record_count=("source_row", "size"),
        unique_titles=("movieName", "nunique"),
        unique_movielens_ids=("movielensId", "nunique"),
        titles=(
            "movieName",
            lambda values: " || ".join(sorted(set(values))),
        ),
        movielens_ids=(
            "movielensId",
            lambda values: " || ".join(sorted(set(values))),
        ),
    )
    .reset_index()
)

database_conflicts = database_audit[
    database_audit["record_count"] > 1
].copy()

# Also report the more specific cases in which a database ID has multiple
# distinct titles or multiple non-missing MovieLens identifiers.
database_identity_conflicts = database_audit[
    (database_audit["unique_titles"] > 1)
    | (database_audit["unique_movielens_ids"] > 1)
].copy()


# ============================================================
# 10. MENTION CROSS-REFERENCE OUTPUT
# ============================================================

mention_audit = merged[
    [
        "source_row",
        "databaseId",
        "movieName",
        "movielensId",
        "mention_reference_found",
        "mention_title",
        "mention_title_match",
        "mention_count",
    ]
].copy()


# ============================================================
# 11. CONFLICT-GROUP CLASSIFICATION
# ============================================================

group_classifications: list[dict] = []

for movielens_id, group in valid_ml.groupby("movielensId", sort=True):
    database_id_count = int(group["databaseId"].nunique())

    if database_id_count <= 1:
        continue

    group_pairs = (
        pair_df[pair_df["movielens_id"] == movielens_id]
        if not pair_df.empty
        else pair_df
    )

    categories = (
        group_pairs["candidate_classification"].tolist()
        if not group_pairs.empty
        else []
    )

    if categories and all(
        category == "POTENTIAL_DUPLICATE_SAME_TITLE_AND_YEAR"
        for category in categories
    ):
        status = "POTENTIAL_DUPLICATE_REQUIRES_INDEPENDENT_VERIFICATION"
    elif any(
        category == "POTENTIAL_REMAKE_OR_DIFFERENT_VERSION"
        for category in categories
    ):
        status = "POTENTIAL_DISTINCT_FILMS_REQUIRES_VERIFICATION"
    elif any(
        category == "SAME_DATABASE_ID_MULTIPLE_SOURCE_RECORDS"
        for category in categories
    ):
        status = "MIXED_DATABASE_MULTIPLICITY_REQUIRES_REVIEW"
    else:
        status = "UNRESOLVED_REQUIRES_REVIEW"

    group_classifications.append(
        {
            "movielens_id": movielens_id,
            "database_id_count": database_id_count,
            "record_count": int(len(group)),
            "unique_title_count": int(group["movieName"].nunique()),
            "database_ids": " || ".join(
                sorted(group["databaseId"].unique())
            ),
            "titles": " || ".join(
                sorted(group["movieName"].unique())
            ),
            "pairwise_comparison_count": int(len(group_pairs)),
            "pairwise_candidate_classes": " || ".join(
                sorted(set(categories))
            ),
            "candidate_status": status,
            "identity_verified": False,
            "automatic_merge": False,
        }
    )

group_columns = [
    "movielens_id",
    "database_id_count",
    "record_count",
    "unique_title_count",
    "database_ids",
    "titles",
    "pairwise_comparison_count",
    "pairwise_candidate_classes",
    "candidate_status",
    "identity_verified",
    "automatic_merge",
]

group_df = pd.DataFrame(
    group_classifications,
    columns=group_columns,
)


# ============================================================
# 12. PREVIOUS EXPERIMENT 21.16C RECONCILIATION
# ============================================================

def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> Optional[str]:
    """Return the first matching column name from a list of aliases."""
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate
    return None


previous_reconciliation_status = "SKIPPED_NO_PREVIOUS_AUDIT"
previous_conflict_ids: set[str] = set()
current_conflict_ids = set(conflicts["movielensId"].astype(str))
previous_conflict_ids_column = None
previous_count_column = None

if previous_audit is not None:
    previous_conflict_ids_column = find_column(
        previous_audit,
        [
            "movielensId",
            "movielens_id",
            "movielensID",
            "movieLensId",
            "movie_lens_id",
        ],
    )

    previous_count_column = find_column(
        previous_audit,
        [
            "unique_database_ids",
            "database_id_count",
            "database_ids_count",
            "database_id_multiplicity",
            "database_id_nunique",
        ],
    )

    if previous_conflict_ids_column and previous_count_column:
        previous_counts = pd.to_numeric(
            previous_audit[previous_count_column],
            errors="coerce",
        )

        previous_conflict_ids = set(
            previous_audit.loc[
                previous_counts > 1,
                previous_conflict_ids_column,
            ]
            .map(clean_text)
            .tolist()
        )

        previous_reconciliation_status = "COMPARABLE"
    else:
        previous_reconciliation_status = (
            "SKIPPED_UNRECOGNIZED_PREVIOUS_AUDIT_SCHEMA"
        )

previous_only_ids = sorted(previous_conflict_ids - current_conflict_ids)
current_only_ids = sorted(current_conflict_ids - previous_conflict_ids)
shared_conflict_ids = sorted(previous_conflict_ids & current_conflict_ids)

previous_reconciliation = pd.DataFrame(
    [
        {
            "movielens_id": movielens_id,
            "previous_experiment_21_16C_conflict": (
                movielens_id in previous_conflict_ids
            ),
            "current_experiment_21_16E_conflict": (
                movielens_id in current_conflict_ids
            ),
            "reconciliation_status": (
                "PRESENT_IN_BOTH"
                if movielens_id in previous_conflict_ids
                and movielens_id in current_conflict_ids
                else "CURRENT_ONLY"
                if movielens_id in current_conflict_ids
                else "PREVIOUS_ONLY"
            ),
        }
        for movielens_id in sorted(
            previous_conflict_ids | current_conflict_ids
        )
    ],
    columns=[
        "movielens_id",
        "previous_experiment_21_16C_conflict",
        "current_experiment_21_16E_conflict",
        "reconciliation_status",
    ],
)


# ============================================================
# 13. VALIDATION
# ============================================================

identity_ids = identity["movie_entity_id"].map(clean_text)
identity_canonical_ids = identity["canonical_identifier"].map(clean_text)

validation = {
    "merged_source_population_preserved": (
        len(merged) == SOURCE_MERGED_ROW_COUNT
    ),
    "merged_database_ids_nonblank_for_audit": (
        len(valid_database) == int(merged["databaseId"].ne("").sum())
    ),
    "mention_ids_unique": (
        not mentions["movieId"].duplicated().any()
    ),
    "identity_population_6924": (
        len(identity) == EXPECTED_IDENTITY_POPULATION
    ),
    "identity_internal_ids_unique": (
        identity_ids.is_unique
    ),
    "identity_canonical_identifiers_nonblank": (
        identity_canonical_ids.ne("").all()
    ),
    "identity_canonical_identifiers_unique": (
        identity_canonical_ids.is_unique
    ),
    "conflict_groups_preserved": (
        len(group_df) == len(conflicts)
    ),
    "pairwise_records_complete": (
        len(pair_df) == expected_pairwise_count
    ),
    "previous_conflict_groups_reconciled": (
        previous_reconciliation_status == "COMPARABLE"
        and previous_conflict_ids == current_conflict_ids
    ) if previous_reconciliation_status == "COMPARABLE" else None,
    "no_automatic_merges": (
        pair_df.empty or not pair_df["automatic_merge"].any()
    ),
    "no_verified_identity_claims": (
        pair_df.empty
        or pair_df["identity_verification"].eq("NOT_VERIFIED").all()
    ),
}

# Do not fail the experiment solely because the optional previous audit
# has a different schema or is absent. Report that comparison as skipped.
required_validation = {
    key: value
    for key, value in validation.items()
    if value is not None
}

print("\n[2] CONFLICT ANALYSIS")
print("Conflict groups:", len(group_df))
print("Pairwise comparisons:", len(pair_df))
print("Database IDs with repeated source records:", len(database_conflicts))
print("Database IDs with title/MovieLens multiplicity:", len(database_identity_conflicts))

print("\n[3] PAIRWISE CANDIDATE CLASSIFICATION")
if not pair_df.empty:
    print(
        pair_df["candidate_classification"]
        .value_counts()
        .to_string()
    )
else:
    print("No pairwise conflict records generated.")

print("\n[4] MENTION RECONCILIATION")
print(
    "Merged rows with mention reference:",
    int(merged["mention_reference_found"].sum()),
)
print(
    "Merged rows with exact normalized title agreement:",
    int(merged["mention_title_match"].sum()),
)
print(
    "Merged rows without mention reference:",
    int((~merged["mention_reference_found"]).sum()),
)

print("\n[5] PREVIOUS EXPERIMENT RECONCILIATION")
print("Status:", previous_reconciliation_status)
if previous_reconciliation_status == "COMPARABLE":
    print("Previous conflict groups:", len(previous_conflict_ids))
    print("Current conflict groups:", len(current_conflict_ids))
    print("Shared conflict groups:", len(shared_conflict_ids))
    print("Previous-only groups:", len(previous_only_ids))
    print("Current-only groups:", len(current_only_ids))
else:
    print(
        "Previous audit comparison was not used as a pass/fail condition. "
        "Check the manifest for the reason."
    )

print("\n[6] VALIDATION")
for name, passed in validation.items():
    if passed is None:
        print(f"{name}: SKIPPED")
    else:
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

failed_checks = [
    name
    for name, passed in required_validation.items()
    if not passed
]

if failed_checks:
    raise RuntimeError(
        "Experiment validation failed: "
        + ", ".join(failed_checks)
        + ". Inspect source schemas and generated diagnostics."
    )


# ============================================================
# 14. EXPORT RESULTS
# ============================================================

group_df.to_csv(
    OUTPUT / "movielens_conflict_group_classification.csv",
    index=False,
    encoding="utf-8-sig",
)

pair_df.to_csv(
    OUTPUT / "movielens_conflict_pairwise_analysis.csv",
    index=False,
    encoding="utf-8-sig",
)

database_conflicts.to_csv(
    OUTPUT / "database_id_repeated_record_audit.csv",
    index=False,
    encoding="utf-8-sig",
)

database_identity_conflicts.to_csv(
    OUTPUT / "database_id_identity_multiplicity_audit.csv",
    index=False,
    encoding="utf-8-sig",
)

mention_audit.to_csv(
    OUTPUT / "merged_mention_cross_reference.csv",
    index=False,
    encoding="utf-8-sig",
)

group_stats.to_csv(
    OUTPUT / "complete_movielens_group_inventory.csv",
    index=False,
    encoding="utf-8-sig",
)

previous_reconciliation.to_csv(
    OUTPUT / "previous_21_16C_conflict_reconciliation.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 15. MANIFEST
# ============================================================

candidate_counts = (
    {
        str(key): int(value)
        for key, value in pair_df[
            "candidate_classification"
        ].value_counts().items()
    }
    if not pair_df.empty
    else {}
)

manifest = {
    "experiment": "21.16E",
    "name": "MovieLens-Database ID Cross-Reference Validation",
    "purpose": (
        "Conservatively investigate MovieLens identifier multiplicity, "
        "database-ID multiplicity, and cross-reference consistency."
    ),
    "source_files": {
        "merged_reference": str(MERGED_PATH.relative_to(ROOT)),
        "mention_reference": str(MENTION_PATH.relative_to(ROOT)),
        "unified_identity_mapping": str(IDENTITY_PATH.relative_to(ROOT)),
        "previous_21_16C_audit": str(PREVIOUS_AUDIT_PATH.relative_to(ROOT)),
        "previous_21_16C_audit_status": previous_audit_load_status,
    },
    "counts": {
        "merged_records": int(len(merged)),
        "merged_rows_with_database_id": int(len(valid_database)),
        "mention_records": int(len(mentions)),
        "unified_identities": int(len(identity)),
        "valid_movielens_records": int(len(valid_ml)),
        "unique_valid_movielens_ids": int(valid_ml["movielensId"].nunique()),
        "conflicting_movielens_ids": int(len(conflicts)),
        "pairwise_comparisons": int(len(pair_df)),
        "expected_pairwise_comparisons": int(expected_pairwise_count),
        "database_ids_with_repeated_source_records": int(
            len(database_conflicts)
        ),
        "database_ids_with_identity_multiplicity": int(
            len(database_identity_conflicts)
        ),
        "mention_reference_matches": int(
            merged["mention_reference_found"].sum()
        ),
        "mention_title_matches": int(
            merged["mention_title_match"].sum()
        ),
        "mention_reference_missing": int(
            (~merged["mention_reference_found"]).sum()
        ),
    },
    "candidate_classification": candidate_counts,
    "previous_experiment_reconciliation": {
        "status": previous_reconciliation_status,
        "previous_conflict_group_count": int(len(previous_conflict_ids)),
        "current_conflict_group_count": int(len(current_conflict_ids)),
        "shared_conflict_group_count": int(len(shared_conflict_ids)),
        "previous_only_group_count": int(len(previous_only_ids)),
        "current_only_group_count": int(len(current_only_ids)),
        "previous_id_column": previous_conflict_ids_column,
        "previous_multiplicity_column": previous_count_column,
    },
    "validation": {
        key: (None if value is None else bool(value))
        for key, value in validation.items()
    },
    "identity_policy": {
        "title_match_is_identity_proof": False,
        "shared_movielens_id_is_identity_proof": False,
        "automatic_merging": False,
        "original_database_ids_preserved": True,
        "unresolved_records_preserved": True,
        "external_identity_verification": False,
        "candidate_classes_are_not_identity_decisions": True,
    },
    "outputs": [
        "movielens_conflict_group_classification.csv",
        "movielens_conflict_pairwise_analysis.csv",
        "database_id_repeated_record_audit.csv",
        "database_id_identity_multiplicity_audit.csv",
        "merged_mention_cross_reference.csv",
        "complete_movielens_group_inventory.csv",
        "previous_21_16C_conflict_reconciliation.csv",
        "experiment_21_16E_manifest.json",
    ],
    "completion_status": "COMPLETED_ALL_REQUIRED_VALIDATIONS_PASSED",
    "limitations": [
        "A shared MovieLens identifier does not prove that two database records "
        "are the same real-world movie.",
        "A title or release-year match is candidate evidence only.",
        "Mention-dataset title agreement is a cross-reference consistency "
        "check, not independent proof of movie identity.",
        "No external authoritative movie database was queried.",
        "No graph modification, model training, model inference, or entity "
        "merging was performed.",
    ],
}

manifest_path = OUTPUT / "experiment_21_16E_manifest.json"
with manifest_path.open("w", encoding="utf-8") as file:
    json.dump(manifest, file, indent=4, ensure_ascii=False)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("EXPERIMENT 21.16E COMPLETED")
print("=" * 75)

print("Output directory:", OUTPUT.resolve())

for output_file in sorted(OUTPUT.iterdir()):
    if output_file.is_file():
        print(
            f"{output_file.name} | "
            f"{output_file.stat().st_size} bytes"
        )

print("\nOriginal source files unchanged.")
print("Unified identity mapping unchanged.")
print("No graph modification.")
print("No model training or inference.")
print("No automatic entity merging.")
print("Candidate classifications are not verified identity decisions.")
print("\nEXPERIMENT 21.16E FINISHED.")
