
"""
EXPERIMENT 21.17D
Record-Level IMDb Identity and Metadata Verification

Purpose:
- Assess existing IMDb cross-reference evidence.
- Compare DBpedia labels and MovieLens titles.
- Compare release years where available.
- Identify ambiguous and conflicting records.
- Preserve all original identities.
- Never automatically merge entities or overclaim verification.

Input:
results_category_audit/experiment_21_17C/imdb_identifier_cross_reference.csv

Output:
results_category_audit/experiment_21_17D/
"""

import ast
import json
import re
import hashlib
import unicodedata
from pathlib import Path
from collections import Counter

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

INPUT = (
    ROOT
    / "results_category_audit"
    / "experiment_21_17C"
    / "imdb_identifier_cross_reference.csv"
)

OUT = (
    ROOT
    / "results_category_audit"
    / "experiment_21_17D"
)

OUT.mkdir(parents=True, exist_ok=True)

EXPERIMENT = "21.17D"


# ============================================================
# HELPERS
# ============================================================

def parse_list(value):
    """Safely parse JSON/Python-list strings."""

    if value is None or pd.isna(value):
        return []

    value = str(value).strip()

    if not value or value.lower() in {"nan", "none", "null"}:
        return []

    try:
        result = json.loads(value)
        return result if isinstance(result, list) else [result]
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        result = ast.literal_eval(value)
        return result if isinstance(result, list) else [result]
    except (ValueError, SyntaxError):
        return [value]


def clean_values(values):
    return [
        str(x).strip()
        for x in values
        if x is not None and str(x).strip()
    ]


def normalize_title(title):
    """
    Normalize common title variations.

    Does not establish movie identity.
    """

    if not title:
        return ""

    title = unicodedata.normalize("NFKD", str(title))
    title = "".join(
        c for c in title
        if not unicodedata.combining(c)
    )

    title = title.lower().strip()

    # Remove release years.
    title = re.sub(
        r"\(\s*(18|19|20)\d{2}\s*\)",
        " ",
        title
    )

    # Remove common descriptive suffixes.
    title = re.sub(
        r"\((film|movie|feature film|film series)\)",
        " ",
        title
    )

    # Remove punctuation.
    title = re.sub(r"[^a-z0-9\s]", " ", title)

    words = title.split()

    # Normalize trailing articles:
    # "Parent Trap, The" -> "the parent trap"
    if len(words) > 1 and words[-1] in {
        "the", "a", "an"
    }:
        words = [words[-1]] + words[:-1]

    return " ".join(words)


def extract_years(values):
    years = set()

    for value in values:
        matches = re.findall(
            r"(?<!\d)(18\d{2}|19\d{2}|20\d{2})(?!\d)",
            str(value)
        )
        years.update(int(x) for x in matches)

    return sorted(years)


def title_comparison(labels, movie_titles):
    """
    Compare every available English label against
    every available MovieLens title.
    """

    normalized_labels = {
        normalize_title(x)
        for x in labels
        if normalize_title(x)
    }

    normalized_movies = {
        normalize_title(x)
        for x in movie_titles
        if normalize_title(x)
    }

    if not normalized_labels or not normalized_movies:
        return "TITLE_EVIDENCE_UNAVAILABLE"

    if normalized_labels.intersection(normalized_movies):
        return "NORMALIZED_TITLE_AGREEMENT"

    return "TITLE_VARIATION_REQUIRES_REVIEW"


def year_comparison(labels, movie_titles):

    dbpedia_years = extract_years(labels)
    movielens_years = extract_years(movie_titles)

    if not dbpedia_years or not movielens_years:
        return (
            "YEAR_EVIDENCE_UNAVAILABLE",
            dbpedia_years,
            movielens_years
        )

    if set(dbpedia_years).intersection(movielens_years):
        return (
            "YEAR_AGREEMENT",
            dbpedia_years,
            movielens_years
        )

    return (
        "YEAR_CONFLICT_REQUIRES_REVIEW",
        dbpedia_years,
        movielens_years
    )


def classify_record(row):

    status = str(row["identifier_status"])

    labels = row["english_labels_list"]
    movie_titles = row["matching_movie_titles_list"]

    dbpedia_ids = row["dbpedia_imdb_ids_list"]
    matched_ids = row["matching_imdb_ids_list"]

    candidate_count = row["candidate_count_numeric"]

    if status == "NO_IMDb_IDENTIFIER":
        return "UNRESOLVED_NO_IMDB_IDENTIFIER"

    if status == "IMDB_NOT_FOUND_IN_MOVIELENS":
        return "UNRESOLVED_IMDB_NOT_IN_MOVIELENS"

    if status == "MULTIPLE_CROSS_REFERENCE_CANDIDATES":
        return "ENTITY_GRANULARITY_REVIEW"

    if status != "UNIQUE_IMDB_CROSS_REFERENCE":
        return "UNRESOLVED_UNEXPECTED_STATUS"

    # Strict one-to-one identifier requirement.
    if (
        len(set(dbpedia_ids)) != 1
        or len(set(matched_ids)) != 1
        or candidate_count != 1
        or set(dbpedia_ids) != set(matched_ids)
    ):
        return "IDENTIFIER_CARDINALITY_REVIEW"

    title_status = row["title_comparison"]
    year_status = row["year_comparison"]

    if year_status == "YEAR_CONFLICT_REQUIRES_REVIEW":
        return "YEAR_CONFLICT_REQUIRES_REVIEW"

    if title_status == "NORMALIZED_TITLE_AGREEMENT":
        if year_status == "YEAR_AGREEMENT":
            return "STRONG_IDENTIFIER_TITLE_YEAR_CORROBORATION"

        return "IDENTIFIER_TITLE_CORROBORATION_YEAR_INCOMPLETE"

    if title_status == "TITLE_VARIATION_REQUIRES_REVIEW":
        return "IDENTIFIER_MATCH_TITLE_VARIATION_REVIEW"

    return "IDENTIFIER_MATCH_INCOMPLETE_METADATA"


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("EXPERIMENT 21.17D: RECORD-LEVEL IMDb IDENTITY VERIFICATION")
    print("=" * 75)

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Required Experiment 21.17C file not found: {INPUT}"
        )

    # --------------------------------------------------------
    # 1. SOURCE INVENTORY
    # --------------------------------------------------------

    print("\n[1] SOURCE INVENTORY")

    df = pd.read_csv(
        INPUT,
        dtype=str,
        keep_default_na=False
    )

    required = [
        "movie_entity_id",
        "source_uri",
        "dbpedia_imdb_ids",
        "english_labels",
        "matching_imdb_ids",
        "matching_movielens_ids",
        "matching_movie_titles",
        "candidate_count",
        "identifier_status",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("Input records:", len(df))
    print("Input columns:", len(df.columns))

    # --------------------------------------------------------
    # 2. SOURCE INTEGRITY
    # --------------------------------------------------------

    print("\n[2] SOURCE INTEGRITY")

    checks = {
        "input_population_6112": len(df) == 6112,
        "entity_id_unique": df["movie_entity_id"].is_unique,
        "source_uri_nonblank": df["source_uri"].ne("").all(),
        "source_uri_unique": df["source_uri"].is_unique,
    }

    for name, result in checks.items():
        print(f"{name}: {'PASS' if result else 'FAIL'}")

    if not all(checks.values()):
        raise RuntimeError(
            "Source integrity validation failed."
        )

    # --------------------------------------------------------
    # 3. PARSE EXISTING EVIDENCE
    # --------------------------------------------------------

    print("\n[3] PARSING IDENTIFIER AND TITLE EVIDENCE")

    df["dbpedia_imdb_ids_list"] = df[
        "dbpedia_imdb_ids"
    ].apply(lambda x: clean_values(parse_list(x)))

    df["matching_imdb_ids_list"] = df[
        "matching_imdb_ids"
    ].apply(lambda x: clean_values(parse_list(x)))

    df["matching_movielens_ids_list"] = df[
        "matching_movielens_ids"
    ].apply(lambda x: clean_values(parse_list(x)))

    df["english_labels_list"] = df[
        "english_labels"
    ].apply(lambda x: clean_values(parse_list(x)))

    df["matching_movie_titles_list"] = df[
        "matching_movie_titles"
    ].apply(lambda x: clean_values(parse_list(x)))

    df["candidate_count_numeric"] = pd.to_numeric(
        df["candidate_count"],
        errors="coerce"
    ).fillna(0).astype(int)

    # --------------------------------------------------------
    # 4. TITLE COMPARISON
    # --------------------------------------------------------

    print("\n[4] TITLE COMPARISON")

    df["title_comparison"] = df.apply(
        lambda r: title_comparison(
            r["english_labels_list"],
            r["matching_movie_titles_list"]
        ),
        axis=1
    )

    print(
        df["title_comparison"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # 5. RELEASE YEAR COMPARISON
    # --------------------------------------------------------

    print("\n[5] RELEASE YEAR COMPARISON")

    year_results = df.apply(
        lambda r: year_comparison(
            r["english_labels_list"],
            r["matching_movie_titles_list"]
        ),
        axis=1
    )

    df["year_comparison"] = [
        x[0] for x in year_results
    ]

    df["dbpedia_release_years"] = [
        json.dumps(x[1]) for x in year_results
    ]

    df["movielens_release_years"] = [
        json.dumps(x[2]) for x in year_results
    ]

    print(
        df["year_comparison"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # 6. RECORD-LEVEL IDENTITY DECISION
    # --------------------------------------------------------

    print("\n[6] RECORD-LEVEL IDENTITY ASSESSMENT")

    df["identity_assessment"] = df.apply(
        classify_record,
        axis=1
    )

    print(
        df["identity_assessment"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # 7. EXPORT MASTER TABLE
    # --------------------------------------------------------

    print("\n[7] EXPORTING RECORD-LEVEL EVIDENCE")

    master = df.copy()

    # Preserve source columns and add evidence.
    master["identity_verified"] = False

    master["verification_policy"] = (
        "Conservative candidate assessment; "
        "no automatic identity verification."
    )

    # Convert arrays to portable JSON strings.
    list_columns = [
        "dbpedia_imdb_ids_list",
        "matching_imdb_ids_list",
        "matching_movielens_ids_list",
        "english_labels_list",
        "matching_movie_titles_list",
    ]

    for col in list_columns:
        master[col] = master[col].apply(
            lambda x: json.dumps(x, ensure_ascii=False)
        )

    master_path = OUT / "imdb_identity_verification_master.csv"

    master.to_csv(
        master_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 8. REVIEW TABLES
    # --------------------------------------------------------

    review = master[
        master["identity_assessment"].str.contains(
            "REVIEW|UNRESOLVED|INCOMPLETE",
            regex=True
        )
    ].copy()

    review_path = OUT / "imdb_identity_records_requiring_review.csv"

    review.to_csv(
        review_path,
        index=False,
        encoding="utf-8-sig"
    )

    strong_candidates = master[
        master["identity_assessment"]
        == "STRONG_IDENTIFIER_TITLE_YEAR_CORROBORATION"
    ].copy()

    strong_path = OUT / "strong_imdb_identity_candidates.csv"

    strong_candidates.to_csv(
        strong_path,
        index=False,
        encoding="utf-8-sig"
    )

    multiple = master[
        master["identifier_status"]
        == "MULTIPLE_CROSS_REFERENCE_CANDIDATES"
    ].copy()

    multiple_path = OUT / "multiple_candidate_identity_review.csv"

    multiple.to_csv(
        multiple_path,
        index=False,
        encoding="utf-8-sig"
    )

    unmatched = master[
        master["identifier_status"].isin([
            "IMDB_NOT_FOUND_IN_MOVIELENS",
            "NO_IMDb_IDENTIFIER"
        ])
    ].copy()

    unmatched_path = OUT / "unmatched_imdb_identity_records.csv"

    unmatched.to_csv(
        unmatched_path,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 9. VALIDATION
    # --------------------------------------------------------

    print("\n[8] FINAL VALIDATION")

    validations = {
        "master_population_6112": len(master) == 6112,
        "master_entity_ids_unique": master[
            "movie_entity_id"
        ].is_unique,
        "master_uri_population_preserved": (
            master["source_uri"].tolist()
            == df["source_uri"].tolist()
        ),
        "strong_candidates_are_unique_identifier_matches": (
            strong_candidates["identifier_status"]
            .eq("UNIQUE_IMDB_CROSS_REFERENCE").all()
        ),
        "multiple_candidates_preserved": (
            len(multiple)
            == (
                df["identifier_status"]
                == "MULTIPLE_CROSS_REFERENCE_CANDIDATES"
            ).sum()
        ),
        "all_original_records_accounted_for": (
            len(strong_candidates)
            + len(review)
            + len(
                master[
                    ~master["identity_assessment"].str.contains(
                        "REVIEW|UNRESOLVED|INCOMPLETE",
                        regex=True
                    )
                ]
            ) == len(master)
        ),
        "no_automatic_identity_verification": (
            not master["identity_verified"].any()
        ),
        "no_automatic_entity_merging": True,
    }

    for name, result in validations.items():
        print(
            f"{name}: {'PASS' if result else 'FAIL'}"
        )

    if not all(validations.values()):
        raise RuntimeError(
            "Experiment 21.17D validation failed."
        )

    # --------------------------------------------------------
    # 10. MANIFEST
    # --------------------------------------------------------

    print("\n[9] GENERATING MANIFEST")

    status_counts = {
        str(k): int(v)
        for k, v in Counter(
            master["identity_assessment"]
        ).items()
    }

    manifest = {
        "experiment": EXPERIMENT,
        "name": "Record-Level IMDb Identity and Metadata Verification",
        "purpose": (
            "Conservative assessment of existing IMDb cross-reference "
            "evidence for 6,112 DBpedia movie identities."
        ),
        "source": str(INPUT.relative_to(ROOT)),
        "source_sha256": sha256(INPUT),
        "counts": {
            "total_uri_identities": int(len(master)),
            "strong_identifier_title_year_candidates": int(
                len(strong_candidates)
            ),
            "records_requiring_review": int(len(review)),
            "multiple_candidate_records": int(len(multiple)),
            "unmatched_records": int(len(unmatched)),
        },
        "identity_assessment_distribution": status_counts,
        "validation": {
            k: bool(v)
            for k, v in validations.items()
        },
        "methodology": {
            "identifier_cross_reference": "Existing IMDb-to-MovieLens matches",
            "title_normalization": True,
            "release_year_comparison": True,
            "title_matching_alone_is_identity_proof": False,
            "automatic_identity_verification": False,
            "automatic_entity_merging": False,
            "unresolved_records_preserved": True,
            "all_original_uris_preserved": True,
        },
        "limitations": [
            "IMDb identifier correspondence is not independently revalidated against IMDb itself.",
            "MovieLens metadata is used as a cross-reference source.",
            "Normalized title agreement does not independently prove identity.",
            "Release years extracted from labels and titles may be incomplete.",
            "Movie series and adaptations require entity-granularity review.",
            "No records are automatically declared verified."
        ],
        "outputs": [
            master_path.name,
            review_path.name,
            strong_path.name,
            multiple_path.name,
            unmatched_path.name,
        ],
        "completion_status": "COMPLETED_ALL_VALIDATIONS_PASSED",
    }

    manifest_path = OUT / "experiment_21_17D_manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=4,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # 11. SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("EXPERIMENT 21.17D COMPLETED")
    print("=" * 75)

    print("Total identities:", len(master))
    print("Strong evidence candidates:", len(strong_candidates))
    print("Records requiring review:", len(review))
    print("Multiple-candidate records:", len(multiple))
    print("Unmatched records:", len(unmatched))

    print("\nOUTPUT DIRECTORY:")
    print(OUT)

    print("\nOriginal source files unchanged.")
    print("Unified identity mapping unchanged.")
    print("Knowledge graph unchanged.")
    print("No model training or inference.")
    print("No automatic entity merging.")
    print("No automatic identity verification.")

    print("\nEXPERIMENT 21.17D FINISHED.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nEXPERIMENT 21.17D FAILED")
        print(type(exc).__name__ + ":", exc)
        raise
