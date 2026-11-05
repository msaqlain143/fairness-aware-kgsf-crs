
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ============================================================
# EXPERIMENT 21.17F
# IMDb Evidence Quality and Entity Granularity Audit
# ============================================================

ROOT = Path(__file__).resolve().parent

SOURCE = (
    ROOT / "results_category_audit"
    / "experiment_21_17E"
    / "imdb_identity_evidence_master.csv"
)

OUTPUT = (
    ROOT / "results_category_audit"
    / "experiment_21_17F"
)

EXPECTED_POPULATION = 6112
EXPECTED_IMDB_MATCHES = 577

SERIES_TERMS = re.compile(
    r"\b(film series|film franchise|television series|"
    r"film saga|media franchise)\b",
    re.IGNORECASE,
)


# ============================================================
# UTILITIES
# ============================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def parse_json(value, default=None):
    if default is None:
        default = []

    if value is None or pd.isna(value):
        return default

    if isinstance(value, (list, dict)):
        return value

    value = str(value).strip()

    if not value or value in ("nan", "None", "null"):
        return default

    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def clean(value):
    if value is None or pd.isna(value):
        return ""

    return str(value).strip()


def safe_bool(value):
    return clean(value).lower() in ("true", "1", "yes")


def normalize_identifier(value):
    value = clean(value)

    if not value:
        return ""

    value = value.removeprefix("tt")

    if value.isdigit():
        return "tt" + value.zfill(7)

    return ""


def normalize_title(value):
    value = clean(value).casefold()
    return re.sub(r"[^a-z0-9]+", "", value)


def get_year(value):
    match = re.search(r"\b(18|19|20)\d{2}\b", clean(value))
    return int(match.group()) if match else None


def unique_values(values):
    result = []

    for value in values:
        value = clean(value)

        if value and value not in result:
            result.append(value)

    return result


def write_csv(data, path):
    pd.DataFrame(data).to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# RECORD-LEVEL ASSESSMENT
# ============================================================

def assess_record(record):

    title_type = clean(
        record.get("imdb_title_type")
    )

    title_comparison = clean(
        record.get("title_comparison")
    )

    year_comparison = clean(
        record.get("year_comparison")
    )

    imdb_id = normalize_identifier(
        record.get("imdb_id")
    )

    if not imdb_id:
        return "INVALID_IMDB_IDENTIFIER"

    if title_type == "tvEpisode":
        return "TV_EPISODE_REVIEW"

    if title_type == "tvSeries":
        return "TV_SERIES_REVIEW"

    if title_type == "tvMiniSeries":
        return "TV_MINISERIES_REVIEW"

    if title_type == "tvSpecial":
        return "TV_SPECIAL_REVIEW"

    if year_comparison == "YEAR_CONFLICT":
        return "YEAR_CONFLICT_REVIEW"

    if title_comparison != "EXACT_NORMALIZED_LABEL_MATCH":
        return "TITLE_VARIATION_REVIEW"

    if year_comparison != "YEAR_AGREEMENT":
        return "YEAR_EVIDENCE_INCOMPLETE"

    if title_type == "movie":
        return "STRONG_MOVIE_CORROBORATION"

    if title_type == "short":
        return "STRONG_SHORT_CORROBORATION"

    if title_type == "video":
        return "NON_FEATURE_VIDEO_REVIEW"

    if title_type == "tvMovie":
        return "TV_MOVIE_REVIEW"

    return "OTHER_MEDIA_TYPE_REVIEW"


# ============================================================
# ENTITY-LEVEL GRANULARITY ASSESSMENT
# ============================================================

def assess_granularity(row, records):

    uri = clean(row.get("source_uri"))

    labels = parse_json(
        row.get("english_labels")
    )

    labels_text = " ".join(
        str(x) for x in labels
    )

    combined = uri + " " + labels_text

    identifiers = unique_values(
        record.get("imdb_id", "")
        for record in records
    )

    if SERIES_TERMS.search(combined):
        return "FILM_SERIES_ENTITY_REVIEW"

    if len(identifiers) > 1:

        titles = unique_values(
            record.get("imdb_primary_title", "")
            for record in records
        )

        years = unique_values(
            record.get("imdb_start_year", "")
            for record in records
        )

        if len(titles) > 1 or len(years) > 1:
            return "MULTIPLE_ADAPTATIONS_OR_RELATED_FILMS_REVIEW"

        return "MULTIPLE_IDENTIFIER_REVIEW"

    if not records:
        return "NO_OFFICIAL_IMDB_EVIDENCE"

    return "SINGLE_IDENTIFIER_ENTITY"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("EXPERIMENT 21.17F: IMDb EVIDENCE QUALITY AND GRANULARITY AUDIT")
    print("=" * 75)

    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Required source file not found: {SOURCE}"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    source_hash = sha256_file(SOURCE)

    print("\n[1] SOURCE INVENTORY")

    df = pd.read_csv(
        SOURCE,
        dtype=str,
        keep_default_na=False
    )

    print("Source identities:", len(df))
    print("Source columns:", len(df.columns))

    required = [
        "movie_entity_id",
        "source_uri",
        "official_imdb_records",
        "identity_assessment",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if len(df) != EXPECTED_POPULATION:
        raise ValueError(
            f"Expected {EXPECTED_POPULATION} identities; "
            f"found {len(df)}."
        )

    if df["movie_entity_id"].duplicated().any():
        raise ValueError(
            "Duplicate internal entity IDs detected."
        )

    print("\n[2] SOURCE INTEGRITY")

    checks = {
        "population_6112": len(df) == EXPECTED_POPULATION,
        "internal_ids_unique": not df[
            "movie_entity_id"
        ].duplicated().any(),
        "source_uris_nonblank": df[
            "source_uri"
        ].map(bool).all(),
    }

    for name, result in checks.items():
        print(
            f"{name}: {'PASS' if result else 'FAIL'}"
        )

    if not all(checks.values()):
        raise RuntimeError(
            "Source integrity validation failed."
        )

    print("\n[3] RECORD-LEVEL EVIDENCE AUDIT")

    entity_results = []
    record_results = []

    for index, row in df.iterrows():

        if index > 0 and index % 1000 == 0:
            print(
                f"Processed {index:,}/{len(df):,}"
            )

        records = parse_json(
            row["official_imdb_records"]
        )

        if not isinstance(records, list):
            records = []

        entity_id = clean(
            row["movie_entity_id"]
        )

        valid_records = []

        for record in records:

            if not isinstance(record, dict):
                continue

            assessment = assess_record(record)

            imdb_id = normalize_identifier(
                record.get("imdb_id")
            )

            output_record = {
                "movie_entity_id": entity_id,
                "source_uri": row["source_uri"],
                "imdb_id": imdb_id,
                "imdb_title": clean(
                    record.get("imdb_primary_title")
                ),
                "imdb_original_title": clean(
                    record.get("imdb_original_title")
                ),
                "imdb_start_year": clean(
                    record.get("imdb_start_year")
                ),
                "imdb_title_type": clean(
                    record.get("imdb_title_type")
                ),
                "imdb_genres": clean(
                    record.get("imdb_genres")
                ),
                "title_comparison": clean(
                    record.get("title_comparison")
                ),
                "year_comparison": clean(
                    record.get("year_comparison")
                ),
                "record_evidence_assessment": assessment,
                "automatic_identity_verification": False,
                "automatic_merging": False,
            }

            record_results.append(output_record)
            valid_records.append(record)

        granularity = assess_granularity(
            row,
            valid_records
        )

        record_assessments = [
            assess_record(r)
            for r in valid_records
        ]

        strong_movie = sum(
            x == "STRONG_MOVIE_CORROBORATION"
            for x in record_assessments
        )

        strong_short = sum(
            x == "STRONG_SHORT_CORROBORATION"
            for x in record_assessments
        )

        entity_results.append({
            "movie_entity_id": entity_id,
            "source_uri": row["source_uri"],
            "previous_identity_assessment":
                row["identity_assessment"],
            "official_imdb_record_count":
                len(valid_records),
            "unique_imdb_identifier_count":
                len(set(
                    normalize_identifier(
                        r.get("imdb_id")
                    )
                    for r in valid_records
                    if normalize_identifier(
                        r.get("imdb_id")
                    )
                )),
            "strong_movie_record_count": strong_movie,
            "strong_short_record_count": strong_short,
            "granularity_assessment": granularity,
            "record_assessments": json.dumps(
                record_assessments,
                ensure_ascii=False
            ),
            "identity_verification": "NOT_AUTOMATICALLY_VERIFIED",
            "automatic_merging": False,
        })

    print("\n[4] POPULATION AND MEDIA SUMMARY")

    master = pd.DataFrame(entity_results)
    record_df = pd.DataFrame(record_results)

    media_summary = (
        record_df.groupby(
            "imdb_title_type",
            dropna=False
        )
        .size()
        .reset_index(name="record_count")
    )

    assessment_summary = (
        record_df.groupby(
            "record_evidence_assessment"
        )
        .size()
        .reset_index(name="record_count")
    )

    print("Entity population:", len(master))
    print("Official IMDb records:", len(record_df))

    print("\nMedia types:")
    print(media_summary.to_string(index=False))

    print("\nRecord evidence assessments:")
    print(assessment_summary.to_string(index=False))

    print("\n[5] EXCEPTION REGISTERS")

    exceptions = record_df[
        ~record_df[
            "record_evidence_assessment"
        ].isin([
            "STRONG_MOVIE_CORROBORATION",
            "STRONG_SHORT_CORROBORATION",
        ])
    ].copy()

    granularity_review = master[
        ~master[
            "granularity_assessment"
        ].isin([
            "SINGLE_IDENTIFIER_ENTITY",
            "NO_OFFICIAL_IMDB_EVIDENCE",
        ])
    ].copy()

    print("Record-level exceptions:", len(exceptions))
    print("Granularity-review entities:", len(granularity_review))

    print("\n[6] FINAL VALIDATION")

    unique_ids = record_df[
        "imdb_id"
    ].dropna()

    checks.update({
        "master_population_preserved":
            len(master) == EXPECTED_POPULATION,

        "master_entity_ids_unique":
            master["movie_entity_id"].is_unique,

        "all_source_uris_preserved":
            master["source_uri"].notna().all(),

        "record_population_consistent":
            len(record_df) == sum(
                len(parse_json(x))
                for x in df["official_imdb_records"]
            ),

        "no_automatic_identity_verification":
            (
                master["identity_verification"]
                == "NOT_AUTOMATICALLY_VERIFIED"
            ).all(),

        "no_automatic_merging":
            (~master["automatic_merging"]).all(),

        "unresolved_entities_preserved":
            len(master) == EXPECTED_POPULATION,
    })

    for name, result in checks.items():
        print(
            f"{name}: {'PASS' if result else 'FAIL'}"
        )

    if not all(checks.values()):
        raise RuntimeError(
            "Final validation failed. "
            "Inspect source and output records."
        )

    print("\n[7] EXPORTING RESULTS")

    outputs = {
        "imdb_evidence_quality_master.csv": master,
        "imdb_evidence_record_level.csv": record_df,
        "imdb_evidence_exception_register.csv": exceptions,
        "imdb_media_type_summary.csv": media_summary,
        "imdb_granularity_review.csv": granularity_review,
        "imdb_record_assessment_summary.csv": assessment_summary,
    }

    for filename, data in outputs.items():

        path = OUTPUT / filename

        data.to_csv(
            path,
            index=False,
            encoding="utf-8-sig"
        )

        print(
            f"{filename}: {len(data):,} rows, "
            f"{path.stat().st_size:,} bytes"
        )

    manifest = {
        "experiment": "21.17F",
        "name": (
            "IMDb Evidence Quality and "
            "Entity Granularity Audit"
        ),
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "source": str(
            SOURCE.relative_to(ROOT)
        ),

        "source_sha256": source_hash,

        "counts": {
            "source_entities": len(df),
            "master_entities": len(master),
            "official_imdb_records": len(record_df),
            "record_exceptions": len(exceptions),
            "granularity_review_entities":
                len(granularity_review),
        },

        "media_type_distribution": {
            str(row["imdb_title_type"]):
                int(row["record_count"])
            for _, row in media_summary.iterrows()
        },

        "record_assessment_distribution": {
            str(row["record_evidence_assessment"]):
                int(row["record_count"])
            for _, row in assessment_summary.iterrows()
        },

        "validation": {
            key: bool(value)
            for key, value in checks.items()
        },

        "methodology": {
            "identifier_matching":
                "Preserved official IMDb identifiers",
            "title_comparison":
                "Inherited record-level official dataset comparison",
            "year_comparison":
                "Inherited record-level official dataset comparison",
            "media_type_assessment":
                "Explicit IMDb title-type classification",
            "entity_granularity":
                "Multiple identifier and series-label screening",
            "automatic_identity_verification": False,
            "automatic_entity_merging": False,
            "original_source_files_modified": False,
            "knowledge_graph_modified": False,
        },

        "limitations": [
            "Strong corroboration is not absolute identity proof.",
            "Series detection uses conservative textual screening.",
            "Multiple identifiers may represent adaptations or related works.",
            "The source comparison results are inherited from 21.17E.",
            "No new independent IMDb retrieval was performed.",
        ],

        "outputs": list(outputs.keys()),

        "completion_status":
            "COMPLETED_ALL_REQUIRED_VALIDATIONS_PASSED",
    }

    manifest_path = (
        OUTPUT / "experiment_21_17F_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=4,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    report = [
        "EXPERIMENT 21.17F",
        "IMDb Evidence Quality and Entity Granularity Audit",
        "=" * 65,
        f"Source entities: {len(df)}",
        f"Master entities: {len(master)}",
        f"Official IMDb records: {len(record_df)}",
        f"Record exceptions: {len(exceptions)}",
        f"Granularity reviews: {len(granularity_review)}",
        "",
        "VALIDATION",
    ]

    report.extend(
        f"{name}: {'PASS' if result else 'FAIL'}"
        for name, result in checks.items()
    )

    report.extend([
        "",
        "No automatic identity verification.",
        "No automatic entity merging.",
        "No knowledge graph modification.",
        "Original source files unchanged.",
    ])

    (
        OUTPUT / "reproducibility_report.txt"
    ).write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("\n" + "=" * 75)
    print("EXPERIMENT 21.17F COMPLETED")
    print("=" * 75)

    print("Output:", OUTPUT)
    print("Manifest:", manifest_path.name)
    print("Reproducibility report: generated")

    print("\nNo graph modification.")
    print("No model training or inference.")
    print("No automatic identity merging.")


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\nEXPERIMENT 21.17F FAILED")
        print(f"{type(error).__name__}: {error}")

        sys.exit(1)
