from pathlib import Path
from datetime import datetime, timezone, date
import hashlib
import json
import re
import sys
import unicodedata
import urllib.parse

import numpy as np
import pandas as pd


# ============================================================
# EXPERIMENT 21.17H
# TMDb EVIDENCE QUALITY AND IDENTITY VERIFICATION
# ============================================================

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results_category_audit"

SOURCES = {
    "tmdb_master": BASE / "experiment_21_17G" / "tmdb_identity_evidence_master.csv",
    "tmdb_crossref": BASE / "experiment_21_17G" / "tmdb_identifier_cross_reference.csv",
    "tmdb_conflicts": BASE / "experiment_21_17G" / "tmdb_identifier_conflicts.csv",
    "imdb_master": BASE / "experiment_21_17F" / "imdb_evidence_quality_master.csv",
    "imdb_records": BASE / "experiment_21_17F" / "imdb_evidence_record_level.csv",
    "dbpedia": BASE / "experiment_21_17B" / "dbpedia_independent_metadata_evidence.csv",
}

OUT = BASE / "experiment_21_17H"

EXPECTED_ENTITIES = 6112
EXPECTED_IDENTIFIER_PAIRS = 578
EXPECTED_CONFLICTS = 4


# ============================================================
# GENERAL FUNCTIONS
# ============================================================

def log(message):
    print(message, flush=True)


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required source missing: {path}"
        )

    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False
    )


def parse_json(value):
    if value is None:
        return []

    if isinstance(value, (list, dict)):
        return value

    value = str(value).strip()

    if not value or value.casefold() in {
        "nan", "none", "null", "\\n"
    }:
        return []

    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return []


def json_safe(value):
    """Convert NumPy, Pandas and date objects to JSON-safe values."""

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable"
    )


def sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def normalize_title(value):
    if value is None:
        return ""

    value = unicodedata.normalize(
        "NFKC", str(value)
    ).casefold()

    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"\[[^]]*\]", " ", value)
    value = re.sub(
        r"[^\w]+", " ", value, flags=re.UNICODE
    )

    return " ".join(value.split())


def extract_year(value):
    if value is None:
        return None

    match = re.search(
        r"\b(18|19|20)\d{2}\b",
        str(value)
    )

    return int(match.group(0)) if match else None


def unique(values):
    result = []
    seen = set()

    for value in values:
        if value is None:
            continue

        item = str(value).strip()

        if (
            item
            and item.casefold() not in {
                "nan", "none", "null"
            }
            and item not in seen
        ):
            seen.add(item)
            result.append(item)

    return result


def extract_values(value, languages=None):
    records = parse_json(value)

    if not isinstance(records, list):
        return []

    output = []

    for record in records:

        if isinstance(record, dict):

            language = record.get("language", "")

            if (
                languages is not None
                and language not in languages
            ):
                continue

            item = record.get("value", "")

        else:
            item = record

        if item:
            output.append(str(item))

    return unique(output)


def get_source_years(row):
    years = []

    for label in row.get("english_labels", []):
        year = extract_year(label)

        if year:
            years.append(year)

    for release_date in row.get(
        "release_dates_list", []
    ):
        year = extract_year(release_date)

        if year:
            years.append(year)

    uri = urllib.parse.unquote(
        str(row.get("source_uri", ""))
    )

    year = extract_year(uri)

    if year:
        years.append(year)

    return sorted(set(years))


def compare_years(source_years, candidate_year):

    if not candidate_year:
        return "YEAR_EVIDENCE_INCOMPLETE"

    if not source_years:
        return "SOURCE_YEAR_UNAVAILABLE"

    if candidate_year in source_years:
        return "YEAR_AGREEMENT"

    return "YEAR_CONFLICT"


def compare_titles(source_titles, candidate_titles):

    source_norm = {
        normalize_title(x)
        for x in source_titles
        if normalize_title(x)
    }

    candidate_norm = {
        normalize_title(x)
        for x in candidate_titles
        if normalize_title(x)
    }

    if not source_norm or not candidate_norm:
        return "TITLE_EVIDENCE_UNAVAILABLE"

    if source_norm.intersection(candidate_norm):
        return "EXACT_NORMALIZED_TITLE_AGREEMENT"

    return "TITLE_VARIATION_REVIEW"


def pair_key(row):
    return (
        str(row.get("movie_entity_id", "")).strip(),
        str(row.get("dbpedia_imdb_id", "")).strip()
    )


# ============================================================
# SOURCE LOADING AND VALIDATION
# ============================================================

def load_sources():

    log("\n[1] SOURCE INVENTORY")

    sources = {}

    for name, path in SOURCES.items():

        sources[name] = read_csv(path)

        log(
            f"{name}: {len(sources[name]):,} records"
        )

    return sources


def validate_sources(sources):

    log("\n[2] SOURCE INTEGRITY")

    checks = {
        "tmdb_master_population_6112":
            len(sources["tmdb_master"]) == EXPECTED_ENTITIES,

        "tmdb_crossref_population_578":
            len(sources["tmdb_crossref"]) == EXPECTED_IDENTIFIER_PAIRS,

        "conflict_population_4":
            len(sources["tmdb_conflicts"]) == EXPECTED_CONFLICTS,

        "dbpedia_population_6112":
            len(sources["dbpedia"]) == EXPECTED_ENTITIES,

        "imdb_master_population_6112":
            len(sources["imdb_master"]) == EXPECTED_ENTITIES,

        "tmdb_master_entity_ids_unique":
            sources["tmdb_master"]["movie_entity_id"].is_unique,

        "dbpedia_entity_ids_unique":
            sources["dbpedia"]["movie_entity_id"].is_unique,

        "imdb_master_entity_ids_unique":
            sources["imdb_master"]["movie_entity_id"].is_unique,

        "crossref_identifiers_nonblank":
            sources["tmdb_crossref"][
                "dbpedia_imdb_id"
            ].str.strip().ne("").all(),

        "crossref_entity_ids_nonblank":
            sources["tmdb_crossref"][
                "movie_entity_id"
            ].str.strip().ne("").all(),
    }

    crossref_pairs = [
        pair_key(row)
        for _, row in sources["tmdb_crossref"].iterrows()
    ]

    checks["crossref_identifier_pairs_unique"] = (
        len(crossref_pairs) == len(set(crossref_pairs))
    )

    for name, passed in checks.items():

        log(
            f"{name}: {'PASS' if passed else 'FAIL'}"
        )

    if not all(checks.values()):
        raise ValueError(
            "Source integrity validation failed."
        )

    return checks


# ============================================================
# SOURCE PREPARATION
# ============================================================

def prepare_dbpedia(df):

    records = []

    for _, row in df.iterrows():

        records.append({

            "movie_entity_id": row["movie_entity_id"],

            "source_uri": row["source_uri"],

            "english_labels": extract_values(
                row["labels"],
                languages={"en"}
            ),

            "all_labels": extract_values(
                row["labels"]
            ),

            "release_dates_list": extract_values(
                row["release_dates"]
            ),

            "dbpedia_imdb_ids": extract_values(
                row["imdb_ids"]
            ),

            "dbpedia_tmdb_ids": extract_values(
                row["tmdb_ids"]
            ),
        })

    result = pd.DataFrame(records)

    result["source_years"] = result.apply(
        get_source_years,
        axis=1
    )

    return result


def prepare_imdb(df):

    records = {}

    required = {
        "movie_entity_id",
        "imdb_id",
        "imdb_title",
        "imdb_original_title",
        "imdb_start_year",
        "imdb_title_type",
        "record_evidence_assessment",
    }

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            "IMDb record-level source missing columns: "
            f"{sorted(missing)}"
        )

    for _, row in df.iterrows():

        key = (
            row["movie_entity_id"],
            row["imdb_id"]
        )

        records[key] = {

            "imdb_title": row["imdb_title"],

            "imdb_original_title": row[
                "imdb_original_title"
            ],

            "imdb_year": row["imdb_start_year"],

            "imdb_type": row["imdb_title_type"],

            "imdb_evidence": row[
                "record_evidence_assessment"
            ],
        }

    return records


# ============================================================
# TMDb RECORD-LEVEL EVIDENCE
# ============================================================

def build_record_evidence(sources, dbpedia, imdb):

    log("\n[3] RECORD-LEVEL TMDb EVIDENCE")

    dbpedia_map = (
        dbpedia.set_index("movie_entity_id")
        .to_dict("index")
    )

    output = []

    crossref = sources["tmdb_crossref"]

    total = len(crossref)

    for index, (_, row) in enumerate(
        crossref.iterrows(), 1
    ):

        entity = str(row["movie_entity_id"])

        imdb_id = str(row["dbpedia_imdb_id"])

        source = dbpedia_map.get(entity)

        if source is None:
            raise ValueError(
                f"Cross-reference entity {entity} "
                "is absent from DBpedia metadata."
            )

        imdb_info = imdb.get(
            (entity, imdb_id), {}
        )

        uri_slug = urllib.parse.unquote(
            str(row["source_uri"]).rsplit("/", 1)[-1]
        )

        source_titles = unique(
            source["english_labels"]
            + [uri_slug.replace("_", " ")]
        )

        results = parse_json(
            row.get("tmdb_results", "")
        )

        if not isinstance(results, list):
            results = []

        results = [
            item for item in results
            if isinstance(item, dict)
        ]

        # Preserve pairs with no TMDb candidates.
        if not results:

            output.append({

                "movie_entity_id": entity,

                "source_uri": row["source_uri"],

                "dbpedia_imdb_id": imdb_id,

                "tmdb_id": "",

                "tmdb_title": "",

                "tmdb_original_title": "",

                "tmdb_media_type": "",

                "tmdb_release_date": "",

                "tmdb_year": "",

                "tmdb_request_status": row.get(
                    "request_status", ""
                ),

                "tmdb_http_status": row.get(
                    "http_status", ""
                ),

                "tmdb_result_count_reported": row.get(
                    "tmdb_result_count", "0"
                ),

                "dbpedia_english_labels": json.dumps(
                    source["english_labels"],
                    ensure_ascii=False
                ),

                "dbpedia_years": json.dumps(
                    source["source_years"]
                ),

                "title_comparison":
                    "TITLE_EVIDENCE_UNAVAILABLE",

                "year_comparison":
                    "YEAR_EVIDENCE_INCOMPLETE",

                "imdb_title_comparison": compare_titles(
                    source_titles,
                    [
                        imdb_info.get("imdb_title", ""),
                        imdb_info.get(
                            "imdb_original_title", ""
                        )
                    ]
                ),

                "imdb_tmdb_year_comparison":
                    "INCOMPLETE",

                "imdb_evidence": imdb_info.get(
                    "imdb_evidence", ""
                ),

                "movielens_shared_tmdb_id": False,

                "record_evidence_assessment":
                    "NO_TMDB_RESULT",

                "automatic_identity_verification": False,

                "automatic_entity_merging": False,
            })

            if index % 100 == 0:
                log(
                    f"Processed {index:,}/{total:,}"
                )

            continue

        shared_ids = parse_json(
            row.get("shared_tmdb_ids", "")
        )

        shared_ids = {
            str(value) for value in shared_ids
        }

        for result in results:

            tmdb_id = str(
                result.get("tmdb_id", "")
            )

            tmdb_title = str(
                result.get("tmdb_title", "")
            )

            original_title = str(
                result.get("tmdb_original_title", "")
            )

            media_type = str(
                result.get("tmdb_media_type", "")
            ).lower()

            release_date = str(
                result.get("tmdb_release_date", "")
            )

            tmdb_year = extract_year(
                release_date
            )

            title_result = compare_titles(
                source_titles,
                [tmdb_title, original_title]
            )

            year_result = compare_years(
                source["source_years"],
                tmdb_year
            )

            imdb_title_result = compare_titles(
                source_titles,
                [
                    imdb_info.get("imdb_title", ""),
                    imdb_info.get(
                        "imdb_original_title", ""
                    )
                ]
            )

            imdb_year = extract_year(
                imdb_info.get("imdb_year", "")
            )

            if imdb_year and tmdb_year:

                imdb_tmdb_year_agreement = (
                    "AGREEMENT"
                    if tmdb_year == imdb_year
                    else "CONFLICT"
                )

            else:
                imdb_tmdb_year_agreement = "INCOMPLETE"

            if media_type != "movie":

                assessment = (
                    "NON_MOVIE_MEDIA_TYPE_REVIEW"
                )

            elif year_result == "YEAR_CONFLICT":

                assessment = "YEAR_CONFLICT_REVIEW"

            elif (
                title_result ==
                "EXACT_NORMALIZED_TITLE_AGREEMENT"
                and year_result == "YEAR_AGREEMENT"
            ):

                assessment = (
                    "STRONG_TMDb_TITLE_YEAR_CORROBORATION"
                )

            elif (
                title_result ==
                "EXACT_NORMALIZED_TITLE_AGREEMENT"
            ):

                assessment = (
                    "TITLE_AGREEMENT_YEAR_INCOMPLETE"
                )

            elif title_result == "TITLE_VARIATION_REVIEW":

                assessment = "TITLE_VARIATION_REVIEW"

            else:

                assessment = "INSUFFICIENT_EVIDENCE"

            output.append({

                "movie_entity_id": entity,

                "source_uri": row["source_uri"],

                "dbpedia_imdb_id": imdb_id,

                "tmdb_id": tmdb_id,

                "tmdb_title": tmdb_title,

                "tmdb_original_title": original_title,

                "tmdb_media_type": media_type,

                "tmdb_release_date": release_date,

                "tmdb_year": tmdb_year or "",

                "tmdb_request_status": row.get(
                    "request_status", ""
                ),

                "tmdb_http_status": row.get(
                    "http_status", ""
                ),

                "tmdb_result_count_reported": row.get(
                    "tmdb_result_count", ""
                ),

                "dbpedia_english_labels": json.dumps(
                    source["english_labels"],
                    ensure_ascii=False
                ),

                "dbpedia_years": json.dumps(
                    source["source_years"]
                ),

                "title_comparison": title_result,

                "year_comparison": year_result,

                "imdb_title_comparison":
                    imdb_title_result,

                "imdb_tmdb_year_comparison":
                    imdb_tmdb_year_agreement,

                "imdb_evidence": imdb_info.get(
                    "imdb_evidence", ""
                ),

                "movielens_shared_tmdb_id":
                    tmdb_id in shared_ids,

                "record_evidence_assessment":
                    assessment,

                "automatic_identity_verification":
                    False,

                "automatic_entity_merging":
                    False,
            })

        if index % 100 == 0:
            log(
                f"Processed {index:,}/{total:,}"
            )

    return pd.DataFrame(output)


# ============================================================
# ENTITY-LEVEL EVIDENCE
# ============================================================

def build_entity_master(master, records):

    log("\n[4] ENTITY-LEVEL EVIDENCE")

    output = []

    grouped = {
        entity: group
        for entity, group in records.groupby(
            "movie_entity_id"
        )
    }

    for _, row in master.iterrows():

        entity = str(row["movie_entity_id"])

        group = grouped.get(entity)

        if group is None or group.empty:
            group = pd.DataFrame(
                columns=records.columns
            )

        candidate_group = group[
            group["tmdb_id"].astype(str).ne("")
        ]

        assessments = candidate_group[
            "record_evidence_assessment"
        ].tolist()

        no_result_count = int(
            group[
                "record_evidence_assessment"
            ].eq("NO_TMDB_RESULT").sum()
        )

        strong = sum(
            value ==
            "STRONG_TMDb_TITLE_YEAR_CORROBORATION"
            for value in assessments
        )

        conflicts = sum(
            value == "YEAR_CONFLICT_REVIEW"
            for value in assessments
        )

        non_movie = sum(
            value == "NON_MOVIE_MEDIA_TYPE_REVIEW"
            for value in assessments
        )

        variations = sum(
            value == "TITLE_VARIATION_REVIEW"
            for value in assessments
        )

        title_year_incomplete = sum(
            value == "TITLE_AGREEMENT_YEAR_INCOMPLETE"
            for value in assessments
        )

        insufficient = sum(
            value == "INSUFFICIENT_EVIDENCE"
            for value in assessments
        )

        if not assessments:

            status = "UNRESOLVED_NO_TMDb_RESULT"

        elif conflicts:

            status = "YEAR_CONFLICT_REQUIRES_REVIEW"

        elif non_movie:

            status = "MEDIA_TYPE_REQUIRES_REVIEW"

        elif strong:

            status = (
                "TMDb_STRONG_CORROBORATION_CANDIDATE"
            )

        elif variations:

            status = "TITLE_VARIATION_REQUIRES_REVIEW"

        else:

            status = "UNRESOLVED_INSUFFICIENT_EVIDENCE"

        output.append({

            "movie_entity_id": entity,

            "source_uri": row["source_uri"],

            "tmdb_identifier_pair_count":
                int(
                    group[
                        ["movie_entity_id",
                         "dbpedia_imdb_id"]
                    ].drop_duplicates().shape[0]
                ) if not group.empty else 0,

            "tmdb_candidate_record_count":
                int(len(candidate_group)),

            "tmdb_no_result_identifier_count":
                no_result_count,

            "strong_candidate_count":
                int(strong),

            "year_conflict_count":
                int(conflicts),

            "non_movie_result_count":
                int(non_movie),

            "title_variation_count":
                int(variations),

            "title_year_incomplete_count":
                int(title_year_incomplete),

            "insufficient_evidence_count":
                int(insufficient),

            "entity_evidence_assessment":
                status,

            "identity_verified": False,

            "automatic_entity_merging": False,
        })

    return pd.DataFrame(output)


# ============================================================
# EXCEPTION REGISTER AND OUTPUTS
# ============================================================

def build_exceptions(records):

    if records.empty:
        return records.copy()

    return records[
        records["record_evidence_assessment"]
        != "STRONG_TMDb_TITLE_YEAR_CORROBORATION"
    ].copy()


def distribution(series):

    return {
        str(key): int(value)
        for key, value in
        series.value_counts(dropna=False).items()
    }


def write_outputs(
    sources,
    checks,
    records,
    entities
):

    log("\n[6] EXPORTING RESULTS")

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    exceptions = build_exceptions(records)

    outputs = {

        "tmdb_evidence_quality_master.csv":
            entities,

        "tmdb_evidence_record_level.csv":
            records,

        "tmdb_evidence_exception_register.csv":
            exceptions,

        "tmdb_conflict_review.csv":
            sources["tmdb_conflicts"],
    }

    for filename, dataframe in outputs.items():

        dataframe.to_csv(
            OUT / filename,
            index=False,
            encoding="utf-8-sig"
        )

        log(
            f"{filename}: {len(dataframe):,} records, "
            f"{(OUT / filename).stat().st_size:,} bytes"
        )

    manifest = {

        "experiment": "21.17H",

        "name":
            "TMDb Evidence Quality and Identity Verification",

        "generated_utc":
            datetime.now(timezone.utc).isoformat(),

        "source_hashes": {
            name: sha256(path)
            for name, path in SOURCES.items()
        },

        "counts": {

            "source_entities":
                int(len(sources["tmdb_master"])),

            "tmdb_identifier_pairs":
                int(len(sources["tmdb_crossref"])),

            "record_level_rows_including_no_result_placeholders":
                int(len(records)),

            "candidate_result_rows":
                int(
                    records["tmdb_id"]
                    .astype(str).ne("").sum()
                ),

            "no_result_identifier_rows":
                int(
                    records[
                        "record_evidence_assessment"
                    ].eq("NO_TMDB_RESULT").sum()
                ),

            "exception_records":
                int(len(exceptions)),

            "entity_master_rows":
                int(len(entities)),

            "conflict_register_rows":
                int(len(sources["tmdb_conflicts"])),
        },

        "record_assessment_distribution":
            distribution(
                records["record_evidence_assessment"]
            ),

        "entity_assessment_distribution":
            distribution(
                entities["entity_evidence_assessment"]
            ),

        "validation": {
            key: bool(value)
            for key, value in checks.items()
        },

        "methodology": {

            "identifier_unit":
                "One DBpedia entity / IMDb identifier pair; "
                "all TMDb candidate rows retained",

            "no_result_handling":
                "Explicit placeholder row retained for every "
                "identifier pair without returned TMDb candidates",

            "title_comparison":
                "Unicode NFKC and case-folded normalized exact "
                "equality after removal of parenthetical/bracketed qualifiers",

            "year_comparison":
                "Available English DBpedia labels, DBpedia release dates, "
                "DBpedia URI year, and TMDb release year",

            "media_type_assessment":
                "Explicit TMDb media-type evaluation; "
                "non-movie records flagged for review",

            "imdb_role":
                "Supporting title/year evidence from prior official "
                "IMDb cross-validation",

            "movielens_role":
                "Supporting identifier corroboration when TMDb IDs are shared",

            "automatic_identity_verification": False,

            "automatic_entity_merging": False,
        },

        "limitations": [

            "Title and year agreement are corroborating evidence, "
            "not absolute proof of identity.",

            "DBpedia metadata may be incomplete or contain aliases "
            "for related works.",

            "TMDb search results may contain unrelated candidates "
            "or multiple media types.",

            "IMDb-derived TMDb lookups are not fully independent "
            "identifier chains.",

            "MovieLens identifier agreement is supporting evidence "
            "and does not independently prove identity.",

            "No automatic identity verification or entity merging "
            "was performed.",
        ],

        "completion_status":
            "COMPLETED_ALL_VALIDATIONS_PASSED",
    }

    manifest_path = (
        OUT / "experiment_21_17H_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=4,
            ensure_ascii=False,
            default=json_safe
        ),
        encoding="utf-8"
    )

    report = [

        "EXPERIMENT 21.17H",

        "TMDb Evidence Quality and Identity Verification",

        "",

        f"DBpedia entities preserved: {len(entities)}",

        f"IMDb/TMDb identifier pairs: "
        f"{len(sources['tmdb_crossref'])}",

        f"Record-level rows including no-result placeholders: "
        f"{len(records)}",

        f"TMDb candidate rows: "
        f"{manifest['counts']['candidate_result_rows']}",

        f"No-result identifier rows: "
        f"{manifest['counts']['no_result_identifier_rows']}",

        "",

        "Record assessments:",

        json.dumps(
            manifest["record_assessment_distribution"],
            indent=4,
            ensure_ascii=False
        ),

        "",

        "Entity assessments:",

        json.dumps(
            manifest["entity_assessment_distribution"],
            indent=4,
            ensure_ascii=False
        ),

        "",

        "Validation:",

        json.dumps(
            manifest["validation"],
            indent=4,
            ensure_ascii=False
        ),

        "",

        "Automatic identity verification: DISABLED",

        "Automatic entity merging: DISABLED",

        "Original source files modified: NO",

        "Knowledge graph modified: NO",
    ]

    (
        OUT / "reproducibility_report.txt"
    ).write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    log(f"Manifest: {manifest_path}")

    log("Reproducibility report generated.")


# ============================================================
# MAIN
# ============================================================

def main():

    log("=" * 75)

    log(
        "EXPERIMENT 21.17H: TMDb EVIDENCE QUALITY AUDIT"
    )

    log("=" * 75)

    try:

        sources = load_sources()

        checks = validate_sources(sources)

        dbpedia = prepare_dbpedia(
            sources["dbpedia"]
        )

        imdb = prepare_imdb(
            sources["imdb_records"]
        )

        records = build_record_evidence(
            sources,
            dbpedia,
            imdb
        )

        entities = build_entity_master(
            sources["tmdb_master"],
            records
        )

        log("\n[5] FINAL VALIDATION")

        source_pairs = {
            pair_key(row)
            for _, row in
            sources["tmdb_crossref"].iterrows()
        }

        output_pairs = set(
            zip(
                records["movie_entity_id"].astype(str),
                records["dbpedia_imdb_id"].astype(str)
            )
        )

        entity_ids = set(
            sources["tmdb_master"][
                "movie_entity_id"
            ].astype(str)
        )

        output_entity_ids = set(
            entities["movie_entity_id"].astype(str)
        )

        checks.update({

            "entity_population_6112":
                len(entities) == EXPECTED_ENTITIES,

            "entity_ids_unique":
                entities["movie_entity_id"].is_unique,

            "all_source_entities_preserved":
                output_entity_ids == entity_ids,

            "identifier_pair_population_preserved":
                output_pairs == source_pairs,

            "all_578_identifier_pairs_accounted_for":
                len(output_pairs) == EXPECTED_IDENTIFIER_PAIRS,

            "record_rows_cover_identifier_pairs":
                len(records) >= EXPECTED_IDENTIFIER_PAIRS,

            "record_tmdb_ids_nonblank_only_for_candidates":
                records.loc[
                    records[
                        "record_evidence_assessment"
                    ] == "NO_TMDB_RESULT",
                    "tmdb_id"
                ].astype(str).eq("").all(),

            "no_automatic_identity_verification":
                not entities[
                    "identity_verified"
                ].astype(bool).any(),

            "no_automatic_entity_merging":
                not entities[
                    "automatic_entity_merging"
                ].astype(bool).any(),

            "conflict_register_population_4":
                len(sources["tmdb_conflicts"])
                == EXPECTED_CONFLICTS,
        })

        for name, passed in checks.items():

            log(
                f"{name}: "
                f"{'PASS' if passed else 'FAIL'}"
            )

        if not all(checks.values()):

            failed = [
                name
                for name, passed in checks.items()
                if not passed
            ]

            raise ValueError(
                f"Final validation failed: {failed}"
            )

        write_outputs(
            sources,
            checks,
            records,
            entities
        )

        log("\n" + "=" * 75)

        log("EXPERIMENT 21.17H COMPLETED")

        log("=" * 75)

        log(
            f"DBpedia entities preserved: "
            f"{len(entities):,}"
        )

        log(
            f"IMDb/TMDb identifier pairs accounted for: "
            f"{len(output_pairs):,}"
        )

        log(
            f"Record-level rows including no-result placeholders: "
            f"{len(records):,}"
        )

        log(
            f"TMDb candidate rows: "
            f"{int(records['tmdb_id'].astype(str).ne('').sum()):,}"
        )

        log(f"Output directory: {OUT}")

        log(
            "No automatic identity verification "
            "or entity merging."
        )

    except Exception as exc:

        log("\nEXPERIMENT 21.17H FAILED")

        log(
            f"{type(exc).__name__}: {exc}"
        )

        log(
            "Do not interpret partial outputs "
            "as completed results."
        )

        sys.exit(1)


if __name__ == "__main__":
    main()