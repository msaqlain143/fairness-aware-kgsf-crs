
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import sys
import unicodedata

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# EXPERIMENT 21.17E
# Official IMDb Identifier Cross-Validation
# ============================================================

ROOT = Path(__file__).resolve().parent

SOURCE_C = (
    ROOT / "results_category_audit" / "experiment_21_17C"
    / "imdb_identifier_cross_reference.csv"
)

SOURCE_D = (
    ROOT / "results_category_audit" / "experiment_21_17D"
    / "imdb_identity_verification_master.csv"
)

IMDB_FILE = (
    ROOT / "external_validation" / "imdb"
    / "title.basics.tsv.gz"
)

OUT = (
    ROOT / "results_category_audit"
    / "experiment_21_17E"
)

EXPECTED_URI_POPULATION = 6112


# ============================================================
# UTILITIES
# ============================================================

def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty output: {path}")

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
            extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_list(value) -> list:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    text = str(value).strip()

    if not text or text.lower() in {"nan", "none", "null"}:
        return []

    try:
        result = json.loads(text)
        return result if isinstance(result, list) else [result]
    except (json.JSONDecodeError, TypeError):
        return [text]


def normalize_imdb_id(value):
    """Normalize numeric IMDb IDs without truncating long IDs."""

    if value is None:
        return None

    value = str(value).strip()

    if value.startswith("tt"):
        value = value[2:]

    if not value.isdigit():
        return None

    return "tt" + value.zfill(7)


def normalize_title(value):
    """Conservative Unicode-aware title normalization."""

    if not value:
        return ""

    value = unicodedata.normalize("NFKC", str(value))
    value = value.casefold()

    value = re.sub(r"[^\w\s]", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def valid_year(value):
    value = str(value).strip()

    if not value.isdigit():
        return None

    year = int(value)

    if 1800 <= year <= 2100:
        return year

    return None


def extract_years(values):
    years = set()

    for value in values:
        for match in re.findall(
            r"(?<!\d)(18\d{2}|19\d{2}|20\d{2}|2100)(?!\d)",
            str(value)
        ):
            year = valid_year(match)

            if year:
                years.add(year)

    return sorted(years)


def clean_value(value):
    if value in (None, "", r"\N"):
        return ""

    return str(value).strip()


def json_value(value):
    return json.dumps(value, ensure_ascii=False)


# ============================================================
# LOAD SOURCE RECORDS
# ============================================================

def load_sources():

    for path in [SOURCE_C, SOURCE_D, IMDB_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required source missing: {path}")

    source_c = read_csv(SOURCE_C)
    source_d = read_csv(SOURCE_D)

    if len(source_c) != EXPECTED_URI_POPULATION:
        raise RuntimeError(
            f"Expected {EXPECTED_URI_POPULATION} URI records, "
            f"found {len(source_c)}"
        )

    if len(source_d) != EXPECTED_URI_POPULATION:
        raise RuntimeError(
            f"Expected {EXPECTED_URI_POPULATION} Experiment 21.17D "
            f"records, found {len(source_d)}"
        )

    required_c = {
        "movie_entity_id",
        "source_uri",
        "dbpedia_imdb_ids",
        "english_labels",
        "matching_movie_titles",
    }

    required_d = {
        "movie_entity_id",
        "dbpedia_release_years",
        "movielens_release_years",
        "identity_assessment",
    }

    if not required_c.issubset(source_c[0].keys()):
        raise RuntimeError("Unexpected Experiment 21.17C schema.")

    if not required_d.issubset(source_d[0].keys()):
        raise RuntimeError("Unexpected Experiment 21.17D schema.")

    c_ids = [r["movie_entity_id"] for r in source_c]
    d_ids = [r["movie_entity_id"] for r in source_d]

    if len(set(c_ids)) != EXPECTED_URI_POPULATION:
        raise RuntimeError("Duplicate entity IDs in Experiment 21.17C.")

    if len(set(d_ids)) != EXPECTED_URI_POPULATION:
        raise RuntimeError("Duplicate entity IDs in Experiment 21.17D.")

    d_index = {r["movie_entity_id"]: r for r in source_d}

    if set(c_ids) != set(d_ids):
        raise RuntimeError(
            "Experiment 21.17C and 21.17D entity populations differ."
        )

    return source_c, d_index


# ============================================================
# PREPARE TARGET IMDb IDENTIFIERS
# ============================================================

def prepare_targets(source):

    targets = defaultdict(list)
    master = []

    for record in source:

        entity_id = record["movie_entity_id"]

        raw_ids = parse_list(record["dbpedia_imdb_ids"])

        normalized_ids = []

        for raw_id in raw_ids:
            normalized = normalize_imdb_id(raw_id)

            if normalized:
                normalized_ids.append(normalized)

        normalized_ids = sorted(set(normalized_ids))

        targets_for_entity = normalized_ids

        for imdb_id in targets_for_entity:
            targets[imdb_id].append(entity_id)

        master.append({
            "movie_entity_id": entity_id,
            "source_uri": record["source_uri"],
            "dbpedia_imdb_ids": json_value(raw_ids),
            "normalized_imdb_ids": json_value(normalized_ids),
            "english_labels": json_value(
                parse_list(record["english_labels"])
            ),
            "matching_movie_titles": json_value(
                parse_list(record["matching_movie_titles"])
            ),
        })

    return targets, master


# ============================================================
# STREAM OFFICIAL IMDb DATASET
# ============================================================

def retrieve_official_records(target_ids):

    official = {}
    duplicate_tconsts = set()

    total_scanned = 0
    matched_rows = 0

    print("\n[3] STREAMING OFFICIAL IMDb DATASET")

    with gzip.open(
        IMDB_FILE,
        "rt",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f, delimiter="\t")

        required = {
            "tconst",
            "titleType",
            "primaryTitle",
            "originalTitle",
            "startYear",
            "genres",
        }

        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError("Unexpected official IMDb schema.")

        for row in reader:

            total_scanned += 1

            tconst = clean_value(row["tconst"])

            if tconst not in target_ids:
                continue

            matched_rows += 1

            if tconst in official:
                duplicate_tconsts.add(tconst)
                continue

            official[tconst] = {
                "tconst": tconst,
                "titleType": clean_value(row["titleType"]),
                "primaryTitle": clean_value(row["primaryTitle"]),
                "originalTitle": clean_value(row["originalTitle"]),
                "startYear": clean_value(row["startYear"]),
                "genres": clean_value(row["genres"]),
            }

            if total_scanned % 1_000_000 == 0:
                print(
                    f"Scanned {total_scanned:,} records | "
                    f"Matched {len(official):,}"
                )

    print("Total scanned:", f"{total_scanned:,}")
    print("Matched official records:", len(official))
    print("Duplicate tconst records:", len(duplicate_tconsts))

    if duplicate_tconsts:
        raise RuntimeError(
            "Duplicate official IMDb identifiers detected."
        )

    return official, total_scanned, matched_rows


# ============================================================
# IDENTITY EVIDENCE ASSESSMENT
# ============================================================

def assess_entity(record, previous, official):

    imdb_ids = parse_list(record["normalized_imdb_ids"])
    labels = parse_list(record["english_labels"])
    movielens_titles = parse_list(
        record["matching_movie_titles"]
    )

    dbpedia_years = {
        valid_year(x)
        for x in extract_years(labels)
    }

    dbpedia_years.update(
        valid_year(x)
        for x in parse_list(previous["dbpedia_release_years"])
    )

    dbpedia_years.discard(None)

    movielens_years = {
        valid_year(x)
        for x in parse_list(previous["movielens_release_years"])
    }

    movielens_years.discard(None)

    official_records = [
        official[x] for x in imdb_ids if x in official
    ]

    identifier_status = (
        "NO_IMDB_IDENTIFIER" if not imdb_ids
        else "ALL_IDENTIFIERS_FOUND"
        if len(official_records) == len(imdb_ids)
        else "PARTIAL_IDENTIFIER_MATCH"
        if official_records
        else "NO_OFFICIAL_IMDB_MATCH"
    )

    comparisons = []

    for item in official_records:

        imdb_title = item["primaryTitle"]

        imdb_norm = normalize_title(imdb_title)

        label_matches = [
            str(x) for x in labels
            if normalize_title(x) == imdb_norm
        ]

        movielens_matches = [
            str(x) for x in movielens_titles
            if normalize_title(x) == imdb_norm
        ]

        imdb_year = valid_year(item["startYear"])

        year_sources = dbpedia_years | movielens_years

        if imdb_year is None or not year_sources:
            year_status = "YEAR_INCOMPLETE"
        elif imdb_year in year_sources:
            year_status = "YEAR_AGREEMENT"
        else:
            year_status = "YEAR_CONFLICT"

        if label_matches:
            title_status = "EXACT_NORMALIZED_LABEL_MATCH"
        elif movielens_matches:
            title_status = "EXACT_NORMALIZED_MOVIELENS_MATCH"
        elif labels or movielens_titles:
            title_status = "TITLE_VARIATION_REVIEW"
        else:
            title_status = "TITLE_EVIDENCE_UNAVAILABLE"

        comparisons.append({
            "imdb_id": item["tconst"],
            "imdb_title_type": item["titleType"],
            "imdb_primary_title": imdb_title,
            "imdb_original_title": item["originalTitle"],
            "imdb_start_year": item["startYear"],
            "imdb_genres": item["genres"],
            "title_comparison": title_status,
            "year_comparison": year_status,
            "matching_dbpedia_labels": json_value(label_matches),
            "matching_movielens_titles": json_value(
                movielens_matches
            ),
        })

    if not imdb_ids:
        assessment = "UNRESOLVED_NO_IMDB_IDENTIFIER"

    elif not official_records:
        assessment = "UNRESOLVED_NO_OFFICIAL_IMDB_MATCH"

    elif len(official_records) != len(imdb_ids):
        assessment = "UNRESOLVED_PARTIAL_IDENTIFIER_MATCH"

    elif len(imdb_ids) > 1:
        assessment = "IDENTIFIER_CARDINALITY_REVIEW"

    elif any(
        x["year_comparison"] == "YEAR_CONFLICT"
        for x in comparisons
    ):
        assessment = "YEAR_CONFLICT_REQUIRES_REVIEW"

    elif all(
        x["title_comparison"] == "EXACT_NORMALIZED_LABEL_MATCH"
        for x in comparisons
    ) and all(
        x["year_comparison"] == "YEAR_AGREEMENT"
        for x in comparisons
    ):
        assessment = "STRONG_IDENTIFIER_TITLE_YEAR_CORROBORATION"

    elif any(
        x["title_comparison"] in {
            "EXACT_NORMALIZED_LABEL_MATCH",
            "EXACT_NORMALIZED_MOVIELENS_MATCH",
        }
        for x in comparisons
    ):
        assessment = "IDENTIFIER_TITLE_CORROBORATION_REVIEW"

    else:
        assessment = "IDENTIFIER_TITLE_VARIATION_REVIEW"

    return {
        "official_identifier_status": identifier_status,
        "official_record_count": len(official_records),
        "official_imdb_records": json_value(comparisons),
        "dbpedia_release_years": json_value(sorted(dbpedia_years)),
        "movielens_release_years": json_value(sorted(movielens_years)),
        "identity_assessment": assessment,
        "identity_verified": False,
        "automatic_merging": False,
    }, comparisons


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 75)
    print("EXPERIMENT 21.17E: OFFICIAL IMDb IDENTIFIER CROSS-VALIDATION")
    print("=" * 75)

    OUT.mkdir(parents=True, exist_ok=True)

    print("\n[1] SOURCE INVENTORY")

    source, previous_index = load_sources()

    print("DBpedia identities:", len(source))
    print("IMDb compressed dataset:", IMDB_FILE.stat().st_size)
    print("Previous Experiment 21.17D:", len(previous_index))

    print("\n[2] TARGET IDENTIFIERS")

    targets, master = prepare_targets(source)

    target_ids = set(targets)

    print("Unique target IMDb identifiers:", len(target_ids))
    print(
        "Identities with IMDb identifiers:",
        sum(bool(parse_list(x["normalized_imdb_ids"])) for x in master)
    )
    print(
        "Identities without IMDb identifiers:",
        sum(not parse_list(x["normalized_imdb_ids"]) for x in master)
    )

    print("\n[3] OFFICIAL DATASET RETRIEVAL")

    official, scanned, matched_rows = retrieve_official_records(
        target_ids
    )

    print("\n[4] RECORD-LEVEL ASSESSMENT")

    master_output = []
    comparisons_output = []
    matched_output = []
    unmatched_output = []

    distribution = Counter()

    for record in master:

        entity_id = record["movie_entity_id"]

        previous = previous_index[entity_id]

        evidence, comparisons = assess_entity(
            record,
            previous,
            official
        )

        result = {
            **record,
            **evidence,
            "previous_identity_assessment":
                previous["identity_assessment"],
        }

        master_output.append(result)

        distribution[evidence["identity_assessment"]] += 1

        for comparison in comparisons:

            row = {
                "movie_entity_id": entity_id,
                "source_uri": record["source_uri"],
                **comparison,
            }

            comparisons_output.append(row)
            matched_output.append(row)

        if not comparisons:
            unmatched_output.append({
                "movie_entity_id": entity_id,
                "source_uri": record["source_uri"],
                "normalized_imdb_ids":
                    record["normalized_imdb_ids"],
                "official_identifier_status":
                    evidence["official_identifier_status"],
                "identity_assessment":
                    evidence["identity_assessment"],
            })

    print("\nIDENTITY ASSESSMENT DISTRIBUTION")

    for name, count in sorted(distribution.items()):
        print(f"{name}: {count:,}")

    print("\n[5] VALIDATION")

    checks = {
        "master_population_6112":
            len(master_output) == EXPECTED_URI_POPULATION,

        "master_entity_ids_unique":
            len({
                x["movie_entity_id"] for x in master_output
            }) == EXPECTED_URI_POPULATION,

        "all_source_uris_preserved":
            all(x["source_uri"] for x in master_output),

        "official_identifiers_match_target_set":
            set(official).issubset(target_ids),

        "all_comparisons_have_official_records":
            all(
                x["imdb_id"] in official
                for x in comparisons_output
            ),

        "no_automatic_identity_verification":
            all(
                x["identity_verified"] is False
                for x in master_output
            ),

        "no_automatic_merging":
            all(
                x["automatic_merging"] is False
                for x in master_output
            ),

        "all_entities_accounted_for":
            len(master_output) == len(source),
    }

    for name, status in checks.items():
        print(f"{name}: {'PASS' if status else 'FAIL'}")

    if not all(checks.values()):
        raise RuntimeError("Experiment validation failed.")

    print("\n[6] EXPORTING RESULTS")

    write_csv(
        OUT / "imdb_identity_evidence_master.csv",
        master_output
    )

    write_csv(
        OUT / "imdb_official_identifier_matches.csv",
        matched_output
    )

    write_csv(
        OUT / "imdb_title_year_comparison.csv",
        comparisons_output
    )

    if unmatched_output:
        write_csv(
            OUT / "imdb_unmatched_identifiers.csv",
            unmatched_output
        )

    manifest = {
        "experiment": "21.17E",
        "name": "Official IMDb Identifier Cross-Validation",
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "sources": {
            "experiment_21_17C": str(SOURCE_C.relative_to(ROOT)),
            "experiment_21_17D": str(SOURCE_D.relative_to(ROOT)),
            "official_imdb_dataset": str(IMDB_FILE.relative_to(ROOT)),
        },

        "source_hashes": {
            "experiment_21_17C": sha256(SOURCE_C),
            "experiment_21_17D": sha256(SOURCE_D),
            "official_imdb_dataset": sha256(IMDB_FILE),
        },

        "counts": {
            "dbpedia_identities": len(source),
            "unique_target_imdb_ids": len(target_ids),
            "official_imdb_records_matched": len(official),
            "identifier_comparisons": len(comparisons_output),
            "unmatched_entity_records": len(unmatched_output),
            "official_dataset_records_scanned": scanned,
            "official_dataset_matching_rows": matched_rows,
        },

        "assessment_distribution": dict(distribution),

        "validation": checks,

        "methodology": {
            "identifier_matching": "Exact normalized IMDb tconst",
            "title_matching": "Conservative Unicode-normalized equality",
            "year_matching": "Comparison with available DBpedia and MovieLens years",
            "title_type_preserved": True,
            "identity_verification_automatic": False,
            "automatic_entity_merging": False,
            "unresolved_records_preserved": True,
        },

        "limitations": [
            "Official IMDb identifier existence does not independently prove DBpedia identity.",
            "Title agreement is corroborating evidence, not standalone identity proof.",
            "Missing years remain incomplete rather than being assumed to agree.",
            "IMDb dataset coverage and update date may affect available records.",
            "Film series and individual film granularity require separate investigation.",
            "No automatic identity verification or merging was performed.",
        ],

        "completion_status": "COMPLETED_ALL_VALIDATIONS_PASSED",

        "outputs": [
            "imdb_identity_evidence_master.csv",
            "imdb_official_identifier_matches.csv",
            "imdb_title_year_comparison.csv",
            "imdb_unmatched_identifiers.csv",
            "experiment_21_17E_manifest.json",
            "reproducibility_report.txt",
        ],
    }

    manifest_path = OUT / "experiment_21_17E_manifest.json"

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=4,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    report = [
        "EXPERIMENT 21.17E REPRODUCIBILITY REPORT",
        "=" * 60,
        f"Generated UTC: {manifest['generated_utc']}",
        f"DBpedia identities: {len(source)}",
        f"Unique target IMDb IDs: {len(target_ids)}",
        f"Official IMDb matches: {len(official)}",
        f"Comparison records: {len(comparisons_output)}",
        "",
        "ASSESSMENT DISTRIBUTION",
        "-" * 40,
    ]

    report.extend(
        f"{k}: {v}" for k, v in sorted(distribution.items())
    )

    report.extend([
        "",
        "VALIDATION",
        "-" * 40,
    ])

    report.extend(
        f"{k}: {'PASS' if v else 'FAIL'}"
        for k, v in checks.items()
    )

    report.extend([
        "",
        "No automatic identity verification.",
        "No automatic entity merging.",
        "Original source files unchanged.",
    ])

    (OUT / "reproducibility_report.txt").write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("\n" + "=" * 75)
    print("EXPERIMENT 21.17E COMPLETED")
    print("=" * 75)

    print("DBpedia identities:", len(master_output))
    print("Official IMDb matches:", len(official))
    print("Comparison records:", len(comparisons_output))
    print("Output:", OUT)

    print("\nOriginal source files unchanged.")
    print("No graph modification.")
    print("No model training or inference.")
    print("No automatic entity merging.")

    print("\nEXPERIMENT 21.17E FINISHED.")


if __name__ == "__main__":

    try:
        main()

    except Exception as exc:
        print("\nEXPERIMENT 21.17E FAILED")
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)
