

from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd


# ============================================================
# EXPERIMENT 21.17C
# IMDb IDENTIFIER CROSS-REFERENCE VALIDATION
# ============================================================

ROOT = Path(__file__).resolve().parent

DBPEDIA_FILE = (
    ROOT / "results_category_audit" / "experiment_21_17B"
    / "dbpedia_independent_metadata_evidence.csv"
)

LINKS_FILE = (
    ROOT / "external_validation" / "movielens_32m" / "links.csv"
)

MOVIES_FILE = (
    ROOT / "external_validation" / "movielens_32m" / "movies.csv"
)

OUTPUT_DIR = (
    ROOT / "results_category_audit" / "experiment_21_17C"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def normalize_imdb(value):
    """Normalize IMDb numeric identifiers without losing leading zeros."""

    if value is None:
        return ""

    value = str(value).strip()

    if not value or value.lower() in {"nan", "none", "null"}:
        return ""

    value = value.lower()

    value = re.sub(r"^tt", "", value)

    if not value.isdigit():
        return ""

    # MovieLens stores IMDb identifiers numerically.
    # Seven digits is a minimum width, not a fixed width.
    return value.zfill(7)


def parse_json_values(value):
    """Safely extract values from JSON-array fields."""

    if value is None:
        return []

    if isinstance(value, list):
        return value

    value = str(value).strip()

    if not value or value.lower() in {"nan", "none", "null"}:
        return []

    try:
        parsed = json.loads(value)

        if isinstance(parsed, list):
            return parsed

        if isinstance(parsed, dict):
            return [parsed]

        return []

    except (json.JSONDecodeError, TypeError):
        return []


def extract_imdb_ids(value):
    """Extract distinct normalized IMDb IDs."""

    items = parse_json_values(value)
    identifiers = set()

    for item in items:

        if isinstance(item, dict):
            raw = item.get("value", "")
        else:
            raw = item

        identifier = normalize_imdb(raw)

        if identifier:
            identifiers.add(identifier)

    return sorted(identifiers)


def extract_english_labels(value):
    """Extract English-language DBpedia labels."""

    labels = []

    for item in parse_json_values(value):

        if not isinstance(item, dict):
            continue

        language = str(item.get("language", "")).lower()

        if language == "en":
            label = str(item.get("value", "")).strip()

            if label:
                labels.append(label)

    return sorted(set(labels))


def normalized_title(value):
    """Conservative title normalization for supporting comparisons."""

    value = str(value or "").lower()

    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)

    return " ".join(value.split())


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# LOAD SOURCES
# ============================================================

def main():

    print("=" * 75)
    print("EXPERIMENT 21.17C: IMDb IDENTIFIER CROSS-REFERENCE")
    print("=" * 75)

    for path in [DBPEDIA_FILE, LINKS_FILE, MOVIES_FILE]:

        if not path.exists():
            raise FileNotFoundError(f"Required source missing: {path}")

    dbpedia = pd.read_csv(DBPEDIA_FILE, dtype=str).fillna("")
    links = pd.read_csv(LINKS_FILE, dtype=str).fillna("")
    movies = pd.read_csv(MOVIES_FILE, dtype=str).fillna("")

    print("\n[1] SOURCE INVENTORY")

    print("DBpedia records:", len(dbpedia))
    print("MovieLens links:", len(links))
    print("MovieLens movies:", len(movies))

    required_dbpedia = {
        "movie_entity_id",
        "source_uri",
        "imdb_ids",
        "labels"
    }

    required_links = {"movieId", "imdbId"}
    required_movies = {"movieId", "title", "genres"}

    if not required_dbpedia.issubset(dbpedia.columns):
        raise ValueError(
            f"DBpedia columns missing: "
            f"{required_dbpedia - set(dbpedia.columns)}"
        )

    if not required_links.issubset(links.columns):
        raise ValueError("MovieLens links columns are incomplete.")

    if not required_movies.issubset(movies.columns):
        raise ValueError("MovieLens movies columns are incomplete.")

    # ========================================================
    # SOURCE INTEGRITY
    # ========================================================

    print("\n[2] SOURCE INTEGRITY")

    checks = {}

    checks["dbpedia_population_6112"] = len(dbpedia) == 6112

    checks["dbpedia_entity_ids_unique"] = (
        dbpedia["movie_entity_id"].is_unique
    )

    checks["dbpedia_uris_unique"] = (
        dbpedia["source_uri"].is_unique
    )

    checks["movielens_movie_ids_unique"] = (
        links["movieId"].is_unique
    )

    checks["movielens_metadata_ids_unique"] = (
        movies["movieId"].is_unique
    )

    movie_lookup = movies.set_index("movieId").to_dict("index")

    # Build IMDb-to-MovieLens mapping.
    imdb_lookup = defaultdict(list)

    for _, row in links.iterrows():

        imdb_id = normalize_imdb(row["imdbId"])

        if not imdb_id:
            continue

        movie_id = str(row["movieId"]).strip()

        metadata = movie_lookup.get(movie_id, {})

        imdb_lookup[imdb_id].append({
            "movielens_id": movie_id,
            "imdb_id": imdb_id,
            "movie_title": metadata.get("title", ""),
            "genres": metadata.get("genres", "")
        })

    print("\nIMDb identifiers in MovieLens:", len(imdb_lookup))

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    if not all(checks.values()):
        raise RuntimeError(
            "Source integrity validation failed. "
            "No reconciliation will be exported."
        )

    # ========================================================
    # RECORD-LEVEL CROSS-REFERENCE
    # ========================================================

    print("\n[3] RECORD-LEVEL IMDb CROSS-REFERENCE")

    results = []

    for index, row in dbpedia.iterrows():

        identifiers = extract_imdb_ids(row["imdb_ids"])

        english_labels = extract_english_labels(row["labels"])

        matches = []

        for imdb_id in identifiers:
            matches.extend(imdb_lookup.get(imdb_id, []))

        # Deduplicate matching records.
        unique_matches = {
            (m["imdb_id"], m["movielens_id"]): m
            for m in matches
        }

        matches = list(unique_matches.values())

        matching_imdb_ids = sorted({
            m["imdb_id"] for m in matches
        })

        matching_movie_ids = sorted({
            m["movielens_id"] for m in matches
        })

        matching_titles = sorted({
            m["movie_title"]
            for m in matches
            if m["movie_title"]
        })

        if not identifiers:
            status = "NO_IMDb_IDENTIFIER"

        elif not matches:
            status = "IMDB_NOT_FOUND_IN_MOVIELENS"

        elif len(matching_imdb_ids) > 1 or len(matching_movie_ids) > 1:
            status = "MULTIPLE_CROSS_REFERENCE_CANDIDATES"

        elif len(matches) == 1:
            status = "UNIQUE_IMDB_CROSS_REFERENCE"

        else:
            status = "REVIEW_REQUIRED"

        label_norms = {
            normalized_title(label)
            for label in english_labels
        }

        movie_norms = {
            normalized_title(title)
            for title in matching_titles
        }

        if label_norms and movie_norms:
            title_agreement = bool(label_norms & movie_norms)
        else:
            title_agreement = None

        results.append({

            "movie_entity_id": row["movie_entity_id"],

            "source_uri": row["source_uri"],

            "dbpedia_imdb_ids": json.dumps(
                identifiers, ensure_ascii=False
            ),

            "english_labels": json.dumps(
                english_labels, ensure_ascii=False
            ),

            "matching_imdb_ids": json.dumps(
                matching_imdb_ids, ensure_ascii=False
            ),

            "matching_movielens_ids": json.dumps(
                matching_movie_ids, ensure_ascii=False
            ),

            "matching_movie_titles": json.dumps(
                matching_titles, ensure_ascii=False
            ),

            "candidate_count": len(matches),

            "identifier_status": status,

            "english_label_title_agreement": (
                title_agreement
                if title_agreement is not None
                else ""
            ),

            "identity_verified": False,

            "evidence_interpretation": (
                "Independent identifier cross-reference; "
                "requires identity-level assessment."
            )
        })

        if (index + 1) % 1000 == 0:
            print(f"Processed {index + 1:,}/{len(dbpedia):,}")

    evidence = pd.DataFrame(results)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n[4] CROSS-REFERENCE SUMMARY")

    status_counts = evidence["identifier_status"].value_counts()

    print(status_counts.to_string())

    records_with_ids = evidence[
        evidence["dbpedia_imdb_ids"] != "[]"
    ]

    records_with_matches = evidence[
        evidence["candidate_count"] > 0
    ]

    print("\nTotal URI identities:", len(evidence))
    print("Records with IMDb identifiers:", len(records_with_ids))
    print("Records with MovieLens matches:", len(records_with_matches))
    print(
        "Records without IMDb identifiers:",
        len(evidence) - len(records_with_ids)
    )

    # ========================================================
    # EXPORT
    # ========================================================

    print("\n[5] EXPORTING RESULTS")

    evidence_path = OUTPUT_DIR / "imdb_identifier_cross_reference.csv"

    unmatched_path = OUTPUT_DIR / "imdb_identifier_unmatched.csv"

    summary_path = OUTPUT_DIR / "imdb_match_summary.json"

    manifest_path = OUTPUT_DIR / "experiment_21_17C_manifest.json"

    evidence.to_csv(
        evidence_path,
        index=False,
        encoding="utf-8-sig"
    )

    unmatched = evidence[
        evidence["candidate_count"] == 0
    ]

    unmatched.to_csv(
        unmatched_path,
        index=False,
        encoding="utf-8-sig"
    )

    summary = {
        "experiment": "21.17C",
        "name": "IMDb Identifier Cross-Reference Validation",

        "counts": {
            "dbpedia_uri_population": len(dbpedia),
            "unique_dbpedia_entities": int(
                evidence["movie_entity_id"].nunique()
            ),
            "records_with_imdb_identifiers": len(records_with_ids),
            "records_with_movielens_matches": len(records_with_matches),
            "records_without_matches": len(unmatched),
            "unique_movielens_imdb_identifiers": len(imdb_lookup)
        },

        "status_distribution": {
            str(k): int(v)
            for k, v in status_counts.items()
        },

        "validation": {
            **checks,
            "output_population_preserved": len(evidence) == 6112,
            "output_entity_ids_unique": evidence[
                "movie_entity_id"
            ].is_unique,
            "all_source_uris_preserved": (
                set(evidence["source_uri"]) ==
                set(dbpedia["source_uri"])
            ),
            "no_automatic_identity_verification": (
                not evidence["identity_verified"].any()
            )
        },

        "methodology": {
            "identifier": "Normalized IMDb numeric identifier",
            "minimum_imdb_width": 7,
            "title_matching_is_identity_proof": False,
            "automatic_entity_merging": False,
            "automatic_identity_verification": False,
            "unresolved_records_preserved": True
        },

        "source_hashes": {
            "dbpedia_metadata": sha256_file(DBPEDIA_FILE),
            "movielens_links": sha256_file(LINKS_FILE),
            "movielens_movies": sha256_file(MOVIES_FILE)
        }
    }

    summary["completion_status"] = (
        "COMPLETED"
        if all(summary["validation"].values())
        else "VALIDATION_FAILED"
    )

    summary_path.write_text(
        json.dumps(summary, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    manifest = {
        "experiment": "21.17C",
        "purpose": (
            "Cross-reference DBpedia IMDb identifiers "
            "against MovieLens 32M."
        ),
        "outputs": [
            evidence_path.name,
            unmatched_path.name,
            summary_path.name,
            manifest_path.name
        ],
        "summary": summary
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    print("\n[6] FINAL VALIDATION")

    for name, passed in summary["validation"].items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    print("\n" + "=" * 75)
    print("EXPERIMENT 21.17C FINISHED")
    print("=" * 75)

    print("Total URI identities:", len(evidence))
    print("IMDb-bearing records:", len(records_with_ids))
    print("Cross-referenced records:", len(records_with_matches))
    print("Unmatched records:", len(unmatched))

    print("\nOutput directory:", OUTPUT_DIR)

    if summary["completion_status"] != "COMPLETED":
        raise RuntimeError("Final validation failed.")


if __name__ == "__main__":
    main()
