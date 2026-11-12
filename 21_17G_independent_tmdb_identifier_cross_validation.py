"""
Experiment 21.17G
Independent TMDb Identifier Cross-Validation

Purpose
-------
Independently query TMDb using IMDb identifiers explicitly present in
DBpedia metadata (Experiment 21.17C). Compare returned TMDb IDs with
MovieLens links for the same IMDb identifier.

Important methodological safeguards
-----------------------------------
1. Only dbpedia_imdb_ids are used as lookup identifiers.
2. matching_imdb_ids are NOT treated as independent DBpedia identifiers.
3. MovieLens is an intermediate comparison source, not independent proof.
4. All TMDb result categories are retained.
5. No automatic identity verification or entity merging is performed.
6. Requests are checkpointed and resumable.
7. API tokens are read from TMDB_ACCESS_TOKEN and never exported.

Run from project root:
    python -m py_compile .\\21_17G_independent_tmdb_identifier_cross_validation.py
    python .\\21_17G_independent_tmdb_identifier_cross_validation.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parent

SOURCE_CROSSREF = (
    ROOT / "results_category_audit" / "experiment_21_17C"
    / "imdb_identifier_cross_reference.csv"
)
SOURCE_MOVIELENS_LINKS = (
    ROOT / "external_validation" / "movielens_32m" / "links.csv"
)
SOURCE_MOVIELENS_MOVIES = (
    ROOT / "external_validation" / "movielens_32m" / "movies.csv"
)
SOURCE_IMDB_EVIDENCE = (
    ROOT / "results_category_audit" / "experiment_21_17E"
    / "imdb_identity_evidence_master.csv"
)

OUT = ROOT / "results_category_audit" / "experiment_21_17G"
CHECKPOINT = OUT / "tmdb_api_checkpoint.jsonl"

API_BASE = "https://api.themoviedb.org/3"
REQUEST_DELAY_SECONDS = 0.30
REQUEST_TIMEOUT_SECONDS = 40
MAX_ATTEMPTS = 5

RESULT_ARRAYS = (
    ("movie_results", "movie"),
    ("tv_results", "tv"),
    ("tv_season_results", "tv_season"),
    ("tv_episode_results", "tv_episode"),
)

OUT.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_json_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def normalize_imdb(value: Any) -> str | None:
    """Convert tt1234567 or 1234567 to a normalized title tconst."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "\\n"}:
        return None

    text = text.lower()
    match = re.search(r"(?:tt)?(\d{1,12})", text)
    if not match:
        return None

    digits = match.group(1)
    return "tt" + digits.zfill(7)


def normalize_tmdb(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy scalar values into JSON-safe Python values."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def load_sources():
    required = [
        SOURCE_CROSSREF,
        SOURCE_MOVIELENS_LINKS,
        SOURCE_MOVIELENS_MOVIES,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required source not found: {path}")

    cross = pd.read_csv(SOURCE_CROSSREF, dtype=str).fillna("")
    links = pd.read_csv(SOURCE_MOVIELENS_LINKS, dtype=str).fillna("")
    movies = pd.read_csv(SOURCE_MOVIELENS_MOVIES, dtype=str).fillna("")

    required_cross = {
        "movie_entity_id", "source_uri", "dbpedia_imdb_ids"
    }
    missing = required_cross - set(cross.columns)
    if missing:
        raise ValueError(
            "21.17C source lacks required columns: "
            + ", ".join(sorted(missing))
        )

    if not {"movieId", "imdbId", "tmdbId"}.issubset(links.columns):
        raise ValueError("MovieLens links.csv has an unexpected schema.")

    if not {"movieId", "title"}.issubset(movies.columns):
        raise ValueError("MovieLens movies.csv has an unexpected schema.")

    if cross["movie_entity_id"].duplicated().any():
        raise ValueError("Duplicate movie_entity_id values in 21.17C.")

    if links["movieId"].duplicated().any():
        raise ValueError("Duplicate MovieLens movieId values.")

    if movies["movieId"].duplicated().any():
        raise ValueError("Duplicate MovieLens movieId values in movies.csv.")

    # Attach titles without changing the source files.
    ml = links.merge(
        movies[["movieId", "title"]],
        on="movieId",
        how="left",
        validate="one_to_one",
    )

    ml["imdb_normalized"] = ml["imdbId"].map(normalize_imdb)
    ml["tmdb_normalized"] = ml["tmdbId"].map(normalize_tmdb)

    # IMDb -> all MovieLens rows. Keep duplicate TMDb candidates visible.
    ml_by_imdb = defaultdict(list)
    for row in ml.to_dict("records"):
        imdb = row.get("imdb_normalized")
        if imdb:
            ml_by_imdb[imdb].append(row)

    # Use DBpedia-origin identifiers only.
    tasks = []
    for row in cross.to_dict("records"):
        dbpedia_ids = parse_json_list(row.get("dbpedia_imdb_ids"))
        normalized_ids = sorted({
            norm
            for raw in dbpedia_ids
            if (norm := normalize_imdb(raw))
        })
        for imdb_id in normalized_ids:
            tasks.append({
                "movie_entity_id": str(row["movie_entity_id"]),
                "source_uri": str(row["source_uri"]),
                "dbpedia_imdb_id": imdb_id,
            })

    tasks_df = pd.DataFrame(tasks)
    if tasks_df.empty:
        raise RuntimeError(
            "No valid DBpedia IMDb identifiers found in dbpedia_imdb_ids. "
            "Do not substitute matching_imdb_ids: that would change the "
            "independence of the experiment."
        )

    return cross, ml, ml_by_imdb, tasks_df


def load_checkpoint() -> dict[str, dict]:
    completed = {}
    if not CHECKPOINT.exists():
        return completed

    with CHECKPOINT.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                imdb_id = item.get("dbpedia_imdb_id")
                if imdb_id:
                    completed[imdb_id] = item
            except json.JSONDecodeError:
                print(
                    f"Warning: ignoring malformed checkpoint line "
                    f"{line_number}."
                )
    return completed


def make_session(token: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "accept": "application/json",
        "User-Agent": "Research-Experiment-21.17G/1.0",
    })

    # Retry connection errors and selected server errors. HTTP 429 is
    # handled explicitly below so Retry-After can be respected.
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=1.0,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


def fetch_tmdb(session: requests.Session, imdb_id: str) -> dict:
    url = f"{API_BASE}/find/{imdb_id}"
    params = {"external_source": "imdb_id"}

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = session.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "")
                try:
                    wait = max(1.0, float(retry_after))
                except (TypeError, ValueError):
                    wait = min(60.0, 5.0 * attempt)
                print(
                    f"Rate limited for {imdb_id}; waiting {wait:.1f}s."
                )
                time.sleep(wait)
                continue

            if response.status_code in (401, 403):
                raise RuntimeError(
                    f"TMDb authentication/permission failure "
                    f"(HTTP {response.status_code}). Check token."
                )

            response.raise_for_status()
            payload = response.json()

            results = []
            for array_name, media_type in RESULT_ARRAYS:
                for item in payload.get(array_name, []) or []:
                    results.append({
                        "tmdb_id": normalize_tmdb(item.get("id")),
                        "tmdb_media_type": media_type,
                        "tmdb_title": (
                            item.get("title")
                            or item.get("name")
                            or item.get("original_title")
                            or item.get("original_name")
                            or ""
                        ),
                        "tmdb_original_title": (
                            item.get("original_title")
                            or item.get("original_name")
                            or ""
                        ),
                        "tmdb_release_date": (
                            item.get("release_date")
                            or item.get("first_air_date")
                            or ""
                        ),
                        "tmdb_original_language": (
                            item.get("original_language") or ""
                        ),
                    })

            return {
                "dbpedia_imdb_id": imdb_id,
                "request_status": "HTTP_200",
                "http_status": response.status_code,
                "retrieved_at_utc": utc_now(),
                "tmdb_results": results,
                "error": "",
            }

        except RuntimeError:
            raise
        except requests.RequestException as exc:
            if attempt == MAX_ATTEMPTS:
                return {
                    "dbpedia_imdb_id": imdb_id,
                    "request_status": "REQUEST_FAILED",
                    "http_status": "",
                    "retrieved_at_utc": utc_now(),
                    "tmdb_results": [],
                    "error": str(exc)[:1000],
                }
            wait = min(60.0, 2.0 ** attempt)
            print(
                f"Request error for {imdb_id}, attempt "
                f"{attempt}/{MAX_ATTEMPTS}: {exc}"
            )
            time.sleep(wait)
        except (ValueError, json.JSONDecodeError) as exc:
            return {
                "dbpedia_imdb_id": imdb_id,
                "request_status": "INVALID_JSON_RESPONSE",
                "http_status": "",
                "retrieved_at_utc": utc_now(),
                "tmdb_results": [],
                "error": str(exc)[:1000],
            }

    return {
        "dbpedia_imdb_id": imdb_id,
        "request_status": "REQUEST_FAILED",
        "http_status": "",
        "retrieved_at_utc": utc_now(),
        "tmdb_results": [],
        "error": "Maximum attempts exceeded.",
    }


def append_checkpoint(item: dict) -> None:
    with CHECKPOINT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(item), ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    print("=" * 75)
    print("EXPERIMENT 21.17G: INDEPENDENT TMDb IDENTIFIER CROSS-VALIDATION")
    print("=" * 75)

    token = os.getenv("TMDB_ACCESS_TOKEN")
    if not token:
        raise RuntimeError(
            "TMDB_ACCESS_TOKEN is not configured in this PowerShell session."
        )

    print("\n[1] SOURCE INVENTORY")
    cross, ml, ml_by_imdb, tasks = load_sources()

    print(f"DBpedia identities: {len(cross):,}")
    print(f"MovieLens link records: {len(ml):,}")
    print(f"DBpedia identity/IMDb identifier pairs: {len(tasks):,}")
    unique_ids = sorted(tasks["dbpedia_imdb_id"].unique())
    print(f"Unique DBpedia-origin IMDb identifiers: {len(unique_ids):,}")

    print("\n[2] SOURCE INTEGRITY")
    checks = {
        "dbpedia_population_6112": len(cross) == 6112,
        "dbpedia_entity_ids_unique": cross["movie_entity_id"].is_unique,
        "all_task_ids_normalized": all(
            re.fullmatch(r"tt\d{7,}", x) for x in unique_ids
        ),
        "movielens_movie_ids_unique": ml["movieId"].is_unique,
    }
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    if not all(checks.values()):
        raise RuntimeError("Source integrity validation failed.")

    print("\n[3] TMDb API RETRIEVAL")
    checkpoint = load_checkpoint()
    remaining = [x for x in unique_ids if x not in checkpoint]

    print(f"Previously checkpointed IMDb IDs: {len(checkpoint):,}")
    print(f"Remaining unique IMDb IDs: {len(remaining):,}")

    session = make_session(token)

    for i, imdb_id in enumerate(remaining, start=1):
        result = fetch_tmdb(session, imdb_id)
        append_checkpoint(result)
        checkpoint[imdb_id] = result

        if i % 25 == 0 or i == len(remaining):
            print(
                f"Progress: {i:,}/{len(remaining):,} new identifiers; "
                f"checkpointed total: {len(checkpoint):,}"
            )
        time.sleep(REQUEST_DELAY_SECONDS)

    session.close()

    # Expand API results back to DBpedia entity/identifier rows.
    api_rows = []
    for task in tasks.to_dict("records"):
        imdb_id = task["dbpedia_imdb_id"]
        response = checkpoint[imdb_id]
        results = response.get("tmdb_results", [])

        ml_rows = ml_by_imdb.get(imdb_id, [])
        ml_tmdb_ids = sorted({
            row["tmdb_normalized"]
            for row in ml_rows
            if row.get("tmdb_normalized")
        })

        api_tmdb_ids = sorted({
            item["tmdb_id"]
            for item in results
            if item.get("tmdb_id")
        })

        shared_ids = sorted(set(api_tmdb_ids) & set(ml_tmdb_ids))

        if response["request_status"] != "HTTP_200":
            assessment = "TMDB_REQUEST_FAILED"
        elif not api_tmdb_ids:
            assessment = "TMDB_NO_RESULT"
        elif not ml_tmdb_ids:
            assessment = "TMDB_FOUND_NO_MOVIELENS_REFERENCE"
        elif len(api_tmdb_ids) > 1:
            assessment = "MULTIPLE_TMDB_RESULTS_REVIEW"
        elif len(ml_tmdb_ids) > 1:
            assessment = "MULTIPLE_MOVIELENS_TMDB_CANDIDATES_REVIEW"
        elif shared_ids:
            assessment = "TMDB_MOVIELENS_IDENTIFIER_AGREEMENT"
        else:
            assessment = "TMDB_IDENTIFIER_CONFLICT"

        api_rows.append({
            "movie_entity_id": task["movie_entity_id"],
            "source_uri": task["source_uri"],
            "dbpedia_imdb_id": imdb_id,
            "request_status": response["request_status"],
            "http_status": response.get("http_status", ""),
            "retrieved_at_utc": response.get("retrieved_at_utc", ""),
            "tmdb_result_count": len(results),
            "tmdb_ids": json.dumps(api_tmdb_ids),
            "tmdb_results": json.dumps(results, ensure_ascii=False),
            "movielens_record_count_for_imdb": len(ml_rows),
            "movielens_tmdb_ids": json.dumps(ml_tmdb_ids),
            "shared_tmdb_ids": json.dumps(shared_ids),
            "identity_evidence_assessment": assessment,
            "error": response.get("error", ""),
            "automatic_identity_verification": False,
            "automatic_entity_merging": False,
        })

    detail = pd.DataFrame(api_rows)

    # Entity-level master: preserve every original DBpedia identity.
    detail_by_entity = defaultdict(list)
    for row in api_rows:
        detail_by_entity[row["movie_entity_id"]].append(row)

    master_rows = []
    for row in cross.to_dict("records"):
        entity_id = str(row["movie_entity_id"])
        evidence = detail_by_entity.get(entity_id, [])
        assessments = sorted({
            x["identity_evidence_assessment"] for x in evidence
        })
        all_api_ids = sorted({
            tmdb_id
            for x in evidence
            for tmdb_id in json.loads(x["tmdb_ids"])
        })
        all_ml_ids = sorted({
            tmdb_id
            for x in evidence
            for tmdb_id in json.loads(x["movielens_tmdb_ids"])
        })

        if not evidence:
            entity_status = "UNRESOLVED_NO_DBPEDIA_IMDB_IDENTIFIER"
        elif any(x == "TMDB_REQUEST_FAILED" for x in assessments):
            entity_status = "TMDB_REQUEST_FAILED_REVIEW"
        elif any(x in assessments for x in (
            "TMDB_IDENTIFIER_CONFLICT",
            "MULTIPLE_TMDB_RESULTS_REVIEW",
            "MULTIPLE_MOVIELENS_TMDB_CANDIDATES_REVIEW",
        )):
            entity_status = "TMDB_CONFLICT_OR_CARDINALITY_REVIEW"
        elif any(
            x == "TMDB_MOVIELENS_IDENTIFIER_AGREEMENT"
            for x in assessments
        ):
            entity_status = "TMDB_CROSS_REFERENCE_CORROBORATION"
        elif any(x == "TMDB_NO_RESULT" for x in assessments):
            entity_status = "UNRESOLVED_TMDB_NO_RESULT"
        else:
            entity_status = "UNRESOLVED_INSUFFICIENT_EVIDENCE"

        master_rows.append({
            "movie_entity_id": entity_id,
            "source_uri": row["source_uri"],
            "dbpedia_imdb_ids": row["dbpedia_imdb_ids"],
            "dbpedia_imdb_identifier_count": len(
                parse_json_list(row["dbpedia_imdb_ids"])
            ),
            "queried_identifier_count": len(evidence),
            "tmdb_result_ids": json.dumps(all_api_ids),
            "movielens_tmdb_candidate_ids": json.dumps(all_ml_ids),
            "identifier_assessments": json.dumps(assessments),
            "entity_evidence_assessment": entity_status,
            "identity_verified": False,
            "automatic_entity_merging": False,
        })

    master = pd.DataFrame(master_rows)

    # Conflict register.
    conflicts = detail[
        detail["identity_evidence_assessment"].isin([
            "TMDB_IDENTIFIER_CONFLICT",
            "MULTIPLE_TMDB_RESULTS_REVIEW",
            "MULTIPLE_MOVIELENS_TMDB_CANDIDATES_REVIEW",
            "TMDB_REQUEST_FAILED",
        ])
    ].copy()

    # Unresolved population includes no-identifier and insufficient-evidence.
    unresolved = master[
        master["entity_evidence_assessment"].str.startswith(
            ("UNRESOLVED", "TMDB_REQUEST_FAILED", "TMDB_CONFLICT")
        )
    ].copy()

    # Output files.
    outputs = {
        "tmdb_identity_evidence_master.csv": master,
        "tmdb_identifier_cross_reference.csv": detail,
        "tmdb_identifier_conflicts.csv": conflicts,
        "tmdb_unresolved_identities.csv": unresolved,
    }

    print("\n[4] EVIDENCE SUMMARY")
    print("Entity assessments:")
    print(master["entity_evidence_assessment"].value_counts().to_string())
    print("\nIdentifier assessments:")
    print(detail["identity_evidence_assessment"].value_counts().to_string())

    for filename, frame in outputs.items():
        frame.to_csv(OUT / filename, index=False, encoding="utf-8-sig")
        print(f"Exported {filename}: {len(frame):,} rows")

    # Population and safety validations.
    final_checks = {
        "master_population_6112": len(master) == 6112,
        "master_entity_ids_unique": master["movie_entity_id"].is_unique,
        "all_source_entities_preserved": set(master["movie_entity_id"])
            == set(cross["movie_entity_id"].astype(str)),
        "detail_population_matches_identifier_pairs": len(detail) == len(tasks),
        "no_automatic_identity_verification": not master["identity_verified"].any(),
        "no_automatic_entity_merging": not master["automatic_entity_merging"].any(),
        "all_master_rows_have_source_uri": master["source_uri"].ne("").all(),
    }

    print("\n[5] FINAL VALIDATION")
    for name, passed in final_checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    if not all(final_checks.values()):
        raise RuntimeError(
            "Final validation failed. Inspect outputs; do not treat "
            "the experiment as completed."
        )

    # Manifest with hashes and explicit limitations.
    source_hashes = {}
    for path in [
        SOURCE_CROSSREF,
        SOURCE_MOVIELENS_LINKS,
        SOURCE_MOVIELENS_MOVIES,
    ]:
        source_hashes[str(path.relative_to(ROOT))] = sha256(path)

    manifest = {
        "experiment": "21.17G",
        "name": "Independent TMDb Identifier Cross-Validation",
        "generated_utc": utc_now(),
        "sources": {
            "dbpedia_imdb_identifier_source": str(
                SOURCE_CROSSREF.relative_to(ROOT)
            ),
            "movielens_links": str(
                SOURCE_MOVIELENS_LINKS.relative_to(ROOT)
            ),
            "movielens_movies": str(
                SOURCE_MOVIELENS_MOVIES.relative_to(ROOT)
            ),
        },
        "source_sha256": source_hashes,
        "counts": {
            "dbpedia_identities": len(cross),
            "entities_with_dbpedia_imdb_identifiers": int(
                master["queried_identifier_count"].gt(0).sum()
            ),
            "entity_identifier_pairs": len(tasks),
            "unique_dbpedia_imdb_identifiers_queried": len(unique_ids),
            "tmdb_identifier_evidence_rows": len(detail),
            "conflict_or_exception_rows": len(conflicts),
            "unresolved_entities": len(unresolved),
        },
        "identifier_assessment_distribution": {
            str(k): int(v)
            for k, v in detail["identity_evidence_assessment"]
            .value_counts().items()
        },
        "entity_assessment_distribution": {
            str(k): int(v)
            for k, v in master["entity_evidence_assessment"]
            .value_counts().items()
        },
        "validation": {
            str(k): bool(v) for k, v in final_checks.items()
        },
        "methodology": {
            "lookup_endpoint": "TMDb API v3 /find/{external_id}",
            "external_source": "imdb_id",
            "lookup_identifier_provenance": (
                "dbpedia_imdb_ids column from Experiment 21.17C only"
            ),
            "matching_imdb_ids_used_as_lookup_source": False,
            "movielens_role": (
                "Intermediate comparison source; not independent identity proof"
            ),
            "all_tmdb_result_types_preserved": True,
            "automatic_identity_verification": False,
            "automatic_entity_merging": False,
            "checkpointing_enabled": True,
        },
        "limitations": [
            "TMDb lookup results corroborate identifier correspondence but do "
            "not alone prove DBpedia entity identity.",
            "MovieLens TMDb identifiers are used as comparison references.",
            "A successful API response with no results is not an API failure.",
            "Multiple results and multiple MovieLens candidates remain review cases.",
            "Identities without DBpedia-origin IMDb identifiers remain unresolved.",
            "No automatic identity verification or entity merging is performed.",
        ],
        "outputs": list(outputs.keys()) + [
            CHECKPOINT.name,
            "experiment_21_17G_manifest.json",
            "reproducibility_report.txt",
        ],
        "completion_status": "COMPLETED_ALL_VALIDATIONS_PASSED",
    }

    manifest_path = OUT / "experiment_21_17G_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    report_lines = [
        "EXPERIMENT 21.17G — REPRODUCIBILITY REPORT",
        f"Generated UTC: {manifest['generated_utc']}",
        "",
        "Lookup identifiers were restricted to DBpedia-origin IMDb IDs.",
        "MovieLens was used only as an intermediate comparison source.",
        "All TMDb result categories were preserved.",
        "No automatic identity verification or entity merging was performed.",
        "",
        "Counts:",
    ]
    report_lines.extend(
        f"- {key}: {value}"
        for key, value in manifest["counts"].items()
    )
    report_lines.extend(["", "Validation:"])
    report_lines.extend(
        f"- {key}: {'PASS' if value else 'FAIL'}"
        for key, value in final_checks.items()
    )
    report_lines.extend(["", "Limitations:"])
    report_lines.extend(f"- {item}" for item in manifest["limitations"])

    (OUT / "reproducibility_report.txt").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 75)
    print("EXPERIMENT 21.17G COMPLETED")
    print("=" * 75)
    print(f"DBpedia identities preserved: {len(master):,}")
    print(f"IMDb identifier pairs queried: {len(detail):,}")
    print(f"Output directory: {OUT}")
    print("Original source files unchanged.")
    print("Knowledge graph unchanged.")
    print("No automatic identity merging.")
    print("EXPERIMENT 21.17G FINISHED.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n" + "=" * 75)
        print("EXPERIMENT 21.17G FAILED")
        print(f"{type(exc).__name__}: {exc}")
        print("=" * 75)
        print(
            "Do not interpret partial outputs as completed results. "
            "Checkpoint data, if present, can be used to resume."
        )
        sys.exit(1)
