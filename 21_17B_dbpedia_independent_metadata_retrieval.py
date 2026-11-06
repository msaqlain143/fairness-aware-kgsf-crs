
from __future__ import annotations

import csv
import json
import re
import time
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import pandas as pd
import requests


# ============================================================
# EXPERIMENT 21.17B
# Independent DBpedia Metadata Retrieval
# ============================================================

ROOT = Path(__file__).resolve().parent

SOURCE = (
    ROOT / "results_category_audit"
    / "experiment_21_17A"
    / "dbpedia_uri_record_inventory.csv"
)

OUT = (
    ROOT / "results_category_audit"
    / "experiment_21_17B"
)

CHECKPOINT = OUT / "retrieval_checkpoint.jsonl"
EVIDENCE = OUT / "dbpedia_independent_metadata_evidence.csv"
FAILED = OUT / "failed_uri_records.csv"
SUMMARY = OUT / "metadata_retrieval_summary.json"
MANIFEST = OUT / "experiment_21_17B_manifest.json"

ENDPOINT = "https://dbpedia.org/sparql"

BATCH_SIZE = 20
MAX_RETRIES = 3
REQUEST_TIMEOUT = 60
REQUEST_DELAY = 1.2

HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "KG-FairCRS-Research/1.0"
}

PROPERTIES = {
    "http://www.w3.org/2000/01/rdf-schema#label": "labels",
    "http://dbpedia.org/ontology/abstract": "abstracts",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type": "rdf_types",
    "http://dbpedia.org/ontology/genre": "genres",
    "http://dbpedia.org/ontology/releaseDate": "release_dates",
    "http://dbpedia.org/ontology/imdbId": "imdb_ids",
    "http://dbpedia.org/property/imdbId": "imdb_ids",
    "http://dbpedia.org/property/tmdbId": "tmdb_ids",
    "http://dbpedia.org/ontology/wikiPageID": "wiki_page_ids",
}

OUTPUT_FIELDS = list(dict.fromkeys(PROPERTIES.values()))


def log(message):
    print(message, flush=True)


def find_column(df, candidates):
    for name in candidates:
        if name in df.columns:
            values = df[name].astype(str)
            if values.str.contains(
                r"^https?://dbpedia\.org/resource/",
                regex=True
            ).any():
                return name

    # Fallback: inspect every column.
    for name in df.columns:
        values = df[name].astype(str)
        if values.str.contains(
            r"^https?://dbpedia\.org/resource/",
            regex=True
        ).any():
            return name

    raise ValueError(
        "Could not identify a DBpedia URI column. "
        f"Available columns: {df.columns.tolist()}"
    )


def load_source():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    df = pd.read_csv(
        SOURCE,
        dtype=str,
        keep_default_na=False
    )

    uri_col = find_column(
        df,
        [
            "normalized_uri",
            "resource_uri",
            "original_movie_uri",
            "canonical_identifier",
            "original_canonical_identifier",
        ]
    )

    id_col = next(
        (
            c for c in [
                "movie_entity_id",
                "canonical_entity_id",
                "entity_id"
            ]
            if c in df.columns
        ),
        None
    )

    records = []

    for _, row in df.iterrows():
        uri = str(row[uri_col]).strip()

        if not re.match(
            r"^https?://dbpedia\.org/resource/[^\s<>]+$",
            uri
        ):
            continue

        records.append({
            "movie_entity_id": (
                str(row[id_col]) if id_col else ""
            ),
            "source_uri": uri,
        })

    if len(records) != 6112:
        raise ValueError(
            f"Expected 6112 URI identities; "
            f"found {len(records)}. "
            "Do not proceed until the population is reconciled."
        )

    if len({x["source_uri"] for x in records}) != len(records):
        raise ValueError("Duplicate URI identities detected.")

    return records, uri_col, id_col


def load_checkpoint():
    completed = {}

    if not CHECKPOINT.exists():
        return completed

    with CHECKPOINT.open(
        "r", encoding="utf-8"
    ) as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            completed[record["source_uri"]] = record

    return completed


def make_query(uris):
    values = "\n".join(
        f"<{uri}>" for uri in uris
    )

    predicates = ",\n".join(
        f"<{p}>" for p in PROPERTIES
    )

    return f"""
    SELECT ?resource ?p ?o
    WHERE {{
        VALUES ?resource {{
            {values}
        }}

        ?resource ?p ?o .

        FILTER (?p IN (
            {predicates}
        ))
    }}
    """


def retrieve_batch(uris):
    query = make_query(uris)

    response = requests.post(
        ENDPOINT,
        data={"query": query},
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    data = response.json()

    if "results" not in data:
        raise ValueError(
            "Unexpected SPARQL response structure."
        )

    return data["results"]["bindings"]


def process_bindings(uris, bindings):
    results = {}

    for uri in uris:
        results[uri] = {
            "source_uri": uri,
            **{field: [] for field in OUTPUT_FIELDS}
        }

    for binding in bindings:
        uri = binding["resource"]["value"]

        if uri not in results:
            continue

        predicate = binding["p"]["value"]
        field = PROPERTIES.get(predicate)

        if not field:
            continue

        obj = binding["o"]

        value = obj.get("value", "")
        datatype = obj.get("datatype", "")
        language = obj.get("xml:lang", "")

        entry = {
            "value": value,
            "language": language,
            "datatype": datatype,
            "term_type": obj.get("type", "")
        }

        results[uri][field].append(entry)

    output = []

    for uri, record in results.items():
        for field in OUTPUT_FIELDS:
            record[field] = json.dumps(
                record[field],
                ensure_ascii=False
            )

        record["retrieval_status"] = (
            "METADATA_RETRIEVED"
            if any(
                json.loads(record[field])
                for field in OUTPUT_FIELDS
            )
            else "NO_METADATA_RETURNED"
        )

        record["retrieved_at_utc"] = (
            datetime.now(timezone.utc).isoformat()
        )

        output.append(record)

    return output


def save_checkpoint(records):
    with CHECKPOINT.open(
        "a", encoding="utf-8"
    ) as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

        f.flush()


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("EXPERIMENT 21.17B: INDEPENDENT DBPEDIA METADATA RETRIEVAL")
    log("=" * 70)

    records, uri_col, id_col = load_source()

    log(f"Source URI records: {len(records):,}")
    log(f"Detected URI column: {uri_col}")
    log(f"Detected identity column: {id_col}")

    checkpoint = load_checkpoint()

    log(
        f"Previously checkpointed: "
        f"{len(checkpoint):,}"
    )

    remaining = [
        r for r in records
        if r["source_uri"] not in checkpoint
    ]

    batches = [
        remaining[i:i + BATCH_SIZE]
        for i in range(0, len(remaining), BATCH_SIZE)
    ]

    failures = []

    log(f"Remaining records: {len(remaining):,}")
    log(f"Batches remaining: {len(batches):,}")

    for number, batch in enumerate(batches, 1):
        uris = [r["source_uri"] for r in batch]

        success = False

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                bindings = retrieve_batch(uris)

                output = process_bindings(
                    uris,
                    bindings
                )

                save_checkpoint(output)

                for record in output:
                    checkpoint[record["source_uri"]] = record

                success = True
                break

            except Exception as exc:
                log(
                    f"Batch {number} attempt {attempt} "
                    f"failed: {exc}"
                )

                if attempt < MAX_RETRIES:
                    time.sleep(3 * attempt)

        if not success:
            for record in batch:
                failures.append({
                    "movie_entity_id": record["movie_entity_id"],
                    "source_uri": record["source_uri"],
                    "failure_reason": "BATCH_RETRIES_EXHAUSTED"
                })

        if number % 10 == 0 or number == len(batches):
            log(
                f"Progress: {number}/{len(batches)} batches | "
                f"checkpointed: {len(checkpoint):,}/6,112"
            )

        time.sleep(REQUEST_DELAY)

    # Build final master evidence using original source order.
    final = []

    for source in records:
        uri = source["source_uri"]

        evidence = checkpoint.get(uri)

        if evidence:
            final.append({
                **source,
                **evidence
            })

    evidence_df = pd.DataFrame(final)

    evidence_df.to_csv(
        EVIDENCE,
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(failures).to_csv(
        FAILED,
        index=False,
        encoding="utf-8-sig"
    )

    statuses = (
        evidence_df["retrieval_status"]
        .value_counts()
        .to_dict()
        if not evidence_df.empty
        else {}
    )

    successful = len(checkpoint)

    summary = {
        "experiment": "21.17B",
        "source_population": len(records),
        "checkpointed_records": successful,
        "exported_records": len(evidence_df),
        "failed_records": len(failures),
        "retrieval_status": statuses,
        "identity_verification_claim": False,
        "automatic_identity_merging": False,
        "master_mapping_modified": False,
        "knowledge_graph_modified": False,
        "completed_at_utc": datetime.now(
            timezone.utc
        ).isoformat()
    }

    SUMMARY.write_text(
        json.dumps(
            summary,
            indent=4,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    manifest = {
        **summary,
        "source_file": str(SOURCE),
        "source_sha256": hashlib.sha256(
            SOURCE.read_bytes()
        ).hexdigest(),
        "endpoint": ENDPOINT,
        "batch_size": BATCH_SIZE,
        "maximum_retries": MAX_RETRIES,
        "output_directory": str(OUT),
        "interpretation": (
            "Metadata retrieval is evidence collection, "
            "not independent identity verification."
        )
    }

    MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=4,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    log("\n" + "=" * 70)
    log("EXPERIMENT 21.17B FINISHED")
    log("=" * 70)

    log(f"Source identities: {len(records):,}")
    log(f"Checkpointed: {successful:,}")
    log(f"Exported: {len(evidence_df):,}")
    log(f"Failed: {len(failures):,}")

    log("\nRetrieval statuses:")
    log(json.dumps(statuses, indent=4))

    log(f"\nOutput: {OUT}")

    if failures:
        log(
            "WARNING: Some records failed. "
            "Rerun the script to retry them."
        )

    if successful != len(records):
        log(
            "INCOMPLETE: Do not treat this experiment "
            "as a complete retrieval."
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"\nEXPERIMENT 21.17B FAILED: {exc}")
        raise
