import csv
import json
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone

ROOT = Path(".")
OUT = ROOT / "results_category_audit" / "experiment_21_17K1"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "dbpedia": ROOT / "results_category_audit/experiment_21_17B/dbpedia_independent_metadata_evidence.csv",
    "imdb": ROOT / "results_category_audit/experiment_21_17F/imdb_evidence_record_level.csv",
    "tmdb": ROOT / "results_category_audit/experiment_21_17H/tmdb_evidence_record_level.csv",
    "integrated": ROOT / "results_category_audit/experiment_21_17I/cross_source_identity_record_level.csv",
    "tmdb_crossref": ROOT / "results_category_audit/experiment_21_17G/tmdb_identifier_cross_reference.csv",
    "movielens": ROOT / "external_validation/movielens_32m/links.csv",
}

def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required source missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def parse_json(value):
    if not value:
        return []
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (ValueError, TypeError):
        return []

def truth(value):
    return str(value).strip().lower() in ("true", "1", "yes")

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def write_csv(name, rows):
    path = OUT / name
    if not rows:
        raise ValueError(f"Refusing to export empty output: {name}")
    fields = list(dict.fromkeys(
        key for row in rows for key in row.keys()
    ))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{name}: {len(rows):,} rows")

def main():
    print("=" * 75)
    print("EXPERIMENT 21.17K1: IDENTIFIER DEPENDENCY AUDIT")
    print("=" * 75)

    print("\n[1] LOADING SOURCES")
    data = {name: read_csv(path) for name, path in SOURCES.items()}

    for name, rows in data.items():
        print(f"{name}: {len(rows):,} records")

    dbpedia = data["dbpedia"]
    imdb = data["imdb"]
    tmdb = data["tmdb"]
    integrated = data["integrated"]
    crossref = data["tmdb_crossref"]
    movielens = data["movielens"]

    # Required schema checks
    required = {
        "dbpedia": {"movie_entity_id", "source_uri", "imdb_ids", "tmdb_ids"},
        "imdb": {"movie_entity_id", "imdb_id", "imdb_title",
                 "record_evidence_assessment"},
        "tmdb": {"movie_entity_id", "dbpedia_imdb_id", "tmdb_id",
                 "tmdb_title", "tmdb_request_status",
                 "movielens_shared_tmdb_id"},
        "integrated": {"movie_entity_id", "evidence_source",
                       "identifier", "record_assessment"},
        "tmdb_crossref": {"movie_entity_id", "dbpedia_imdb_id"},
        "movielens": {"movieId", "imdbId", "tmdbId"},
    }

    for name, columns in required.items():
        actual = set(data[name][0].keys())
        missing = columns - actual
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")

    print("\n[2] SOURCE POPULATION INTEGRITY")

    entity_ids = [r["movie_entity_id"] for r in dbpedia]
    checks = {
        "dbpedia_population_6112": len(dbpedia) == 6112,
        "dbpedia_entity_ids_unique": len(set(entity_ids)) == len(entity_ids),
        "imdb_population_577": len(imdb) == 577,
        "tmdb_population_582": len(tmdb) == 582,
        "integrated_population_1159": len(integrated) == 1159,
        "crossref_population_578": len(crossref) == 578,
    }

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    if not all(checks.values()):
        raise ValueError("Source population validation failed.")

    # Index evidence by DBpedia entity
    imdb_by_entity = defaultdict(list)
    tmdb_by_entity = defaultdict(list)
    integrated_by_entity = defaultdict(list)

    for r in imdb:
        imdb_by_entity[r["movie_entity_id"]].append(r)

    for r in tmdb:
        tmdb_by_entity[r["movie_entity_id"]].append(r)

    for r in integrated:
        integrated_by_entity[r["movie_entity_id"]].append(r)

    # MovieLens cross-reference index
    movielens_by_imdb = defaultdict(list)
    for r in movielens:
        imdb_id = (r.get("imdbId") or "").strip()
        tmdb_id = (r.get("tmdbId") or "").strip()
        if imdb_id and tmdb_id:
            key = imdb_id[2:] if imdb_id.lower().startswith("tt") else imdb_id
            movielens_by_imdb[key].append(tmdb_id)

    # TMDb lookup provenance index
    crossref_by_entity = defaultdict(list)
    for r in crossref:
        crossref_by_entity[r["movie_entity_id"]].append(r)

    print("\n[3] IDENTIFIER DEPENDENCY ANALYSIS")

    master = []
    record_audit = []
    source_counts = Counter()
    exceptions = []

    for d in dbpedia:
        eid = d["movie_entity_id"]
        imdb_rows = imdb_by_entity[eid]
        tmdb_rows = tmdb_by_entity[eid]
        integrated_rows = integrated_by_entity[eid]
        crossref_rows = crossref_by_entity[eid]

        direct_imdb = parse_json(d.get("imdb_ids"))
        direct_tmdb = parse_json(d.get("tmdb_ids"))

        imdb_ids = sorted({
            r["imdb_id"] for r in imdb_rows if r.get("imdb_id")
        })

        tmdb_ids = sorted({
            r["tmdb_id"] for r in tmdb_rows if r.get("tmdb_id")
        })

        tmdb_seed_ids = sorted({
            r["dbpedia_imdb_id"]
            for r in tmdb_rows if r.get("dbpedia_imdb_id")
        })

        shared_movielens = sum(
            1 for r in tmdb_rows
            if truth(r.get("movielens_shared_tmdb_id"))
        )

        recomputed_movielens_shared = sum(
            1 for r in tmdb_rows
            if r.get("dbpedia_imdb_id", "").lower().startswith("tt")
            and r.get("tmdb_id", "").strip() in
            movielens_by_imdb.get(r["dbpedia_imdb_id"][2:], [])
        )

        movielens_flag_disagreements = sum(
            1 for r in tmdb_rows
            if truth(r.get("movielens_shared_tmdb_id")) != (
                r.get("dbpedia_imdb_id", "").lower().startswith("tt")
                and r.get("tmdb_id", "").strip() in
                movielens_by_imdb.get(r["dbpedia_imdb_id"][2:], [])
            )
        )

        # Verify whether the TMDb lookup key matches an IMDb record
        matched_seed_ids = sorted(
            set(tmdb_seed_ids).intersection(imdb_ids)
        )

        direct_tmdb_values = {
            str(x.get("value", "")).strip()
            for x in direct_tmdb if isinstance(x, dict)
        }
        direct_tmdb_values.discard("")

        if direct_tmdb_values:
            direct_tmdb_status = "DIRECT_DBPEDIA_TMDB_IDENTIFIER_PRESENT"
        else:
            direct_tmdb_status = "NO_DIRECT_DBPEDIA_TMDB_IDENTIFIER"

        if tmdb_rows and tmdb_seed_ids:
            tmdb_dependency = "IMDB_IDENTIFIER_SEEDED_TMDB_LOOKUP"
        elif tmdb_rows:
            tmdb_dependency = "TMDB_PROVENANCE_REQUIRES_REVIEW"
        else:
            tmdb_dependency = "NO_TMDB_CANDIDATE_RECORD"

        master_row = {
            "movie_entity_id": eid,
            "source_uri": d["source_uri"],
            "dbpedia_direct_imdb_identifier_count": len(direct_imdb),
            "dbpedia_direct_tmdb_identifier_count": len(direct_tmdb_values),
            "official_imdb_record_count": len(imdb_rows),
            "tmdb_record_count_including_placeholders": len(tmdb_rows),
            "tmdb_candidate_count": len(tmdb_ids),
            "tmdb_lookup_seed_identifier_count": len(tmdb_seed_ids),
            "tmdb_seed_ids_matching_imdb_records": len(matched_seed_ids),
            "movielens_shared_tmdb_record_count": shared_movielens,
            "movielens_recomputed_shared_count": recomputed_movielens_shared,
            "movielens_flag_disagreement_count": movielens_flag_disagreements,
            "tmdb_lookup_dependency": tmdb_dependency,
            "dbpedia_tmdb_identifier_status": direct_tmdb_status,
            "integrated_record_count": len(integrated_rows),
            "crossref_pair_count": len(crossref_rows),
            "identity_verified": False,
            "automatic_entity_merging": False,
        }
        master.append(master_row)

        if direct_tmdb_values:
            source_counts["DBPEDIA_DIRECT_TMDB_IDENTIFIER"] += 1

        if tmdb_rows and tmdb_seed_ids:
            source_counts["IMDB_SEEDED_TMDB_LOOKUP"] += len(tmdb_rows)

        if shared_movielens:
            source_counts["MOVIELENS_SHARED_TMDB_SUPPORT"] += shared_movielens

        if imdb_rows:
            source_counts["OFFICIAL_IMDB_DATASET_RECORD"] += len(imdb_rows)

        if not direct_tmdb_values and tmdb_rows:
            exceptions.append({
                "movie_entity_id": eid,
                "source_uri": d["source_uri"],
                "exception_type": "TMDB_CANDIDATES_WITHOUT_DIRECT_DBPEDIA_TMDB_ID",
                "tmdb_candidate_count": len(tmdb_ids),
                "tmdb_seed_identifier_count": len(tmdb_seed_ids),
            })

        # Record-level evidence dependency
        for r in integrated_rows:
            source = r.get("evidence_source", "")
            identifier = r.get("identifier", "")

            if source == "IMDb":
                dependency = "DBPEDIA_ORIGIN_IMDB_IDENTIFIER"
                retrieval = "OFFICIAL_IMDB_DATASET"
                independent_identity_evidence = False

            elif source == "TMDb":
                dependency = "IMDB_SEEDED_TMDB_LOOKUP"
                retrieval = "TMDB_API"
                independent_identity_evidence = False

            else:
                dependency = "SOURCE_PROVENANCE_REQUIRES_REVIEW"
                retrieval = "UNKNOWN"
                independent_identity_evidence = False

            record_audit.append({
                "movie_entity_id": eid,
                "source_uri": d["source_uri"],
                "evidence_source": source,
                "identifier": identifier,
                "evidence_title": r.get("evidence_title", ""),
                "evidence_year": r.get("evidence_year", ""),
                "record_assessment": r.get("record_assessment", ""),
                "identifier_dependency": dependency,
                "retrieval_source": retrieval,
                "identity_evidence_independence": (
                    "NOT_ESTABLISHED_BY_DEPENDENCY_AUDIT"
                ),
                "automatic_identity_verification": False,
                "automatic_entity_merging": False,
            })

    print("\n[4] RECORD-LEVEL DEPENDENCY SUMMARY")

    dependency_counts = Counter(
        r["identifier_dependency"] for r in record_audit
    )

    for name, count in sorted(dependency_counts.items()):
        print(f"{name}: {count}")

    print("\n[5] SOURCE DEPENDENCY SUMMARY")

    summary = [
        {
            "dependency_category": name,
            "record_or_entity_count": count,
            "interpretation": {
                "DBPEDIA_DIRECT_TMDB_IDENTIFIER":
                    "DBpedia metadata directly contains a TMDb identifier.",
                "IMDB_SEEDED_TMDB_LOOKUP":
                    "TMDb candidate retrieval is associated with an IMDb lookup identifier.",
                "MOVIELENS_SHARED_TMDB_SUPPORT":
                    "MovieLens cross-reference shares a TMDb identifier.",
                "OFFICIAL_IMDB_DATASET_RECORD":
                    "Official IMDb dataset record retrieved for an IMDb identifier.",
            }.get(name, "Requires provenance review."),
        }
        for name, count in sorted(source_counts.items())
    ]

    # Add explicit no-direct-TMDb population count
    no_direct = sum(
        1 for r in master
        if r["dbpedia_direct_tmdb_identifier_count"] == 0
    )
    summary.append({
        "dependency_category": "NO_DIRECT_DBPEDIA_TMDB_IDENTIFIER",
        "record_or_entity_count": no_direct,
        "interpretation": "No direct TMDb identifier found in DBpedia metadata.",
    })

    # Validate record coverage against integrated source
    print("\n[6] FINAL VALIDATION")

    validations = {
        "entity_master_population_6112": len(master) == 6112,
        "entity_master_ids_unique":
            len({r["movie_entity_id"] for r in master}) == 6112,
        "integrated_record_population_preserved":
            len(record_audit) == len(integrated),
        "all_integrated_records_accounted_for":
            len(record_audit) == 1159,
        "all_source_entities_preserved":
            {r["movie_entity_id"] for r in master} == set(entity_ids),
        "no_automatic_identity_verification":
            all(not r["automatic_identity_verification"]
                for r in record_audit),
        "no_automatic_entity_merging":
            all(not r["automatic_entity_merging"]
                for r in master),
    }

    for name, passed in validations.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    if not all(validations.values()):
        raise ValueError("Final validation failed. Outputs not finalized.")

    print("\n[7] EXPORTING RESULTS")

    write_csv("identifier_dependency_master.csv", master)
    write_csv("identifier_dependency_record_level.csv", record_audit)
    write_csv("source_dependency_summary.csv", summary)

    if exceptions:
        write_csv("dependency_exception_register.csv", exceptions)
    else:
        write_csv("dependency_exception_register.csv", [{
            "movie_entity_id": "",
            "source_uri": "",
            "exception_type": "NO_EXCEPTIONS",
            "tmdb_candidate_count": 0,
            "tmdb_seed_identifier_count": 0,
        }])

    manifest = {
        "experiment": "21.17K1",
        "name": "Identifier Dependency Audit",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": {
            name: sha256(path) for name, path in SOURCES.items()
        },
        "counts": {
            "dbpedia_entities": len(dbpedia),
            "imdb_records": len(imdb),
            "tmdb_records": len(tmdb),
            "integrated_records": len(integrated),
            "tmdb_crossref_records": len(crossref),
            "movielens_records": len(movielens),
            "master_entities": len(master),
            "audited_integrated_records": len(record_audit),
        },
        "validation": validations,
        "methodology": {
            "tmdb_lookup_dependency":
                "TMDb records retain the IMDb identifier used in the lookup.",
            "movieLens_role":
                "Shared identifier cross-reference; not independent proof.",
            "direct_dbpedia_tmdb_ids":
                "Extracted from DBpedia metadata without inferring missing values.",
            "identity_verification": False,
            "automatic_entity_merging": False,
        },
        "limitations": [
            "The audit traces recorded identifier relationships, not every internal API operation.",
            "IMDb and TMDb metadata are not automatically independent identity evidence.",
            "A shared MovieLens identifier is a cross-reference, not independent identity proof.",
            "Direct identifier absence in retrieved DBpedia metadata does not prove global absence.",
            "No identity was automatically verified or merged.",
        ],
        "completion_status": "COMPLETED_ALL_VALIDATIONS_PASSED",
    }

    with (OUT / "experiment_21_17K1_manifest.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)

    report = [
        "EXPERIMENT 21.17K1 - REPRODUCIBILITY REPORT",
        "=" * 65,
        f"Entities: {len(master)}",
        f"Integrated records audited: {len(record_audit)}",
        f"IMDb records: {len(imdb)}",
        f"TMDb records: {len(tmdb)}",
        "",
        "All original inputs were read-only.",
        "No knowledge graph modification.",
        "No automatic identity verification.",
        "No automatic entity merging.",
        "",
        "IMPORTANT: Identifier dependency is not equivalent to",
        "independent identity corroboration.",
    ]

    (OUT / "reproducibility_report.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )

    with (OUT / "experiment_21_17K1_validation.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(validations, f, indent=4)

    print("\n" + "=" * 75)
    print("EXPERIMENT 21.17K1 COMPLETED")
    print("=" * 75)
    print(f"Output: {OUT.resolve()}")
    print("Original source files unchanged.")
    print("No graph modification.")
    print("No automatic identity verification or merging.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nEXPERIMENT 21.17K1 FAILED")
        print(f"{type(exc).__name__}: {exc}")
        raise
