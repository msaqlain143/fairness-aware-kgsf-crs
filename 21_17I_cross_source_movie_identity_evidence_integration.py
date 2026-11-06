import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


# ============================================================
# EXPERIMENT 21.17I
# CROSS-SOURCE MOVIE IDENTITY EVIDENCE INTEGRATION
# ============================================================

ROOT = Path(__file__).resolve().parent

BASE = ROOT / "results_category_audit"

SOURCES = {
    "imdb_master": BASE / "experiment_21_17F" /
        "imdb_evidence_quality_master.csv",

    "imdb_records": BASE / "experiment_21_17F" /
        "imdb_evidence_record_level.csv",

    "tmdb_master": BASE / "experiment_21_17H" /
        "tmdb_evidence_quality_master.csv",

    "tmdb_records": BASE / "experiment_21_17H" /
        "tmdb_evidence_record_level.csv",

    "dbpedia": BASE / "experiment_21_17B" /
        "dbpedia_independent_metadata_evidence.csv",
}

OUT = BASE / "experiment_21_17I"
OUT.mkdir(parents=True, exist_ok=True)


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing source: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_int(value):
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return 0


def safe_json(value):
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return []


def unique_values(values):
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def main():

    print("=" * 75)
    print("EXPERIMENT 21.17I: CROSS-SOURCE MOVIE IDENTITY")
    print("EVIDENCE INTEGRATION")
    print("=" * 75)

    # --------------------------------------------------------
    # 1. LOAD SOURCES
    # --------------------------------------------------------

    data = {}

    print("\n[1] SOURCE INVENTORY")

    for name, path in SOURCES.items():
        data[name] = load_csv(path)
        print(f"{name}: {len(data[name]):,} records")

    imdb_master = data["imdb_master"]
    imdb_records = data["imdb_records"]
    tmdb_master = data["tmdb_master"]
    tmdb_records = data["tmdb_records"]
    dbpedia = data["dbpedia"]

    # --------------------------------------------------------
    # 2. SOURCE INTEGRITY
    # --------------------------------------------------------

    print("\n[2] SOURCE INTEGRITY")

    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        print(f"{name}: {'PASS' if condition else 'FAIL'}")

    check("dbpedia_population_6112", len(dbpedia) == 6112)
    check("imdb_master_population_6112", len(imdb_master) == 6112)
    check("tmdb_master_population_6112", len(tmdb_master) == 6112)

    check(
        "dbpedia_entity_ids_unique",
        dbpedia.movie_entity_id.is_unique
    )
    check(
        "imdb_master_entity_ids_unique",
        imdb_master.movie_entity_id.is_unique
    )
    check(
        "tmdb_master_entity_ids_unique",
        tmdb_master.movie_entity_id.is_unique
    )

    check(
        "imdb_record_population_577",
        len(imdb_records) == 577
    )
    check(
        "tmdb_record_population_582",
        len(tmdb_records) == 582
    )

    check(
        "all_source_entity_ids_nonblank",
        all(
            df.movie_entity_id.ne("").all()
            for df in [
                dbpedia, imdb_master, tmdb_master,
                imdb_records, tmdb_records
            ]
        )
    )

    if not all(checks.values()):
        raise RuntimeError("Source integrity validation failed.")

    # --------------------------------------------------------
    # 3. ENTITY RECONCILIATION
    # --------------------------------------------------------

    print("\n[3] ENTITY RECONCILIATION")

    base = dbpedia[
        ["movie_entity_id", "source_uri"]
    ].copy()

    imdb_input = imdb_master.rename(
        columns={"source_uri": "source_uri_imdb"}
    )

    tmdb_input = tmdb_master.rename(
        columns={"source_uri": "source_uri_tmdb"}
    )

    base = base.merge(
        imdb_input,
        on="movie_entity_id",
        how="left",
        validate="one_to_one"
    )

    base = base.merge(
        tmdb_input,
        on="movie_entity_id",
        how="left",
        validate="one_to_one"
    )

    base["imdb_source_uri_match"] = (
        base["source_uri"] == base["source_uri_imdb"]
    )

    base["tmdb_source_uri_match"] = (
        base["source_uri"] == base["source_uri_tmdb"]
    )

    check(
        "integrated_population_6112",
        len(base) == 6112
    )

    check(
        "integrated_entity_ids_unique",
        base.movie_entity_id.is_unique
    )

    check(
        "all_imdb_uris_preserved",
        base["imdb_source_uri_match"].all()
    )

    check(
        "all_tmdb_uris_preserved",
        base["tmdb_source_uri_match"].all()
    )

    if not all(checks.values()):
        raise RuntimeError("Entity reconciliation failed.")

    # --------------------------------------------------------
    # 4. RECORD-LEVEL EVIDENCE NORMALIZATION
    # --------------------------------------------------------

    print("\n[4] RECORD-LEVEL EVIDENCE INTEGRATION")

    normalized = []

    def add_records(df, source, identifier_col,
                    title_col, year_col, assessment_col):

        for index, row in df.iterrows():

            raw = row.to_dict()

            identifier = row.get(identifier_col, "")
            title = row.get(title_col, "")
            year = row.get(year_col, "")
            assessment = row.get(assessment_col, "")

            normalized.append({
                "record_key": f"{source}_{index + 1:06d}",
                "movie_entity_id": row["movie_entity_id"],
                "source_uri": row.get("source_uri", ""),
                "evidence_source": source,
                "identifier": identifier,
                "evidence_title": title,
                "evidence_year": year,
                "record_assessment": assessment,
                "raw_source_record": json.dumps(
                    raw, ensure_ascii=False
                ),
                "automatic_identity_verification": False,
                "automatic_entity_merging": False
            })

    add_records(
        imdb_records,
        "IMDb",
        "imdb_id",
        "imdb_title",
        "imdb_start_year",
        "record_evidence_assessment"
    )

    add_records(
        tmdb_records,
        "TMDb",
        "tmdb_id",
        "tmdb_title",
        "tmdb_year",
        "record_evidence_assessment"
    )

    record_master = pd.DataFrame(normalized)

    record_master.to_csv(
        OUT / "cross_source_identity_record_level.csv",
        index=False,
        encoding="utf-8-sig"
    )

    check(
        "record_population_1159",
        len(record_master) == 577 + 582
    )

    check(
        "all_imdb_records_preserved",
        (record_master.evidence_source == "IMDb").sum() == 577
    )

    check(
        "all_tmdb_records_preserved",
        (record_master.evidence_source == "TMDb").sum() == 582
    )

    check(
        "record_keys_unique",
        record_master.record_key.is_unique
    )

    if not all(checks.values()):
        raise RuntimeError("Record-level validation failed.")

    # --------------------------------------------------------
    # 5. ENTITY-LEVEL EVIDENCE AGGREGATION
    # --------------------------------------------------------

    print("\n[5] ENTITY-LEVEL EVIDENCE AGGREGATION")

    imdb_group = {
        key: group
        for key, group in imdb_records.groupby("movie_entity_id")
    }

    tmdb_group = {
        key: group
        for key, group in tmdb_records.groupby("movie_entity_id")
    }

    final_assessments = []

    for _, row in base.iterrows():

        entity_id = row["movie_entity_id"]

        ir = imdb_group.get(entity_id, pd.DataFrame())
        tr = tmdb_group.get(entity_id, pd.DataFrame())

        imdb_assessments = (
            ir["record_evidence_assessment"].tolist()
            if len(ir) else []
        )

        tmdb_assessments = (
            tr["record_evidence_assessment"].tolist()
            if len(tr) else []
        )

        imdb_strong = sum(
            x in (
                "STRONG_MOVIE_CORROBORATION",
                "STRONG_SHORT_CORROBORATION"
            )
            for x in imdb_assessments
        )

        tmdb_strong = sum(
            x == "STRONG_TMDb_TITLE_YEAR_CORROBORATION"
            for x in tmdb_assessments
        )

        conflicts = sum(
            x == "YEAR_CONFLICT_REVIEW"
            for x in imdb_assessments + tmdb_assessments
        )

        media_review = any(
            x in (
                "NON_FEATURE_VIDEO_REVIEW",
                "TV_EPISODE_REVIEW",
                "TV_MINISERIES_REVIEW",
                "TV_MOVIE_REVIEW",
                "TV_SERIES_REVIEW",
                "TV_SPECIAL_REVIEW"
            )
            for x in imdb_assessments + tmdb_assessments
        )

        granularity_values = [
            str(row.get(column, "")).strip()
            for column in (
                "granularity_assessment_x",
                "granularity_assessment_y",
                "granularity_assessment"
            )
        ]

        granularity_review = any(
            value in (
                "FILM_SERIES_ENTITY_REVIEW",
                "MULTIPLE_ADAPTATIONS_OR_RELATED_FILMS_REVIEW"
            )
            for value in granularity_values
        )

        granularity = "; ".join(
            unique_values([
                value for value in granularity_values if value
            ])
        )

        identifier_review = (
            "CARDINALITY_REVIEW" in " ".join(imdb_assessments)
            or "MULTIPLE_TMDB_RESULTS_REVIEW" in " ".join(
                tmdb_assessments
            )
        )

        has_imdb = len(ir) > 0
        has_tmdb = any(
            str(x).strip()
            for x in (
                tr["tmdb_id"].tolist()
                if len(tr) and "tmdb_id" in tr.columns
                else []
            )
        )

        if conflicts or identifier_review:
            assessment = "CONFLICT_REQUIRES_REVIEW"

        elif granularity_review:
            assessment = "GRANULARITY_REQUIRES_REVIEW"

        elif media_review:
            assessment = "MEDIA_TYPE_REQUIRES_REVIEW"

        elif imdb_strong > 0 and tmdb_strong > 0:
            assessment = "MULTI_SOURCE_CORROBORATION_CANDIDATE"

        elif imdb_strong > 0:
            assessment = "STRONG_IMDB_ONLY_CORROBORATION"

        elif tmdb_strong > 0:
            assessment = "STRONG_TMDB_ONLY_CORROBORATION"

        elif has_imdb or has_tmdb:
            assessment = "PARTIAL_CORROBORATION"

        else:
            assessment = "UNRESOLVED_INSUFFICIENT_EVIDENCE"

        final_assessments.append({
            "movie_entity_id": entity_id,
            "imdb_record_count": len(ir),
            "tmdb_record_count": len(tr),
            "imdb_strong_record_count": imdb_strong,
            "tmdb_strong_record_count": tmdb_strong,
            "year_conflict_count": conflicts,
            "media_type_review": media_review,
            "granularity_assessment": granularity,
            "imdb_record_assessments": json.dumps(
                imdb_assessments
            ),
            "tmdb_record_assessments": json.dumps(
                tmdb_assessments
            ),
            "final_identity_evidence_assessment": assessment,
            "identity_verified": False,
            "automatic_entity_merging": False
        })

    assessment_df = pd.DataFrame(final_assessments)

    master = base.merge(
        assessment_df,
        on="movie_entity_id",
        how="left",
        validate="one_to_one"
    )

    # Explicitly preserve conservative verification policy.
    master["identity_verified"] = False
    master["automatic_entity_merging"] = False

    # --------------------------------------------------------
    # 6. EXCEPTION REGISTER
    # --------------------------------------------------------

    print("\n[6] EXCEPTION REGISTER")

    exceptions = master[
        master.final_identity_evidence_assessment.isin([
            "CONFLICT_REQUIRES_REVIEW",
            "GRANULARITY_REQUIRES_REVIEW",
            "MEDIA_TYPE_REQUIRES_REVIEW"
        ])
    ].copy()

    # --------------------------------------------------------
    # 7. FINAL VALIDATION
    # --------------------------------------------------------

    print("\n[7] FINAL VALIDATION")

    final_checks = {
        "master_population_6112": len(master) == 6112,
        "master_entity_ids_unique": master.movie_entity_id.is_unique,
        "record_population_1159": len(record_master) == 1159,
        "all_entities_preserved": set(
            master.movie_entity_id
        ) == set(dbpedia.movie_entity_id),
        "all_imdb_rows_preserved": (
            record_master.evidence_source == "IMDb"
        ).sum() == 577,
        "all_tmdb_rows_preserved": (
            record_master.evidence_source == "TMDb"
        ).sum() == 582,
        "no_automatic_verification": not master.identity_verified.any(),
        "no_automatic_merging": not master.automatic_entity_merging.any(),
        "all_uris_nonblank": master.source_uri.ne("").all()
    }

    for name, passed in final_checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    if not all(final_checks.values()):
        raise RuntimeError("Final validation failed.")

    # --------------------------------------------------------
    # 8. EXPORT
    # --------------------------------------------------------

    print("\n[8] EXPORTING RESULTS")

    master.to_csv(
        OUT / "cross_source_movie_identity_master.csv",
        index=False,
        encoding="utf-8-sig"
    )

    record_master.to_csv(
        OUT / "cross_source_identity_record_level.csv",
        index=False,
        encoding="utf-8-sig"
    )

    exceptions.to_csv(
        OUT / "cross_source_identity_exceptions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    assessment_summary = (
        master["final_identity_evidence_assessment"]
        .value_counts()
        .to_dict()
    )

    source_hashes = {
        name: sha256(path)
        for name, path in SOURCES.items()
    }

    manifest = {
        "experiment": "21.17I",
        "name": "Cross-Source Movie Identity Evidence Integration",
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "sources": {
            name: str(path.relative_to(ROOT))
            for name, path in SOURCES.items()
        },
        "source_sha256": source_hashes,
        "counts": {
            "dbpedia_entities": len(dbpedia),
            "imdb_record_rows": len(imdb_records),
            "tmdb_record_rows": len(tmdb_records),
            "integrated_record_rows": len(record_master),
            "integrated_entities": len(master),
            "exception_entities": len(exceptions)
        },
        "assessment_distribution": assessment_summary,
        "validation": {
            **checks,
            **final_checks
        },
        "methodology": {
            "entity_matching": "Exact movie_entity_id",
            "uri_validation": "Exact source URI comparison",
            "record_preservation": "All source records retained",
            "tmdb_imdb_dependency": (
                "TMDb lookups originated from IMDb identifiers; "
                "not treated as fully independent proof."
            ),
            "automatic_identity_verification": False,
            "automatic_entity_merging": False
        },
        "completion_status": "COMPLETED_ALL_VALIDATIONS_PASSED"
    }

    with open(
        OUT / "experiment_21_17I_manifest.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            manifest,
            f,
            indent=4,
            ensure_ascii=False,
            default=lambda value: (
                value.item()
                if hasattr(value, "item")
                else str(value)
            )
        )

    with open(
        OUT / "reproducibility_report.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write("EXPERIMENT 21.17I\n")
        f.write("CROSS-SOURCE IDENTITY EVIDENCE INTEGRATION\n\n")
        f.write(f"Entities: {len(master)}\n")
        f.write(f"IMDb records: {len(imdb_records)}\n")
        f.write(f"TMDb records: {len(tmdb_records)}\n")
        f.write(f"Combined records: {len(record_master)}\n\n")

        for name, passed in {
            **checks,
            **final_checks
        }.items():
            f.write(
                f"{name}: {'PASS' if passed else 'FAIL'}\n"
            )

        f.write("\nNo automatic identity verification.\n")
        f.write("No automatic entity merging.\n")

    print("\n" + "=" * 75)
    print("EXPERIMENT 21.17I COMPLETED")
    print("=" * 75)

    print("Integrated entities:", len(master))
    print("IMDb evidence rows:", len(imdb_records))
    print("TMDb evidence rows:", len(tmdb_records))
    print("Combined evidence rows:", len(record_master))
    print("Exceptions:", len(exceptions))
    print("Output:", OUT)


if __name__ == "__main__":
    main()
