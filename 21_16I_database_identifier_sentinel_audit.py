
from pathlib import Path
import pandas as pd
import json
import hashlib
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent

MERGED = ROOT / "external_validation/redial_reference/movies_merged.csv"
MENTIONS = ROOT / "external_validation/redial_reference/movies_with_mentions.csv"
LINKS = ROOT / "external_validation/movielens_32m/links.csv"
MOVIES = ROOT / "external_validation/movielens_32m/movies.csv"

OUT = ROOT / "results_category_audit/experiment_21_16I"
OUT.mkdir(parents=True, exist_ok=True)


def load(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():

    print("=" * 75)
    print("EXPERIMENT 21.16I: DATABASE IDENTIFIER SENTINEL AUDIT")
    print("=" * 75)

    merged = load(MERGED)
    mentions = load(MENTIONS)
    links = load(LINKS)
    movies = load(MOVIES)

    # Normalize source identifier strings without changing original data.
    merged["database_id_clean"] = merged["databaseId"].str.strip()
    merged["movielens_id_clean"] = merged["movielensId"].str.strip()

    mentions["database_id_clean"] = mentions["movieId"].str.strip()

    print("\n[1] SOURCE INVENTORY")
    print("Merged rows:", len(merged))
    print("Mention rows:", len(mentions))
    print("MovieLens links:", len(links))
    print("MovieLens movies:", len(movies))

    print("\n[2] DATABASE IDENTIFIER DISTRIBUTION")

    print(
        merged["database_id_clean"]
        .value_counts()
        .head(15)
        .to_string()
    )

    invalid_values = {"", "-1", "0", "nan", "none", "null"}

    merged["database_id_valid"] = (
        ~merged["database_id_clean"].str.lower().isin(invalid_values)
    )

    valid = merged[merged["database_id_valid"]].copy()
    invalid = merged[~merged["database_id_valid"]].copy()

    print("\nValid database-ID records:", len(valid))
    print("Invalid/sentinel records:", len(invalid))
    print("Distinct valid database IDs:", valid["database_id_clean"].nunique())
    print("Distinct invalid values:", invalid["database_id_clean"].nunique())

    print("\n[3] MENTION REFERENCE RECONCILIATION")

    valid_ids = set(valid["database_id_clean"])
    mention_ids = set(mentions["database_id_clean"])

    print("Valid merged IDs:", len(valid_ids))
    print("Mention IDs:", len(mention_ids))
    print("Intersection:", len(valid_ids & mention_ids))
    print("Valid merged only:", len(valid_ids - mention_ids))
    print("Mention only:", len(mention_ids - valid_ids))

    print("\n[4] SENTINEL MOVIELENS CROSS-REFERENCE")

    ml_links = links.rename(columns={
        "movieId": "movielens_id",
        "imdbId": "external_imdb_id",
        "tmdbId": "external_tmdb_id"
    })

    ml_movies = movies.rename(columns={
        "movieId": "movielens_id",
        "title": "external_movie_title",
        "genres": "external_movie_genres"
    })

    sentinel = invalid.copy()

    sentinel = sentinel.merge(
        ml_links,
        left_on="movielens_id_clean",
        right_on="movielens_id",
        how="left",
        validate="many_to_one"
    )

    sentinel = sentinel.merge(
        ml_movies,
        on="movielens_id",
        how="left",
        validate="many_to_one"
    )

    sentinel["links_found"] = sentinel["external_imdb_id"].ne("")
    sentinel["movie_metadata_found"] = sentinel["external_movie_title"].ne("")

    print("Sentinel records:", len(sentinel))
    print("With external links:", int(sentinel["links_found"].sum()))
    print("With movie metadata:", int(sentinel["movie_metadata_found"].sum()))

    print("\n[5] SAMPLE SENTINEL RECORDS")

    cols = [
        "index", "movieName", "databaseId", "movielensId",
        "external_movie_title", "external_imdb_id",
        "external_tmdb_id"
    ]

    print(sentinel[cols].head(20).to_string(index=False))

    print("\n[6] VALIDATION")

    checks = {
        "merged_population_preserved": len(merged) == 59946,
        "mention_population_preserved": len(mentions) == 6924,
        "sentinel_records_identified": len(invalid) > 0,
        "valid_database_ids_unique_population":
            len(valid_ids) == 6924,
        "all_valid_ids_have_mention_reference":
            valid_ids == mention_ids,
        "sentinel_records_separated":
            not invalid["database_id_valid"].any(),
    }

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    # Export only to the new experiment directory.
    sentinel.to_csv(
        OUT / "invalid_database_id_records.csv",
        index=False,
        encoding="utf-8-sig"
    )

    sentinel[
        [
            "databaseId", "movielensId", "movieName",
            "external_movie_title", "external_imdb_id",
            "external_tmdb_id", "links_found",
            "movie_metadata_found"
        ]
    ].to_csv(
        OUT / "sentinel_movielens_cross_reference.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary = {
        "experiment": "21.16I",
        "name": "Database Identifier Sentinel and Source Integrity Audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "merged_records": len(merged),
            "mention_records": len(mentions),
            "valid_database_records": len(valid),
            "invalid_sentinel_records": len(invalid),
            "valid_database_ids": len(valid_ids),
            "mention_ids": len(mention_ids),
            "valid_ids_with_mentions": len(valid_ids & mention_ids),
            "valid_ids_without_mentions": len(valid_ids - mention_ids),
            "sentinel_external_links": int(sentinel["links_found"].sum()),
            "sentinel_external_metadata": int(
                sentinel["movie_metadata_found"].sum()
            )
        },
        "validation": {
            k: bool(v) for k, v in checks.items()
        },
        "policy": {
            "minus_one_is_valid_database_identity": False,
            "automatic_merging": False,
            "source_files_modified": False,
            "identity_mapping_modified": False,
            "graph_modified": False,
            "model_training": False,
            "model_inference": False
        },
        "source_hashes": {
            "merged": sha256(MERGED),
            "mentions": sha256(MENTIONS),
            "links": sha256(LINKS),
            "movies": sha256(MOVIES)
        }
    }

    (OUT / "experiment_21_16I_manifest.json").write_text(
        json.dumps(summary, indent=4, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n[7] FINAL SUMMARY")
    print(json.dumps(summary["counts"], indent=4))
    print("\nOutput:", OUT)
    print("\nEXPERIMENT 21.16I FINISHED")


if __name__ == "__main__":
    main()
