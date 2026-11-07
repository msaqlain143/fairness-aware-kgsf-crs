from pathlib import Path
import pandas as pd

BASE = Path("results_category_audit")

SOURCES = {
    "IMDb": BASE / "experiment_21_17E" /
            "imdb_official_identifier_matches.csv",

    "TMDb": BASE / "experiment_21_17G" /
            "tmdb_identifier_cross_reference.csv"
}

OUT = BASE / "experiment_21_20B"
OUT.mkdir(parents=True, exist_ok=True)

all_duplicates = []

for source, path in SOURCES.items():

    df = pd.read_csv(path, dtype=str)

    print("\n" + "=" * 70)
    print(source, "DUPLICATE INVESTIGATION")
    print("=" * 70)

    duplicated = df[
        df.duplicated("movie_entity_id", keep=False)
    ].copy()

    duplicated = duplicated.sort_values(
        "movie_entity_id"
    )

    print("Total records:", len(df))
    print("Unique movies:", df.movie_entity_id.nunique())
    print("Duplicated records:", len(duplicated))
    print("Affected movies:",
          duplicated.movie_entity_id.nunique())

    for entity_id, group in duplicated.groupby(
        "movie_entity_id"
    ):

        print("\nENTITY:", entity_id)
        print("Candidate records:", len(group))

        if source == "IMDb":

            columns = [
                "imdb_id",
                "imdb_primary_title",
                "imdb_original_title",
                "imdb_start_year",
                "imdb_genres",
                "title_comparison",
                "year_comparison"
            ]

        else:

            columns = [
                "dbpedia_imdb_id",
                "tmdb_ids",
                "tmdb_results",
                "identity_evidence_assessment",
                "automatic_identity_verification"
            ]

        available = [
            c for c in columns if c in group.columns
        ]

        print(group[available].to_string(index=False))

        group.insert(0, "source", source)
        all_duplicates.append(group)

if all_duplicates:

    result = pd.concat(
        all_duplicates,
        ignore_index=True
    )

    result.to_csv(
        OUT / "duplicate_candidate_records.csv",
        index=False
    )

print("\nInvestigation exported successfully.")