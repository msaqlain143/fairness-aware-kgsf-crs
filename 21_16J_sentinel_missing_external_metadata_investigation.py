
import pandas as pd
from pathlib import Path

BASE = Path("external_validation/movielens_32m")
RESULT = Path("results_category_audit/experiment_21_16I")

sentinel = pd.read_csv(
    RESULT / "sentinel_movielens_cross_reference.csv",
    dtype=str
).fillna("")

links = pd.read_csv(
    BASE / "links.csv",
    dtype=str
).fillna("")

movies = pd.read_csv(
    BASE / "movies.csv",
    dtype=str
).fillna("")

print("Sentinel columns:", sentinel.columns.tolist())

# Identify columns without assuming capitalization.
def find_col(df, candidates):
    lookup = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]
    raise KeyError(f"Missing expected column: {candidates}")

sid = find_col(sentinel, ["movielensId", "movielens_id"])
lid = find_col(links, ["movieId"])
mid = find_col(movies, ["movieId"])

imdb = find_col(links, ["imdbId"])
tmdb = find_col(links, ["tmdbId"])
title = find_col(movies, ["title"])

# Preserve source records.
sentinel["_movie_key"] = sentinel[sid].str.strip()
links["_movie_key"] = links[lid].str.strip()
movies["_movie_key"] = movies[mid].str.strip()

link_keys = set(links["_movie_key"])
movie_keys = set(movies["_movie_key"])

sentinel["link_record_found"] = (
    sentinel["_movie_key"].isin(link_keys)
)

sentinel["movie_record_found"] = (
    sentinel["_movie_key"].isin(movie_keys)
)

link_map = links.set_index("_movie_key")
movie_map = movies.set_index("_movie_key")

sentinel["imdb_reference"] = sentinel["_movie_key"].map(
    link_map[imdb]
).fillna("")

sentinel["tmdb_reference"] = sentinel["_movie_key"].map(
    link_map[tmdb]
).fillna("")

sentinel["external_title"] = sentinel["_movie_key"].map(
    movie_map[title]
).fillna("")

sentinel["metadata_status"] = "AVAILABLE"

sentinel.loc[
    ~sentinel["link_record_found"] &
    ~sentinel["movie_record_found"],
    "metadata_status"
] = "MISSING_BOTH_FILES"

sentinel.loc[
    ~sentinel["link_record_found"] &
    sentinel["movie_record_found"],
    "metadata_status"
] = "MISSING_LINK_RECORD"

sentinel.loc[
    sentinel["link_record_found"] &
    ~sentinel["movie_record_found"],
    "metadata_status"
] = "MISSING_MOVIE_RECORD"

sentinel["imdb_available"] = (
    sentinel["imdb_reference"].str.strip() != ""
)

sentinel["tmdb_available"] = (
    sentinel["tmdb_reference"].str.strip() != ""
)

print("\n===== MISSING METADATA INVESTIGATION =====")

print(
    sentinel["metadata_status"].value_counts(dropna=False)
)

print("\n===== IDENTIFIER COVERAGE =====")
print("IMDb:", int(sentinel["imdb_available"].sum()))
print("TMDb:", int(sentinel["tmdb_available"].sum()))

missing = sentinel[
    sentinel["metadata_status"] != "AVAILABLE"
].copy()

out = RESULT / "experiment_21_16J"
out.mkdir(parents=True, exist_ok=True)

missing.to_csv(
    out / "missing_external_metadata_records.csv",
    index=False,
    encoding="utf-8-sig"
)

sentinel.to_csv(
    out / "sentinel_metadata_reconciliation.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nMissing records exported:", len(missing))
print("Output directory:", out.resolve())
