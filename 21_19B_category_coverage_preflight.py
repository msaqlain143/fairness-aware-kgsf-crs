
from pathlib import Path
import pandas as pd

BASE = Path("results_category_audit")

RAW = BASE / "experiment_21_18A2" / "movie_category_edges_raw.csv"

TAXONOMY = BASE / "dbpedia_pilot" / "canonical_category_inventory.csv"

MOVIES = BASE / "movie_entity_mapping.csv"

def norm(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    if x.startswith("<") and x.endswith(">"):
        return x[1:-1]
    return x

print("=" * 70)
print("EXPERIMENT 21.19B: CATEGORY COVERAGE PREFLIGHT")
print("=" * 70)

raw = pd.read_csv(RAW)
tax = pd.read_csv(TAXONOMY)
movies = pd.read_csv(MOVIES)

print("\n[1] SOURCE SCHEMAS")

for name, df in [
    ("RAW CATEGORY EDGES", raw),
    ("TAXONOMY", tax),
    ("MOVIE MAPPING", movies)
]:
    print(f"\n{name}")
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())
    print(df.head(2).to_string(index=False))

print("\n[2] ORIGINAL CATEGORY EDGE COVERAGE")

print("Unique movies:",
      raw["movie_uri"].map(norm).nunique())

print("Unique category values:",
      raw["value_uri_or_literal"].map(norm).nunique())

print("\nPredicate distribution:")
print(raw["predicate"].value_counts().to_string())

print("\n[3] TAXONOMY ANNOTATION STATUS")

print(tax["annotation_status"].value_counts(
    dropna=False
).to_string())

print("\nManual decisions:")
print(tax["manual_decision"].value_counts(
    dropna=False
).to_string())

print("\n[4] ACCEPTED GENRE COVERAGE")

accepted = tax[
    tax["manual_decision"].isin([
        "ACCEPT_GENRE",
        "ACCEPT_HYBRID"
    ])
]

raw_uris = set(
    raw["value_uri_or_literal"].map(norm)
)

accepted_uris = set(
    accepted["category_uri"].map(norm)
)

connected_accepted = raw_uris & accepted_uris

print("Accepted taxonomy categories:", len(accepted))
print("Accepted categories in raw edges:",
      len(connected_accepted))

print("\nConnected accepted categories:")

for uri in sorted(connected_accepted):
    label = accepted.loc[
        accepted["category_uri"].map(norm) == uri,
        "preferred_label_provisional"
    ].iloc[0]
    print(label, "|", uri)

print("\n[5] MOVIE MAPPING STRUCTURE")

print("Movie mapping records:", len(movies))

for col in movies.columns:
    print(col, "| unique:", movies[col].nunique(dropna=True))

print("\n[6] RAW MOVIE URI MATCH")

if "movie_uri" in movies.columns:
    raw_movie_uris = set(raw["movie_uri"].map(norm))
    catalog_movie_uris = set(movies["movie_uri"].map(norm))

    print("Raw movies:", len(raw_movie_uris))
    print("Catalog mapped movies:", len(catalog_movie_uris))
    print("Overlap:", len(raw_movie_uris & catalog_movie_uris))
    print("Raw movies outside mapping:",
          len(raw_movie_uris - catalog_movie_uris))
else:
    print("movie_uri column absent.")
    print("Do not assume a URI join is possible yet.")

print("\n[7] CATEGORY SOURCE COVERAGE")

print("Original category edges:", len(raw))
print("Unique movie-category pairs:",
      raw[["movie_uri", "predicate",
           "value_uri_or_literal"]].drop_duplicates().shape[0])

print("\nPREFLIGHT COMPLETED")
