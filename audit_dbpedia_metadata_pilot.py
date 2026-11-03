
import json
import random
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT = Path(
    "results_category_audit/catalog_metadata_reconciliation.csv"
)

OUT = Path("results_category_audit/dbpedia_pilot")
OUT.mkdir(parents=True, exist_ok=True)

ENDPOINT = "https://dbpedia.org/sparql"
SAMPLE_SIZE = 25
SEED = 20261102
TIMEOUT = 90

USER_AGENT = "KG-FairCRS-Research-Audit/1.0"


# --------------------------------------------------
# Load and validate catalog
# --------------------------------------------------

df = pd.read_csv(INPUT, dtype=str)

uri_df = df[
    df["mapping_type"] == "dbpedia_resource_uri"
].copy()

assert len(uri_df) == 6112
assert uri_df["movie_entity_id"].is_unique
assert uri_df["resource_uri"].notna().all()

# Deterministic pilot sample
sample = uri_df.sample(
    n=SAMPLE_SIZE,
    random_state=SEED
).sort_values("movie_entity_id")

sample.to_csv(
    OUT / "pilot_selected_movies.csv",
    index=False
)

print("Selected URI records:", len(sample))


# --------------------------------------------------
# Construct SPARQL query
# --------------------------------------------------

resources = []

for uri in sample["resource_uri"]:
    uri = str(uri).strip()

    if not uri.startswith("http://dbpedia.org/resource/"):
        raise ValueError(f"Unexpected URI: {uri}")

    resources.append(f"<{uri}>")


values = " ".join(resources)

query = f"""
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT
    ?movie
    ?genre
    ?genreLabel
    ?mediaType
    ?mediaTypeLabel

WHERE {{

    VALUES ?movie {{ {values} }}

    OPTIONAL {{
        ?movie dbo:genre ?genre .

        OPTIONAL {{
            ?genre rdfs:label ?genreLabel .
            FILTER(LANG(?genreLabel) = "en")
        }}
    }}

    OPTIONAL {{
        ?movie rdf:type ?mediaType .

        OPTIONAL {{
            ?mediaType rdfs:label ?mediaTypeLabel .
            FILTER(LANG(?mediaTypeLabel) = "en")
        }}
    }}
}}
"""


# --------------------------------------------------
# Query DBpedia
# --------------------------------------------------

print("\nSubmitting DBpedia SPARQL pilot query...")

payload = urlencode({"query": query}).encode("utf-8")

request = Request(
    ENDPOINT,
    data=payload,
    headers={
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": USER_AGENT,
    },
    method="POST",
)

try:
    with urlopen(request, timeout=TIMEOUT) as response:
        response_body = response.read()
        response_headers = dict(response.headers.items())

except (HTTPError, URLError, TimeoutError) as exc:
    print("DBpedia request failed:", repr(exc))
    raise

data = json.loads(response_body.decode("utf-8"))

# Preserve original response and diagnostics.
(OUT / "pilot_response.json").write_text(
    json.dumps(data, indent=2),
    encoding="utf-8"
)

(OUT / "pilot_response_headers.json").write_text(
    json.dumps(response_headers, indent=2),
    encoding="utf-8"
)


# --------------------------------------------------
# Parse SPARQL bindings
# --------------------------------------------------

rows = []

for item in data["results"]["bindings"]:

    rows.append({
        "movie_uri": item.get("movie", {}).get("value"),
        "genre_uri": item.get("genre", {}).get("value"),
        "genre_label": item.get("genreLabel", {}).get("value"),
        "media_type_uri": item.get("mediaType", {}).get("value"),
        "media_type_label": item.get(
            "mediaTypeLabel", {}
        ).get("value"),
    })

result = pd.DataFrame(rows)

if result.empty:
    result = pd.DataFrame(columns=[
        "movie_uri",
        "genre_uri",
        "genre_label",
        "media_type_uri",
        "media_type_label",
    ])

# Attach original ReDial IDs.
result = result.merge(
    sample[["movie_entity_id", "resource_uri"]],
    left_on="movie_uri",
    right_on="resource_uri",
    how="left"
)

result.drop(
    columns=["resource_uri"],
    inplace=True
)

result = result.drop_duplicates()

result.to_csv(
    OUT / "dbpedia_pilot_raw.csv",
    index=False,
    encoding="utf-8-sig"
)


# --------------------------------------------------
# Summarize coverage
# --------------------------------------------------

genre_records = result[
    result["genre_uri"].notna()
]

type_records = result[
    result["media_type_uri"].notna()
]

genre_movies = genre_records[
    "movie_entity_id"
].nunique()

type_movies = type_records[
    "movie_entity_id"
].nunique()

print("\n=== DBPEDIA PILOT RESULTS ===")

print("Selected movies:", len(sample))
print("Returned rows:", len(result))

print("Movies with genre:", genre_movies)
print("Movies with media type:", type_movies)

print(
    "Genre coverage:",
    round(100 * genre_movies / len(sample), 2),
    "%"
)

print(
    "Media type coverage:",
    round(100 * type_movies / len(sample), 2),
    "%"
)

print("\n=== GENRE LABELS ===")

print(
    genre_records[
        ["genre_uri", "genre_label"]
    ].drop_duplicates().to_string(index=False)
)

print("\n=== MEDIA TYPE LABELS ===")

print(
    type_records[
        ["media_type_uri", "media_type_label"]
    ].drop_duplicates().to_string(index=False)
)

print("\n=== ENDPOINT DIAGNOSTICS ===")

for key in [
    "X-SPARQL-MaxRows",
    "X-SQL-State",
    "X-SQL-Message",
]:
    print(key, ":", response_headers.get(key, "Not reported"))

print("\nSaved pilot outputs in:", OUT.resolve())