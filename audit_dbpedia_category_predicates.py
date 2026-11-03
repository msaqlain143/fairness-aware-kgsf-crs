
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE = Path("results_category_audit/dbpedia_pilot")

INPUT = BASE / "pilot_selected_movies.csv"
OUTPUT = BASE / "category_predicate_audit.csv"

ENDPOINT = "https://dbpedia.org/sparql"
TIMEOUT = 120

# --------------------------------------------------
# Load the exact same 25 pilot resources
# --------------------------------------------------

movies = pd.read_csv(INPUT, dtype=str)

assert len(movies) == 25
assert movies["resource_uri"].notna().all()

resources = " ".join(
    f"<{uri}>"
    for uri in movies["resource_uri"]
)

# --------------------------------------------------
# Test multiple documented category predicates
# --------------------------------------------------

query = f"""
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbp: <http://dbpedia.org/property/>
PREFIX schema: <http://schema.org/>
PREFIX dct: <http://purl.org/dc/terms/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT
    ?movie
    ?predicate
    ?value
    ?valueLabel
    ?valueType

WHERE {{

    VALUES ?movie {{ {resources} }}

    VALUES ?predicate {{
        dbo:genre
        dbp:genre
        schema:genre
        dct:subject
    }}

    ?movie ?predicate ?value .

    OPTIONAL {{
        ?value rdfs:label ?valueLabel .
        FILTER(LANG(?valueLabel) = "en")
    }}

    OPTIONAL {{
        ?value rdf:type ?valueType .
    }}
}}
"""

# --------------------------------------------------
# Execute query
# --------------------------------------------------

print("Testing four DBpedia predicates...")
print("Sample resources:", len(movies))

payload = urlencode({"query": query}).encode("utf-8")

request = Request(
    ENDPOINT,
    data=payload,
    headers={
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "KG-FairCRS-Category-Audit/1.0",
    },
    method="POST",
)

with urlopen(request, timeout=TIMEOUT) as response:
    raw = response.read()
    headers = dict(response.headers.items())

data = json.loads(raw.decode("utf-8"))

(BASE / "category_predicate_response.json").write_text(
    json.dumps(data, indent=2),
    encoding="utf-8"
)

# --------------------------------------------------
# Parse results
# --------------------------------------------------

rows = []

for item in data["results"]["bindings"]:

    rows.append({
        "movie_uri": item.get("movie", {}).get("value"),
        "predicate": item.get("predicate", {}).get("value"),
        "value_uri_or_literal": item.get("value", {}).get("value"),
        "value_type": item.get("value", {}).get("type"),
        "value_label": item.get("valueLabel", {}).get("value"),
        "value_entity_type": item.get("valueType", {}).get("value"),
    })

result = pd.DataFrame(rows)

if result.empty:
    result = pd.DataFrame(columns=[
        "movie_uri",
        "predicate",
        "value_uri_or_literal",
        "value_type",
        "value_label",
        "value_entity_type",
    ])

# Attach ReDial identifiers
result = result.merge(
    movies[["movie_entity_id", "resource_uri"]],
    left_on="movie_uri",
    right_on="resource_uri",
    how="left"
)

result.drop(columns=["resource_uri"], inplace=True)

result.drop_duplicates(inplace=True)

result.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

# --------------------------------------------------
# Evaluate predicate coverage
# --------------------------------------------------

print("\n=== PREDICATE COVERAGE ===")

for predicate, group in result.groupby("predicate"):

    print("\nPredicate:", predicate)
    print("Relationships:", len(group))
    print("Unique movies:", group["movie_entity_id"].nunique())
    print("Unique values:",
          group["value_uri_or_literal"].nunique())

    print("Example values:")

    print(
        group[
            ["value_uri_or_literal", "value_label"]
        ].drop_duplicates().head(15).to_string(index=False)
    )

print("\n=== OVERALL SUMMARY ===")

print("Selected movies:", len(movies))
print("Total relationships:", len(result))
print("Movies with any category:", result["movie_entity_id"].nunique())

print("\n=== ENDPOINT DIAGNOSTICS ===")

for key in [
    "X-SPARQL-MaxRows",
    "X-SQL-State",
    "X-SQL-Message",
]:
    print(key, ":", headers.get(key, "Not reported"))

print("\nSaved:", OUTPUT)
