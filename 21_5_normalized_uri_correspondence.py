
import csv
from pathlib import Path
from collections import Counter

INPUT = Path(
    "results_category_audit/entity_uri_provenance_v1/"
    "entity_uri_provenance.csv"
)

def normalize_uri(value):
    if value is None:
        return ""

    value = str(value).strip()

    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1].strip()

    return value


with INPUT.open(
    encoding="utf-8-sig",
    newline=""
) as f:

    rows = list(csv.DictReader(f))


counts = Counter()
mismatches = []

for row in rows:

    entity = normalize_uri(row["entity2id_uri"])
    movie = normalize_uri(row["movie_uri"])
    resource = normalize_uri(row["resource_uri"])

    counts["total"] += 1

    counts["entity_movie_match"] += (
        bool(entity) and entity == movie
    )

    counts["entity_resource_match"] += (
        bool(entity) and bool(resource)
        and entity == resource
    )

    counts["movie_resource_match"] += (
        bool(movie) and bool(resource)
        and movie == resource
    )

    counts["missing_resource_uri"] += (
        not bool(resource)
    )

    if entity and resource and entity != resource:
        mismatches.append({
            "movie_entity_id": row["movie_entity_id"],
            "entity_uri": entity,
            "resource_uri": resource
        })


print("=" * 70)
print("EXPERIMENT 21.5: NORMALIZED URI CORRESPONDENCE")
print("=" * 70)

for key, value in counts.items():
    print(f"{key}: {value}")

print("\nActual URI mismatches:", len(mismatches))

print("\nFirst 20 mismatches:")

for item in mismatches[:20]:
    print(item)

print("\nURI CORRESPONDENCE AUDIT COMPLETED")
