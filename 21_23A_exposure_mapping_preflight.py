
import json
from pathlib import Path
from collections import Counter

import pandas as pd

ROOT = Path(__file__).resolve().parent

MOVIE_FILE = ROOT / "data/dataset/redial/nltk/movie_ids.json"

PROFILE_FILE = (
    ROOT / "results_category_audit/experiment_21_22A/"
    "catalog_item_category_profiles.csv"
)

EVENT_FILE = (
    ROOT / "results_corrected/"
    "adaptive_m2_validation_events_clustered.csv"
)

print("=" * 75)
print("EXPERIMENT 21.23A: EXPOSURE MAPPING PREFLIGHT")
print("=" * 75)

# --------------------------------------------------
# 1. ORIGINAL CATALOG
# --------------------------------------------------

with open(MOVIE_FILE, encoding="utf-8") as f:
    movie_ids = json.load(f)

catalog_ids = set(map(int, movie_ids))

profiles = pd.read_csv(PROFILE_FILE)

profile_ids = set(
    profiles["movie_entity_id"].astype(int)
)

assert catalog_ids == profile_ids

print("\n[1] CATALOG")
print("Catalog IDs:", len(catalog_ids))
print("Profile IDs:", len(profile_ids))
print("Exact mapping:", catalog_ids == profile_ids)

# --------------------------------------------------
# 2. LOAD EVENTS
# --------------------------------------------------

events = pd.read_csv(EVENT_FILE)

methods = sorted(events["method"].unique())

print("\n[2] EVENTS")
print("Rows:", len(events))
print("Methods:", methods)
print("Conversations:", events["conv_id"].nunique())

# --------------------------------------------------
# 3. CHECK TOP-10 ITEM MEMBERSHIP
# --------------------------------------------------

item_counts = Counter()
invalid_ids = Counter()
group_counts = Counter()

for _, row in events.iterrows():

    movies = json.loads(row["top10_movie_ids"])
    groups = json.loads(row["top10_group_ids"])

    if len(movies) != 10 or len(groups) != 10:
        raise ValueError(
            f"Invalid top-10 structure: {row['method']}"
        )

    if len(set(movies)) != 10:
        raise ValueError(
            f"Duplicate recommended item: {row['method']}"
        )

    for movie in movies:

        movie = int(movie)
        item_counts[movie] += 1

        if movie not in catalog_ids:
            invalid_ids[movie] += 1

    for group in groups:
        group_counts[str(group)] += 1

print("\n[3] RECOMMENDED ITEM MAPPING")
print("Unique recommended IDs:", len(item_counts))
print("Valid unique IDs:", len(set(item_counts) & catalog_ids))
print("Invalid unique IDs:", len(set(invalid_ids)))
print("Total invalid recommendation occurrences:",
      sum(invalid_ids.values()))

if invalid_ids:
    print("Invalid examples:", invalid_ids.most_common(20))

# --------------------------------------------------
# 4. POPULARITY GROUP CODES
# --------------------------------------------------

print("\n[4] TOP-10 GROUP ID DISTRIBUTION")

for group, count in sorted(group_counts.items()):
    print(group, count)

# --------------------------------------------------
# 5. EVENT TARGET GROUP DISTRIBUTION
# --------------------------------------------------

print("\n[5] TARGET FREQUENCY GROUPS")

print(
    events.groupby(
        ["method", "frequency_group"]
    ).size().to_string()
)

print("\n[6] TARGET FREQUENCY SUMMARY")

print(
    events.groupby("frequency_group")[
        "target_frequency"
    ].agg(["count", "min", "median", "max"]).to_string()
)

# --------------------------------------------------
# 6. ITEM EXPOSURE COVERAGE
# --------------------------------------------------

print("\n[7] ITEM EXPOSURE")

print("Total recommendation slots:", sum(item_counts.values()))
print("Unique exposed items:", len(item_counts))
print("Catalog items never exposed:",
      len(catalog_ids - set(item_counts)))

print("\nPREFLIGHT COMPLETED")
