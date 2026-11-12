
import pandas as pd
import json
import numpy as np

base = "results_corrected/adaptive_m2_validation_export22"

c = pd.read_csv(base + "_candidates.csv")
e = pd.read_csv(base + "_events.csv")

mapping = pd.read_csv(
    "results_corrected/m2_event_conversation_mapping.csv"
)

# Select the primary M2 baseline.
method = "A1_frequency_3"

c = c[c.method == method].copy()
e = e[e.method == method].copy()

assert c.event_id.is_unique
assert e.event_id.is_unique

x = c.merge(
    e[["event_id", "context_nonempty"]],
    on="event_id",
    validate="one_to_one"
)

x = x.merge(
    mapping[["event_id", "conv_id"]],
    on="event_id",
    validate="one_to_one"
)

print("========== KG RANK DISTRIBUTION ==========")

stats = []

for r in x.itertuples():
    ranks = np.asarray(
        json.loads(r.candidate_kg_ranks),
        dtype=int
    )

    stats.append({
        "event_id": r.event_id,
        "conv_id": r.conv_id,
        "context_nonempty": r.context_nonempty,
        "retrieved_candidates": int((ranks > 0).sum()),
        "max_rank": int(ranks.max()),
        "valid_rank_fraction": float((ranks > 0).mean())
    })

rs = pd.DataFrame(stats)

print(rs.groupby("context_nonempty")[
    ["retrieved_candidates", "max_rank", "valid_rank_fraction"]
].agg(["count", "mean", "median", "max"]).to_string())

print("\n========== RETRIEVAL COVERAGE ==========")

print(
    "Events with at least one KG-ranked candidate:",
    int((rs.retrieved_candidates > 0).sum())
)

print(
    "Events with no KG-ranked candidates:",
    int((rs.retrieved_candidates == 0).sum())
)

print("\n========== CATEGORY PROFILE EXAMPLES ==========")

p = pd.read_csv(
    "results_category_audit/experiment_21_22A/"
    "catalog_item_category_profiles.csv"
)

cols = [
    "movie_entity_id",
    "canonical_genres",
    "imdb_genre_evidence",
    "accepted_genre_evidence",
    "accepted_hybrid_evidence",
    "unvalidated_evidence",
    "identity_resolution_status",
    "category_availability"
]

print(
    p.loc[
        p.category_availability != "UNKNOWN",
        cols
    ].head(20).to_string(index=False)
)

print("\n========== CONVERSATION MAPPING ==========")

print("Events:", mapping.event_id.nunique())
print("Conversations:", mapping.conv_id.nunique())

print("\n========== FINAL CHECK ==========")

assert len(x) == 3010
assert len(rs) == 3010

print("PASS: 3,010 events aligned successfully.")