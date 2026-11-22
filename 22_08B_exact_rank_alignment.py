
import ast
import json
from pathlib import Path

import pandas as pd

INPUT = Path(
    "results_final_test_m2/experiment_22_27A2/"
    "kgsf_independent_test_kg_ranks.csv"
)

OUTPUT = Path("results_corrected/experiment_22_08B")
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)


def parse_array(value):
    if isinstance(value, list):
        return value
    return ast.literal_eval(value)


results = {
    "events": len(df),
    "R0_exact_rank_mismatches": 0,
    "R1_exact_rank_mismatches": 0,
    "R0_duplicate_retrieved_ids": 0,
    "R1_duplicate_retrieved_ids": 0,
    "duplicate_candidate_ids": 0,
}

examples = []

for _, row in df.iterrows():

    candidates = list(map(
        int, parse_array(row["candidate_movie_ids"])
    ))

    if len(set(candidates)) != len(candidates):
        results["duplicate_candidate_ids"] += 1

    for graph in ["R0_filtered", "R1_raw"]:

        ranks = list(map(
            int,
            parse_array(row[f"{graph}_candidate_kg_ranks"])
        ))

        retrieved = list(map(
            int,
            parse_array(row[f"{graph}_retrieved_movie_ids"])
        ))

        if len(set(retrieved)) != len(retrieved):
            results[f"{graph}_duplicate_retrieved_ids"] += 1

        if len(ranks) != len(candidates):
            raise RuntimeError(
                f"Invalid rank length: event {row.event_id}, {graph}"
            )

        if len(retrieved) != int(
            row[f"{graph}_retrieved_count"]
        ):
            raise RuntimeError(
                f"Retrieval count mismatch: {row.event_id}, {graph}"
            )

        # Reconstruct the original deterministic retrieval ranking.
        rank_map = {
            movie_id: rank
            for rank, movie_id in enumerate(
                retrieved, start=1
            )
        }

        expected = [
            rank_map.get(movie_id, 0)
            for movie_id in candidates
        ]

        if ranks != expected:

            results[f"{graph}_exact_rank_mismatches"] += 1

            if len(examples) < 10:
                examples.append({
                    "event_id": int(row.event_id),
                    "graph": graph,
                    "mismatch_positions": [
                        i for i, (a, b) in enumerate(
                            zip(ranks, expected)
                        ) if a != b
                    ][:10]
                })

summary = {
    **results,
    "exact_rank_alignment_pass": (
        results["R0_exact_rank_mismatches"] == 0
        and results["R1_exact_rank_mismatches"] == 0
        and results["R0_duplicate_retrieved_ids"] == 0
        and results["R1_duplicate_retrieved_ids"] == 0
        and results["duplicate_candidate_ids"] == 0
    ),
}

with open(
    OUTPUT / "exact_rank_alignment_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=4)

with open(
    OUTPUT / "mismatch_examples.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(examples, f, indent=4)

print("=" * 75)
print("EXPERIMENT 22.08B — EXACT RANK ALIGNMENT")
print("=" * 75)
print(json.dumps(summary, indent=4))
print("\nMismatch examples:")
print(json.dumps(examples, indent=4))
print("\nOutput:", OUTPUT.resolve())
