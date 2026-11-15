
"""
Experiment 22.24A
Independent-test KG candidate export.

- Reuses the frozen KG retrieval implementation.
- Does not train or modify checkpoints.
- Does not use targets for candidate generation.
- Exports R0 and R1 ranks for the frozen KBRD top-500 candidates.
"""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from calibrate_and_evaluate import initialize, load_catalog

from evaluate_kg_retrieval import (
    load_graphs,
    build_index,
    retrieve,
)


def parse_ids(value):
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return list(map(int, json.loads(value)))


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config/crs/kgsf/redial.yaml"
    )

    parser.add_argument(
        "--catalog",
        default="catalog_audit_corrected.pkl"
    )

    parser.add_argument(
        "--mapping",
        default="results_final_test_m2/m2_test_event_conversation_mapping.csv"
    )

    parser.add_argument(
        "--kbrd",
        default="results_final_test_m2/experiment_22_22/kbrd_independent_test_top500.csv"
    )

    parser.add_argument(
        "--raw-dir",
        default="data/dataset/redial/nltk"
    )

    parser.add_argument(
        "--out-dir",
        default="results_final_test_m2/experiment_22_24A"
    )

    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hub-cutoff", type=int, default=500)

    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 22.24A — INDEPENDENT TEST KG EXPORT")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load frozen KGSF infrastructure
    # --------------------------------------------------

    print("\nLoading frozen KGSF infrastructure...")

    system, valid, test, side_data = initialize(
        args.config,
        args.gpu
    )

    catalog = load_catalog(
        args.catalog,
        side_data["item_entity_ids"],
        system.device
    )

    movie_ids = list(map(int, catalog["movie_ids"]))

    if len(movie_ids) != 6924 or len(set(movie_ids)) != 6924:
        raise RuntimeError("Frozen catalog validation failed.")

    print("Catalog:", len(movie_ids))
    print("Device:", system.device)

    # --------------------------------------------------
    # 2. Load frozen independent-test event mapping
    # --------------------------------------------------

    mapping = pd.read_csv(args.mapping)
    kbrd = pd.read_csv(args.kbrd)

    required = {
        "event_id",
        "conv_id",
        "target_movie_id"
    }

    if not required.issubset(mapping.columns):
        raise ValueError("Invalid event mapping schema.")

    if not {"event_id", "conv_id", "top_movie_ids"}.issubset(kbrd.columns):
        raise ValueError("Invalid KBRD candidate export schema.")

    mapping = mapping.sort_values("event_id").reset_index(drop=True)
    kbrd = kbrd.sort_values("event_id").reset_index(drop=True)

    if len(mapping) != 4003 or len(kbrd) != 4003:
        raise RuntimeError("Expected exactly 4,003 test events.")

    if mapping["event_id"].duplicated().any():
        raise RuntimeError("Duplicate event IDs in mapping.")

    if not np.array_equal(
        mapping.event_id.to_numpy(),
        kbrd.event_id.to_numpy()
    ):
        raise RuntimeError("KBRD and event mapping IDs do not align.")

    # --------------------------------------------------
    # 3. Build both frozen KG indexes
    # --------------------------------------------------

    print("\nLoading KG graph variants...")

    graphs = load_graphs(
        side_data,
        Path(args.raw_dir)
    )

    indexes = {
        name: build_index(
            adjacency,
            movie_ids,
            args.hub_cutoff
        )
        for name, adjacency in graphs.items()
    }

    print("Graph variants:", list(indexes))
    print("Hub cutoff:", args.hub_cutoff)

    # --------------------------------------------------
    # 4. Generate target-blind KG retrieval rankings
    # --------------------------------------------------

    padding_id = int(test.pad_entity_idx)

    records = []
    event_counter = 0

    print("\nGenerating independent-test KG rankings...")

    for batch in tqdm(
        test.get_rec_data(
            batch_size=args.batch_size,
            shuffle=False
        ),
        desc="Independent-test KG retrieval"
    ):

        context_batch = batch[0]
        targets = batch[-1]

        contexts = context_batch.tolist()
        targets = targets.tolist()

        for context_row, target in zip(contexts, targets):

            if event_counter >= len(mapping):
                raise RuntimeError("More events than frozen mapping.")

            expected = mapping.iloc[event_counter]

            target = int(target)

            if target != int(expected.target_movie_id):
                raise RuntimeError(
                    f"Target alignment failure at event "
                    f"{expected.event_id}: "
                    f"KGSF={target}, mapping={expected.target_movie_id}"
                )

            context = [
                int(entity)
                for entity in context_row
                if int(entity) != padding_id
            ]

            # Target is deliberately excluded from retrieval.
            retrieved = {
                name: retrieve(context, index)
                for name, index in indexes.items()
            }

            candidate_ids = parse_ids(
                kbrd.iloc[event_counter].top_movie_ids
            )

            if len(candidate_ids) != 500:
                raise RuntimeError(
                    f"Invalid top-500 candidate count at event "
                    f"{expected.event_id}"
                )

            if len(set(candidate_ids)) != 500:
                raise RuntimeError("Duplicate KBRD candidate IDs.")

            if not set(candidate_ids).issubset(set(movie_ids)):
                raise RuntimeError("Candidate outside frozen catalog.")

            rank_maps = {
                name: {
                    movie_id: rank
                    for rank, movie_id in enumerate(ranked, start=1)
                }
                for name, ranked in retrieved.items()
            }

            row = {
                "event_id": int(expected.event_id),
                "conv_id": int(expected.conv_id),
                "target_movie_id": target,
                "context_entity_count": len(set(context)),
                "candidate_count": len(candidate_ids),
            }

            for name in indexes:

                row[f"{name}_retrieved_count"] = len(
                    retrieved[name]
                )

                row[f"{name}_candidate_kg_ranks"] = json.dumps([
                    rank_maps[name].get(movie, 0)
                    for movie in candidate_ids
                ])

                row[f"{name}_retrieved_movie_ids"] = json.dumps(
                    retrieved[name]
                )

            records.append(row)
            event_counter += 1

    if event_counter != 4003:
        raise RuntimeError(
            f"Expected 4003 events, processed {event_counter}"
        )

    # --------------------------------------------------
    # 5. Save candidate-aligned rankings
    # --------------------------------------------------

    result = pd.DataFrame(records)

    result.to_csv(
        out / "independent_test_kg_candidate_ranks.csv",
        index=False
    )

    # --------------------------------------------------
    # 6. Validate export
    # --------------------------------------------------

    checks = {
        "events": len(result),
        "unique_events": int(result.event_id.nunique()),
        "unique_conversations": int(result.conv_id.nunique()),
        "all_candidate_counts_500": bool(
            (result.candidate_count == 500).all()
        ),
        "target_alignment": True,
        "duplicate_event_ids": int(result.event_id.duplicated().sum()),
        "target_used_for_retrieval": False,
        "training_performed": False,
        "parameter_search": False,
        "graph_variants": list(indexes),
    }

    for name in indexes:

        rank_arrays = result[
            f"{name}_candidate_kg_ranks"
        ].map(json.loads)

        checks[f"{name}_rank_array_lengths_valid"] = bool(
            rank_arrays.map(len).eq(500).all()
        )

        checks[f"{name}_positive_rank_count"] = int(
            sum(
                sum(rank > 0 for rank in ranks)
                for ranks in rank_arrays
            )
        )

    checks["overall_pass"] = bool(
        checks["events"] == 4003
        and checks["unique_events"] == 4003
        and checks["duplicate_event_ids"] == 0
        and checks["all_candidate_counts_500"]
        and all(
            checks[f"{name}_rank_array_lengths_valid"]
            for name in indexes
        )
    )

    with open(
        out / "validation_checks.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(checks, f, indent=2)

    manifest = {
        "experiment": "22.24A_independent_test_kg_export",
        "split": "independent_test",
        "events": 4003,
        "catalog_items": 6924,
        "candidate_pool": 500,
        "hub_cutoff": args.hub_cutoff,
        "graph_variants": list(indexes),
        "target_used_for_ranking": False,
        "training_performed": False,
        "parameter_search": False,
        "validation": checks
    }

    with open(
        out / "experiment_manifest.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2)

    print("\n========== EXPERIMENT 22.24A COMPLETE ==========")
    print(json.dumps(checks, indent=2))
    print("\nOutput:", out.resolve())


if __name__ == "__main__":
    main()
