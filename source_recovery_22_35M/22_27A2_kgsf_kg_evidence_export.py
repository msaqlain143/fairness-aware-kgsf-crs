
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
    """Parse movie IDs from JSON strings or Python lists."""

    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)

        return [int(x) for x in parsed]

    raise TypeError(f"Unsupported candidate ID format: {type(value)}")


def validate_candidate_ids(candidate_ids, movie_set, event_id):

    if len(candidate_ids) != 500:
        raise RuntimeError(
            f"Expected 500 candidates at event {event_id}, "
            f"found {len(candidate_ids)}"
        )

    if len(set(candidate_ids)) != 500:
        raise RuntimeError(
            f"Duplicate candidate IDs at event {event_id}"
        )

    if not set(candidate_ids).issubset(movie_set):
        raise RuntimeError(
            f"Candidate outside frozen catalog at event {event_id}"
        )


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
        "--kgsf",
        default="results_final_test_m2/experiment_22_27A/kgsf_independent_test_top500.csv"
    )

    parser.add_argument(
        "--raw-dir",
        default="data/dataset/redial/nltk"
    )

    parser.add_argument(
        "--out-dir",
        default="results_final_test_m2/experiment_22_27A2"
    )

    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hub-cutoff", type=int, default=500)

    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("EXPERIMENT 22.27A2 — KGSF-ALIGNED KG EVIDENCE EXPORT")
    print("=" * 85)

    # --------------------------------------------------
    # 0. Validate input files
    # --------------------------------------------------

    required_files = [
        args.config,
        args.catalog,
        args.mapping,
        args.kgsf,
        args.raw_dir + "/dbpedia_subkg.json",
    ]

    for file in required_files:

        if not Path(file).exists():
            raise FileNotFoundError(file)

        print("FOUND:", file)

    # --------------------------------------------------
    # 1. Load frozen KGSF infrastructure
    # --------------------------------------------------

    print("\n[1] Loading frozen KGSF infrastructure...")

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
    movie_set = set(movie_ids)

    if len(movie_ids) != 6924 or len(movie_set) != 6924:
        raise RuntimeError("Frozen catalog validation failed.")

    print("Catalog movies:", len(movie_ids))
    print("Device:", system.device)

    # --------------------------------------------------
    # 2. Load independent-test event mapping and KGSF
    # --------------------------------------------------

    print("\n[2] Loading independent-test events...")

    mapping = pd.read_csv(args.mapping)
    kgsf = pd.read_csv(args.kgsf)

    required_mapping = {
        "event_id",
        "conv_id",
        "target_movie_id",
    }

    required_kgsf = {
        "event_id",
        "conv_id",
        "target_movie_id",
        "top_movie_ids",
    }

    if not required_mapping.issubset(mapping.columns):
        raise RuntimeError("Invalid event mapping schema.")

    if not required_kgsf.issubset(kgsf.columns):
        raise RuntimeError("Invalid KGSF candidate export schema.")

    mapping = (
        mapping.sort_values("event_id")
        .reset_index(drop=True)
    )

    kgsf = (
        kgsf.sort_values("event_id")
        .reset_index(drop=True)
    )

    if len(mapping) != 4003 or len(kgsf) != 4003:
        raise RuntimeError("Expected exactly 4,003 events.")

    for name, frame in [
        ("Mapping", mapping),
        ("KGSF", kgsf),
    ]:

        if frame["event_id"].duplicated().any():
            raise RuntimeError(
                f"Duplicate event IDs in {name}"
            )

    if not np.array_equal(
        mapping.event_id.to_numpy(),
        kgsf.event_id.to_numpy()
    ):
        raise RuntimeError("Event IDs do not align.")

    if not np.array_equal(
        mapping.conv_id.to_numpy(),
        kgsf.conv_id.to_numpy()
    ):
        raise RuntimeError("Conversation IDs do not align.")

    if not np.array_equal(
        mapping.target_movie_id.to_numpy(),
        kgsf.target_movie_id.to_numpy()
    ):
        raise RuntimeError("Target movie IDs do not align.")

    print("Mapping events:", len(mapping))
    print("KGSF events:", len(kgsf))
    print("Event alignment: PASS")
    print("Conversation alignment: PASS")
    print("Target alignment: PASS")

    # --------------------------------------------------
    # 3. Build ORIGINAL frozen KG indexes
    # --------------------------------------------------

    print("\n[3] Building original KG indexes...")

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
    # 4. Generate KGSF-aligned KG evidence
    # --------------------------------------------------

    print("\n[4] Generating KGSF-aligned KG evidence...")

    padding_id = int(test.pad_entity_idx)

    records = []
    event_counter = 0

    for batch in tqdm(
        test.get_rec_data(
            batch_size=args.batch_size,
            shuffle=False
        ),
        desc="KGSF KG retrieval"
    ):

        context_batch = batch[0]
        targets = batch[-1]

        contexts = context_batch.tolist()
        targets = targets.tolist()

        for context_row, target in zip(contexts, targets):

            if event_counter >= len(mapping):
                raise RuntimeError(
                    "More test events than expected."
                )

            expected = mapping.iloc[event_counter]
            kgsf_row = kgsf.iloc[event_counter]

            event_id = int(expected.event_id)
            target = int(target)

            # ------------------------------------------
            # Validate original event alignment
            # ------------------------------------------

            if target != int(expected.target_movie_id):
                raise RuntimeError(
                    f"Target mismatch at event {event_id}"
                )

            candidate_ids = parse_ids(
                kgsf_row.top_movie_ids
            )

            validate_candidate_ids(
                candidate_ids,
                movie_set,
                event_id
            )

            # ------------------------------------------
            # Extract original conversation context
            # ------------------------------------------

            context = [
                int(entity)
                for entity in context_row
                if int(entity) != padding_id
            ]

            # ------------------------------------------
            # IMPORTANT:
            # Target is NOT used for KG retrieval.
            # The original retrieval method is preserved.
            # ------------------------------------------

            retrieved = {
                name: retrieve(context, index)
                for name, index in indexes.items()
            }

            # ------------------------------------------
            # Construct KG ranking maps
            # ------------------------------------------

            rank_maps = {
                name: {
                    movie_id: rank
                    for rank, movie_id in enumerate(
                        ranked,
                        start=1
                    )
                }
                for name, ranked in retrieved.items()
            }

            row = {
                "event_id": event_id,
                "conv_id": int(expected.conv_id),
                "target_movie_id": target,
                "context_entity_count": len(set(context)),
                "candidate_count": len(candidate_ids),

                # Exact KGSF candidate ordering
                "candidate_movie_ids": json.dumps(candidate_ids),
            }

            # ------------------------------------------
            # Candidate-aligned KG evidence
            # ------------------------------------------

            for name in indexes:

                ranks = [
                    rank_maps[name].get(movie, 0)
                    for movie in candidate_ids
                ]

                if len(ranks) != 500:
                    raise RuntimeError(
                        f"KG rank alignment failure: {event_id}"
                    )

                row[f"{name}_retrieved_count"] = len(
                    retrieved[name]
                )

                row[f"{name}_candidate_kg_ranks"] = json.dumps(
                    ranks
                )

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
    # 5. Create output dataframe
    # --------------------------------------------------

    print("\n[5] Constructing output dataset...")

    result = pd.DataFrame(records)

    result = result.sort_values(
        "event_id"
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 6. Validate candidate-aligned KG rankings
    # --------------------------------------------------

    print("\n[6] Validating KG evidence...")

    checks = {
        "experiment": "22_27A2",
        "split": "independent_test",
        "events": len(result),
        "unique_events": int(result.event_id.nunique()),
        "unique_conversations": int(result.conv_id.nunique()),
        "duplicate_event_ids": int(
            result.event_id.duplicated().sum()
        ),
        "all_candidate_counts_500": bool(
            (result.candidate_count == 500).all()
        ),
        "candidate_source": "Frozen KGSF Top-500",
        "target_used_for_retrieval": False,
        "training_performed": False,
        "parameter_search_performed": False,
        "hub_cutoff": args.hub_cutoff,
        "graph_variants": list(indexes),
    }

    for name in indexes:

        rank_arrays = result[
            f"{name}_candidate_kg_ranks"
        ].map(json.loads)

        candidate_arrays = result[
            "candidate_movie_ids"
        ].map(json.loads)

        checks[f"{name}_rank_lengths_valid"] = bool(
            rank_arrays.map(len).eq(500).all()
        )

        checks[f"{name}_candidate_lengths_valid"] = bool(
            candidate_arrays.map(len).eq(500).all()
        )

        checks[f"{name}_positive_rank_count"] = int(
            sum(
                sum(rank > 0 for rank in ranks)
                for ranks in rank_arrays
            )
        )

        checks[f"{name}_zero_rank_count"] = int(
            sum(
                sum(rank == 0 for rank in ranks)
                for ranks in rank_arrays
            )
        )

        checks[f"{name}_retrieved_count_consistent"] = bool(
            (
                result[f"{name}_retrieved_count"]
                ==
                result[
                    f"{name}_retrieved_movie_ids"
                ].map(lambda x: len(json.loads(x)))
            ).all()
        )

    checks["overall_pass"] = bool(
        checks["events"] == 4003
        and checks["unique_events"] == 4003
        and checks["duplicate_event_ids"] == 0
        and checks["all_candidate_counts_500"]
        and all(
            checks[f"{name}_rank_lengths_valid"]
            and checks[f"{name}_candidate_lengths_valid"]
            and checks[f"{name}_retrieved_count_consistent"]
            for name in indexes
        )
    )

    if not checks["overall_pass"]:
        raise RuntimeError(
            "KG export validation failed. "
            "Inspect validation_checks.json."
        )

    # --------------------------------------------------
    # 7. Save results
    # --------------------------------------------------

    print("\n[7] Saving output files...")

    output_csv = (
        out / "kgsf_independent_test_kg_ranks.csv"
    )

    output_summary = (
        out / "kgsf_kg_evidence_summary.json"
    )

    validation_path = (
        out / "validation_checks.json"
    )

    manifest_path = (
        out / "experiment_manifest.json"
    )

    result.to_csv(
        output_csv,
        index=False
    )

    # Aggregate KG evidence statistics
    summary = {
        "experiment": "22_27A2_kgsf_kg_evidence_export",
        "split": "independent_test",
        "events": len(result),
        "unique_conversations": int(result.conv_id.nunique()),
        "catalog_movies": len(movie_ids),
        "candidates_per_event": 500,
        "candidate_source": "KGSF frozen Top-500",
        "hub_cutoff": args.hub_cutoff,
        "graph_variants": {},
    }

    for name in indexes:

        ranks = result[
            f"{name}_candidate_kg_ranks"
        ].map(json.loads)

        positive = sum(
            sum(rank > 0 for rank in row)
            for row in ranks
        )

        zero = sum(
            sum(rank == 0 for rank in row)
            for row in ranks
        )

        total = len(result) * 500

        summary["graph_variants"][name] = {
            "total_candidate_positions": total,
            "positive_rank_count": positive,
            "zero_rank_count": zero,
            "coverage_percentage": round(
                100 * positive / total,
                4
            ),
            "unretrieved_percentage": round(
                100 * zero / total,
                4
            ),
        }

    summary["validation"] = checks

    with open(
        output_summary,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    with open(
        validation_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(checks, f, indent=2)

    manifest = {
        **summary,
        "config": args.config,
        "catalog": args.catalog,
        "event_mapping": args.mapping,
        "candidate_file": args.kgsf,
        "retrieval_function": "evaluate_kg_retrieval.retrieve",
        "index_function": "evaluate_kg_retrieval.build_index",
        "training_performed": False,
        "parameter_search_performed": False,
        "target_used_for_retrieval": False,
        "files": {
            "kg_ranks": str(output_csv),
            "summary": str(output_summary),
            "validation": str(validation_path),
        },
    }

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2)

    # --------------------------------------------------
    # 8. Final report
    # --------------------------------------------------

    print("\n" + "=" * 85)
    print("EXPERIMENT 22.27A2 COMPLETE")
    print("=" * 85)

    print(json.dumps(checks, indent=2))

    print("\nKG COVERAGE SUMMARY")

    for name, stats in summary["graph_variants"].items():

        print(f"\n{name}")

        for key, value in stats.items():
            print(f"{key}: {value}")

    print("\nOUTPUT DIRECTORY:")
    print(out.resolve())

    print("\nVALIDATION:", "PASS" if checks["overall_pass"] else "FAIL")


if __name__ == "__main__":
    main()



