#!/usr/bin/env python
"""Experiment 22.35B: target-blind KG evidence aligned to KGSF validation Top-K.

Reuses the project's original load_graphs(), build_index(), and retrieve()
functions without changing graph construction or retrieval semantics. Context is
read from the native validation recommendation dataloader; target IDs are used
only to verify event alignment and are never passed to retrieval().

Default inputs are the 22.35A validation export. Outputs are written to a new
validation-only directory; existing independent-test artifacts are untouched.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from calibrate_and_evaluate import initialize, load_catalog
from evaluate_kg_retrieval import load_graphs, build_index, retrieve


def parse_ids(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
        if not isinstance(parsed, (list, tuple)):
            raise TypeError("Candidate field must encode a list.")
        return [int(x) for x in parsed]
    raise TypeError(f"Unsupported candidate ID format: {type(value)}")


def validate_candidate_ids(candidate_ids, movie_set, event_id, top_k):
    if len(candidate_ids) != top_k:
        raise RuntimeError(
            f"Expected {top_k} candidates at event {event_id}; found {len(candidate_ids)}"
        )
    if len(set(candidate_ids)) != top_k:
        raise RuntimeError(f"Duplicate candidate IDs at event {event_id}")
    outside = set(candidate_ids) - movie_set
    if outside:
        raise RuntimeError(
            f"Candidates outside frozen catalog at event {event_id}: {list(outside)[:5]}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/crs/kgsf/redial.yaml")
    parser.add_argument("--catalog", default="catalog_audit_corrected.pkl")
    parser.add_argument(
        "--mapping",
        default="results_corrected/experiment_22_35/kgsf_validation/kgsf_validation_event_manifest.csv",
    )
    parser.add_argument(
        "--kgsf",
        default="results_corrected/experiment_22_35/kgsf_validation/kgsf_validation_top500.csv",
    )
    parser.add_argument("--raw-dir", default="data/dataset/redial/nltk")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--hub-cutoff", type=int, default=500)
    parser.add_argument(
        "--out-dir", default="results_corrected/experiment_22_35/kg_validation"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1 or args.top_k < 1 or args.hub_cutoff < 1:
        raise ValueError("batch-size, top-k, and hub-cutoff must be positive")

    required_files = [
        Path(args.config), Path(args.catalog), Path(args.mapping), Path(args.kgsf),
        Path(args.raw_dir) / "dbpedia_subkg.json",
    ]
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(path)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    output_names = [
        "kgsf_validation_kg_ranks.csv",
        "kgsf_validation_kg_evidence_summary.json",
        "validation_checks.json",
        "experiment_manifest.json",
    ]
    existing = [name for name in output_names if (out / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing 22.35B outputs: {existing}. "
            "Choose another --out-dir or pass --overwrite explicitly."
        )

    print("=" * 85)
    print("EXPERIMENT 22.35B — KGSF-ALIGNED VALIDATION KG EVIDENCE")
    print("=" * 85)

    # Use the existing initializer and frozen checkpoint infrastructure.
    system, valid, _test, side_data = initialize(args.config, args.gpu)
    catalog = load_catalog(args.catalog, side_data["item_entity_ids"], system.device)
    movie_ids = [int(x) for x in catalog["movie_ids"]]
    movie_set = set(movie_ids)
    if len(movie_ids) != 6924 or len(movie_set) != 6924:
        raise RuntimeError("Frozen catalog validation failed; expected 6,924 unique movies.")

    mapping = pd.read_csv(args.mapping)
    kgsf = pd.read_csv(args.kgsf)
    required_mapping = {"event_id", "conv_id", "target_movie_id"}
    required_kgsf = {"event_id", "conv_id", "target_movie_id", "top_movie_ids"}
    if not required_mapping.issubset(mapping.columns):
        raise RuntimeError(f"Invalid event manifest schema; missing {required_mapping - set(mapping.columns)}")
    if not required_kgsf.issubset(kgsf.columns):
        raise RuntimeError(f"Invalid KGSF Top-K schema; missing {required_kgsf - set(kgsf.columns)}")

    mapping = mapping.sort_values("event_id").reset_index(drop=True)
    kgsf = kgsf.sort_values("event_id").reset_index(drop=True)
    for col in ("event_id", "conv_id", "target_movie_id"):
        mapping[col] = pd.to_numeric(mapping[col], errors="raise").astype("int64")
        kgsf[col] = pd.to_numeric(kgsf[col], errors="raise").astype("int64")
    if mapping.empty or len(mapping) != len(kgsf):
        raise RuntimeError(f"Event count mismatch: mapping={len(mapping)}, KGSF={len(kgsf)}")
    for label, frame in (("mapping", mapping), ("KGSF", kgsf)):
        if frame.event_id.duplicated().any():
            raise RuntimeError(f"Duplicate event IDs in {label}")
    for col in ("event_id", "conv_id", "target_movie_id"):
        if not np.array_equal(mapping[col].to_numpy(), kgsf[col].to_numpy()):
            raise RuntimeError(f"Validation {col} mismatch between manifest and KGSF Top-K")

    candidate_lists = []
    for row in kgsf.itertuples(index=False):
        ids = parse_ids(row.top_movie_ids)
        validate_candidate_ids(ids, movie_set, int(row.event_id), args.top_k)
        candidate_lists.append(ids)

    print(f"Validation events: {len(mapping)}")
    print(f"Unique conversations: {mapping.conv_id.nunique()}")
    print("Manifest / KGSF event, conversation and target alignment: PASS")
    print("Candidate count, uniqueness and catalog membership: PASS")

    # Reuse original graph construction and retrieval infrastructure unchanged.
    graphs = load_graphs(side_data, Path(args.raw_dir))
    indexes = {
        name: build_index(adjacency, movie_ids, args.hub_cutoff)
        for name, adjacency in graphs.items()
    }
    print("Graph variants:", list(indexes))
    print("Hub cutoff:", args.hub_cutoff)

    padding_id = int(valid.pad_entity_idx)
    records = []
    event_counter = 0
    observed_targets = []

    for batch in tqdm(
        valid.get_rec_data(batch_size=args.batch_size, shuffle=False),
        desc="Validation KG retrieval",
    ):
        context_batch = batch[0]
        targets = batch[-1]
        contexts = context_batch.tolist()
        target_values = targets.tolist()

        for context_row, target in zip(contexts, target_values):
            if event_counter >= len(mapping):
                raise RuntimeError("More validation dataloader events than manifest rows.")
            expected = mapping.iloc[event_counter]
            event_id = int(expected.event_id)
            target = int(target)
            observed_targets.append(target)
            if target != int(expected.target_movie_id):
                raise RuntimeError(
                    f"Validation target mismatch at event {event_id}: "
                    f"loader={target}, manifest={expected.target_movie_id}"
                )

            candidate_ids = candidate_lists[event_counter]
            context = [int(entity) for entity in context_row if int(entity) != padding_id]

            # Target is intentionally not passed to retrieve().
            retrieved = {
                name: retrieve(context, index)
                for name, index in indexes.items()
            }
            rank_maps = {
                name: {movie_id: rank for rank, movie_id in enumerate(ranked, start=1)}
                for name, ranked in retrieved.items()
            }

            row = {
                "event_id": event_id,
                "conv_id": int(expected.conv_id),
                "target_movie_id": target,
                "context_entity_count": len(set(context)),
                "candidate_count": len(candidate_ids),
                "candidate_movie_ids": json.dumps(candidate_ids),
            }
            for name in indexes:
                ranks = [rank_maps[name].get(movie, 0) for movie in candidate_ids]
                if len(ranks) != args.top_k:
                    raise RuntimeError(f"KG rank alignment failure at event {event_id}")
                row[f"{name}_retrieved_count"] = len(retrieved[name])
                row[f"{name}_candidate_kg_ranks"] = json.dumps(ranks)
                row[f"{name}_retrieved_movie_ids"] = json.dumps(retrieved[name])
            records.append(row)
            event_counter += 1

    if event_counter != len(mapping):
        raise RuntimeError(
            f"Validation event count mismatch: processed={event_counter}, manifest={len(mapping)}"
        )
    if observed_targets != mapping.target_movie_id.astype(int).tolist():
        raise RuntimeError("Complete validation target sequence differs from manifest.")

    result = pd.DataFrame(records).sort_values("event_id").reset_index(drop=True)
    if not np.array_equal(result.event_id.to_numpy(), mapping.event_id.to_numpy()):
        raise RuntimeError("Output event ordering differs from validation manifest.")
    if not np.array_equal(result.conv_id.to_numpy(), mapping.conv_id.to_numpy()):
        raise RuntimeError("Output conversation IDs differ from validation manifest.")
    if not np.array_equal(result.target_movie_id.to_numpy(), mapping.target_movie_id.to_numpy()):
        raise RuntimeError("Output targets differ from validation manifest.")

    checks = {
        "experiment": "22.35B",
        "split": "validation",
        "events": len(result),
        "unique_events": int(result.event_id.nunique()),
        "unique_conversations": int(result.conv_id.nunique()),
        "duplicate_event_ids": int(result.event_id.duplicated().sum()),
        "candidate_count_per_event": args.top_k,
        "all_candidate_counts_valid": bool((result.candidate_count == args.top_k).all()),
        "candidate_source": "22.35A frozen KGSF validation Top-K",
        "event_conv_target_alignment": True,
        "native_validation_target_sequence_matches_manifest": True,
        "target_used_for_retrieval": False,
        "training_performed": False,
        "parameter_search_performed": False,
        "independent_test_evaluated": False,
        "hub_cutoff": args.hub_cutoff,
        "graph_variants": list(indexes),
    }
    for name in indexes:
        rank_arrays = result[f"{name}_candidate_kg_ranks"].map(json.loads)
        retrieved_arrays = result[f"{name}_retrieved_movie_ids"].map(json.loads)
        candidate_arrays = result.candidate_movie_ids.map(json.loads)
        checks[f"{name}_rank_lengths_valid"] = bool(rank_arrays.map(len).eq(args.top_k).all())
        checks[f"{name}_candidate_lengths_valid"] = bool(candidate_arrays.map(len).eq(args.top_k).all())
        checks[f"{name}_retrieved_count_consistent"] = bool(
            (result[f"{name}_retrieved_count"] == retrieved_arrays.map(len)).all()
        )
        checks[f"{name}_positive_rank_count"] = int(sum(sum(rank > 0 for rank in row) for row in rank_arrays))
        checks[f"{name}_zero_rank_count"] = int(sum(sum(rank == 0 for rank in row) for row in rank_arrays))
        # Verify each positive rank points back to the same movie in retrieved order.
        mismatches = 0
        for candidates, ranks, retrieved in zip(candidate_arrays, rank_arrays, retrieved_arrays):
            for movie_id, rank in zip(candidates, ranks):
                if rank > 0 and (rank > len(retrieved) or int(retrieved[rank - 1]) != int(movie_id)):
                    mismatches += 1
        checks[f"{name}_rank_to_movie_mismatches"] = mismatches

    required_bool_checks = [
        value for key, value in checks.items()
        if isinstance(value, bool)
        and key not in {"target_used_for_retrieval", "training_performed", "parameter_search_performed", "independent_test_evaluated"}
    ]
    checks["overall_pass"] = bool(all(required_bool_checks)) and all(
        checks[f"{name}_rank_to_movie_mismatches"] == 0 for name in indexes
    )
    if not checks["overall_pass"]:
        raise RuntimeError(f"22.35B validation failed: {checks}")

    output_csv = out / "kgsf_validation_kg_ranks.csv"
    output_summary = out / "kgsf_validation_kg_evidence_summary.json"
    validation_path = out / "validation_checks.json"
    manifest_path = out / "experiment_manifest.json"
    result.to_csv(output_csv, index=False)

    summary = {
        "experiment": "22.35B_kgsf_validation_kg_evidence_export",
        "split": "validation",
        "events": len(result),
        "unique_conversations": int(result.conv_id.nunique()),
        "catalog_movies": len(movie_ids),
        "candidates_per_event": args.top_k,
        "candidate_source": "Frozen KGSF validation Top-K",
        "hub_cutoff": args.hub_cutoff,
        "graph_variants": {},
    }
    total = len(result) * args.top_k
    for name in indexes:
        positive = int(checks[f"{name}_positive_rank_count"])
        zero = int(checks[f"{name}_zero_rank_count"])
        summary["graph_variants"][name] = {
            "total_candidate_positions": total,
            "positive_rank_count": positive,
            "zero_rank_count": zero,
            "coverage_percentage": round(100.0 * positive / total, 4),
            "unretrieved_percentage": round(100.0 * zero / total, 4),
        }
    summary["validation"] = checks
    output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    validation_path.write_text(json.dumps(checks, indent=2), encoding="utf-8")

    experiment_manifest = {
        **summary,
        "config": args.config,
        "catalog": args.catalog,
        "event_manifest": args.mapping,
        "candidate_file": args.kgsf,
        "raw_graph_directory": args.raw_dir,
        "retrieval_function": "evaluate_kg_retrieval.retrieve",
        "index_function": "evaluate_kg_retrieval.build_index",
        "training_performed": False,
        "parameter_search_performed": False,
        "target_used_for_retrieval": False,
        "independent_test_evaluated": False,
        "files": {
            "kg_ranks": str(output_csv),
            "summary": str(output_summary),
            "validation": str(validation_path),
        },
    }
    manifest_path.write_text(json.dumps(experiment_manifest, indent=2), encoding="utf-8")

    print("\n========== EXPERIMENT 22.35B COMPLETE ==========")
    print(f"Events: {len(result)} | conversations: {result.conv_id.nunique()}")
    for name in indexes:
        item = summary["graph_variants"][name]
        print(
            f"{name}: positive={item['positive_rank_count']:,} "
            f"zero={item['zero_rank_count']:,} coverage={item['coverage_percentage']:.4f}%"
        )
    print("Validation:", "PASS" if checks["overall_pass"] else "FAIL")
    print(json.dumps(checks, indent=2))
    print(f"Output: {out.resolve()}")


if __name__ == "__main__":
    main()
