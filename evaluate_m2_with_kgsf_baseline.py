
"""Phase 5.5: held-out-test adaptive KGSF + KG rank fusion.

A0: fixed alpha=0.25.
A1: candidate-frequency-adaptive KG weight.
A2: frequency-adaptive weight multiplied by context KG coverage.

Ground-truth labels are used only for evaluation, never for ranking.
No model training, checkpoint changes, or test evaluation.

The event-level CSV stores actual top-10 movie IDs and corresponding
group IDs to support paired, conversation-clustered fairness analysis.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from calibrate_and_evaluate import initialize, load_catalog
from evaluate_kg_retrieval import load_graphs, build_index, retrieve


def gini(values):
    """Compute the Gini coefficient of a nonnegative exposure vector."""
    a = np.sort(np.asarray(values, dtype=np.float64))

    if len(a) == 0 or a.sum() == 0:
        return 0.0

    if np.any(a < 0):
        raise ValueError("Gini input must be nonnegative.")

    n = len(a)

    return float(
        2.0 * np.dot(np.arange(1, n + 1), a)
        / (n * a.sum())
        - (n + 1.0) / n
    )


def make_stats(n_movies):
    """Initialize ranking and exposure statistics for one method."""
    return {
        "events": 0,
        "hit10": 0,
        "hit50": 0,
        "ndcg10_sum": 0.0,
        "zero_n": 0,
        "zero_hit10": 0,
        "one_n": 0,
        "one_hit10": 0,
        "overlap_n": 0,
        "overlap_hit10": 0,
        "nonoverlap_n": 0,
        "nonoverlap_hit10": 0,
        "exposures": np.zeros(n_movies, dtype=np.int64),
        "group_counts": np.zeros(5, dtype=np.int64),
    }


def update_stats(
    stats,
    order,
    true_local,
    frequency,
    target_overlap,
    group_ids,
):
    """Update ranking and top-10 exposure statistics."""
    stats["events"] += 1

    positions = np.flatnonzero(order == true_local)
    rank = int(positions[0]) + 1 if len(positions) else None

    if rank is not None:
        if rank <= 50:
            stats["hit50"] += 1

        if rank <= 10:
            stats["hit10"] += 1
            stats["ndcg10_sum"] += 1.0 / np.log2(rank + 1)

    if frequency == 0:
        stats["zero_n"] += 1
        stats["zero_hit10"] += int(rank is not None and rank <= 10)

    if frequency == 1:
        stats["one_n"] += 1
        stats["one_hit10"] += int(rank is not None and rank <= 10)

    segment = "overlap" if target_overlap else "nonoverlap"

    stats[f"{segment}_n"] += 1
    stats[f"{segment}_hit10"] += int(rank is not None and rank <= 10)

    top10 = order[:10]

    np.add.at(stats["exposures"], top10, 1)
    np.add.at(stats["group_counts"], group_ids[top10], 1)


def summarize(name, stats, n_events, n_movies):
    """Summarize ranking accuracy and aggregate exposure fairness."""
    slots = int(stats["exposures"].sum())

    if slots != n_events * 10:
        raise RuntimeError(
            f"{name}: expected {n_events * 10} top-10 exposures, "
            f"observed {slots}."
        )

    if int(stats["group_counts"].sum()) != slots:
        raise RuntimeError(
            f"{name}: group exposure accounting mismatch."
        )

    def rate(hits, count):
        return hits / count if count else None

    return {
        "method": name,
        "events": stats["events"],
        "hit10": rate(stats["hit10"], n_events),
        "hit50": rate(stats["hit50"], n_events),
        "ndcg10": rate(stats["ndcg10_sum"], n_events),
        "zero_shot_events": stats["zero_n"],
        "zero_shot_hit10": rate(
            stats["zero_hit10"], stats["zero_n"]
        ),
        "one_shot_events": stats["one_n"],
        "one_shot_hit10": rate(
            stats["one_hit10"], stats["one_n"]
        ),
        "target_context_overlap_events": stats["overlap_n"],
        "overlap_hit10": rate(
            stats["overlap_hit10"], stats["overlap_n"]
        ),
        "nonoverlap_events": stats["nonoverlap_n"],
        "nonoverlap_hit10": rate(
            stats["nonoverlap_hit10"], stats["nonoverlap_n"]
        ),
        "item_gini": gini(stats["exposures"]),
        "catalog_coverage10": float(
            np.count_nonzero(stats["exposures"]) / n_movies
        ),
        "group_exposure_shares": (
            stats["group_counts"] / slots
        ).tolist(),
    }


def make_event_record(
    event_id,
    method,
    order,
    true_local,
    target,
    frequency,
    target_overlap,
    context_nonempty,
    movie_ids,
    group_ids,
):
    """Record ranking metrics and the actual top-10 recommendations."""
    positions = np.flatnonzero(order == true_local)
    rank = int(positions[0]) + 1 if len(positions) else None

    top10 = order[:10]

    if len(top10) != 10:
        raise RuntimeError(
            f"Event {event_id}, method {method}: "
            f"expected 10 recommendations, got {len(top10)}."
        )

    # Convert catalog positions to actual movie IDs.
    recommended_movies = movie_ids[top10].astype(int).tolist()

    # Preserve the order of the recommendations.
    recommended_groups = group_ids[top10].astype(int).tolist()

    if len(set(recommended_movies)) != 10:
        raise RuntimeError(
            f"Event {event_id}, method {method}: "
            "duplicate movie IDs in top-10 recommendations."
        )

    return {
        "event_id": event_id,
        "method": method,
        "target_movie_id": int(target),
        "target_frequency": int(frequency),
        "frequency_group": (
            "zero_shot"
            if frequency == 0
            else "one_shot"
            if frequency == 1
            else "other"
        ),
        "target_context_overlap": int(target_overlap),
        "context_nonempty": int(context_nonempty),
        "rank_top50": rank if rank is not None else "",
        "hit10": int(rank is not None and rank <= 10),
        "hit50": int(rank is not None and rank <= 50),
        "ndcg10": (
            1.0 / np.log2(rank + 1)
            if rank is not None and rank <= 10
            else 0.0
        ),
        "top10_movie_ids": json.dumps(recommended_movies),
        "top10_group_ids": json.dumps(recommended_groups),
    }


def validate_catalog(movie_ids, group_ids, n_movies):
    """Validate catalog alignment before evaluating recommendations."""
    if len(movie_ids) != n_movies:
        raise ValueError("Movie catalog length mismatch.")

    if len(group_ids) != n_movies:
        raise ValueError(
            "Group IDs must align one-to-one with catalog movie IDs."
        )

    if len(set(map(int, movie_ids))) != n_movies:
        raise ValueError("Duplicate movie IDs in the candidate catalog.")

    if not np.all(np.isfinite(group_ids)):
        raise ValueError("Non-finite group IDs in catalog.")

    if np.any(group_ids < 0) or np.any(group_ids >= 5):
        raise ValueError(
            "Expected group IDs in the range 0-4, matching "
            "the five-group exposure accounting."
        )

    if not np.all(group_ids == np.floor(group_ids)):
        raise ValueError("Group IDs must be integers.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--config",
        default="config/crs/kgsf/redial.yaml",
    )
    parser.add_argument(
        "--catalog",
        default="catalog_audit_corrected.pkl",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/dataset/redial/nltk",
    )
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hub-cutoff", type=int, default=500)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--base-alpha", type=float, default=0.25)

    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=[1.0, 3.0, 7.0],
        help="Frequency-adaptation strengths selected on validation.",
    )

    parser.add_argument(
        "--out",
        default="results_corrected/adaptive_m2_validation.json",
    )

    args = parser.parse_args()

    if args.base_alpha < 0 or args.rrf_k <= 0 or args.hub_cutoff <= 0:
        raise ValueError(
            "Alpha must be nonnegative; RRF and hub cutoffs must be positive."
        )

    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    if not args.strengths or any(
        not np.isfinite(x) or x < 0 for x in args.strengths
    ):
        raise ValueError(
            "Adaptation strengths must be finite and nonnegative."
        )

    # Preserve the supplied order while removing duplicate strengths.
    strengths = list(dict.fromkeys(args.strengths))

    system, _valid, test, side_data = initialize(
        args.config, args.gpu
    )

    catalog = load_catalog(
        args.catalog,
        side_data["item_entity_ids"],
        system.device,
    )

    movie_ids = np.asarray(catalog["movie_ids"], dtype=np.int64)
    group_ids = np.asarray(catalog["group_ids"], dtype=np.int64)

    n_movies = len(movie_ids)

    if n_movies < 50:
        raise ValueError(
            f"Expected at least 50 candidate movies, found {n_movies}."
        )

    validate_catalog(movie_ids, group_ids, n_movies)

    pos = {
        int(movie): i
        for i, movie in enumerate(movie_ids)
    }

    freq_dict = {
        int(k): int(v)
        for k, v in catalog["meta"]["freq_dict"].items()
    }

    missing_freq = [
        int(movie)
        for movie in movie_ids
        if int(movie) not in freq_dict
    ]

    if missing_freq:
        raise ValueError(
            f"{len(missing_freq)} candidate movies lack frequency metadata."
        )

    movie_freqs = np.asarray(
        [freq_dict[int(movie)] for movie in movie_ids],
        dtype=np.float64,
    )

    if np.any(movie_freqs < 0) or not np.all(np.isfinite(movie_freqs)):
        raise ValueError("Invalid movie-frequency metadata.")

    raw_adj = load_graphs(
        side_data, Path(args.raw_dir)
    )["R1_raw"]

    index = build_index(
        raw_adj, movie_ids.tolist(), args.hub_cutoff
    )

    pad = int(test.pad_entity_idx)

    # Candidate-specific frequency multiplier.
    # All variants use the same model scores, KG retrieval and events.
    freq_factors = {
        strength: (
            1.0 + strength / np.sqrt(1.0 + movie_freqs)
        )
        for strength in strengths
    }

    variant_names = ["KGSF_only", "A0_fixed_025"]

    for strength in strengths:
        tag = f"{strength:g}"
        variant_names.extend([
            f"A1_frequency_{tag}",
            f"A2_frequency_context_{tag}",
        ])

    stats = {
        name: make_stats(n_movies)
        for name in variant_names
    }

    n_events = 0
    event_rows = []

    system.model.eval()

    with torch.no_grad():
        batches = test.get_rec_data(
            batch_size=args.batch_size,
            shuffle=False,
        )

        for batch in tqdm(batches, desc="Adaptive M2 held-out test"):
            model_batch = tuple(
                tensor.to(system.device)
                for tensor in batch
            )

            output = system.model.recommend(
                model_batch, mode="test"
            )

            if not isinstance(output, tuple) or len(output) != 3:
                raise RuntimeError(
                    "Unexpected KGSF recommendation output."
                )

            scores = output[2]

            scores = scores.index_select(
                1, catalog["candidate_ids"]
            ).detach().cpu().numpy()

            if scores.ndim != 2 or scores.shape[1] != n_movies:
                raise RuntimeError(
                    "Movie score shape does not match candidate catalog."
                )

            if not np.all(np.isfinite(scores)):
                raise RuntimeError(
                    "Non-finite KGSF scores encountered."
                )

            for context_row, target, row_scores in zip(
                batch[0].tolist(),
                batch[-1].tolist(),
                scores,
            ):
                target = int(target)

                if target not in pos:
                    raise RuntimeError(
                        f"Target {target} missing from candidate catalog."
                    )

                context = [
                    int(entity)
                    for entity in context_row
                    if int(entity) != pad
                ]

                unique_context = set(context)

                # Fraction of unique context entities with a raw KG neighbor.
                context_confidence = (
                    sum(
                        bool(raw_adj.get(entity, ()))
                        for entity in unique_context
                    ) / len(unique_context)
                    if unique_context
                    else 0.0
                )

                kg_ranked = retrieve(context, index)

                # Rank movies by KGSF scores.
                kgsf_order = np.lexsort(
                    (movie_ids, -row_scores)
                )

                kgsf_rank = np.empty(
                    n_movies, dtype=np.int32
                )

                kgsf_rank[kgsf_order] = np.arange(
                    1, n_movies + 1, dtype=np.int32
                )

                kgsf_vote = 1.0 / (
                    args.rrf_k + kgsf_rank.astype(np.float64)
                )

                # Reciprocal-rank fusion contribution from KG retrieval.
                kg_vote = np.zeros(
                    n_movies, dtype=np.float64
                )

                for rank, movie in enumerate(
                    kg_ranked, start=1
                ):
                    local = pos.get(int(movie))

                    if local is not None:
                        kg_vote[local] = (
                            1.0 / (args.rrf_k + rank)
                        )

                true_local = pos[target]
                frequency = freq_dict[target]
                target_overlap = target in unique_context

                event_id = n_events + 1

                # KGSF-only: original scores, without KG retrieval fusion.
                order = kgsf_order[:50]

                method = "KGSF_only"

                update_stats(
                    stats[method],
                    order,
                    true_local,
                    frequency,
                    target_overlap,
                    group_ids,
                )

                event_rows.append(
                    make_event_record(
                        event_id,
                        method,
                        order,
                        true_local,
                        target,
                        frequency,
                        target_overlap,
                        bool(unique_context),
                        movie_ids,
                        group_ids,
                    )
                )

                # A1/A2: adaptive variants.
                # A0: fixed-weight baseline.
                fused = (
                    kgsf_vote
                    + args.base_alpha * kg_vote
                )

                order = np.lexsort(
                    (movie_ids, -fused)
                )[:50]

                method = "A0_fixed_025"

                update_stats(
                    stats[method],
                    order,
                    true_local,
                    frequency,
                    target_overlap,
                    group_ids,
                )

                event_rows.append(
                    make_event_record(
                        event_id,
                        method,
                        order,
                        true_local,
                        target,
                        frequency,
                        target_overlap,
                        bool(unique_context),
                        movie_ids,
                        group_ids,
                    )
                )

                # A1/A2: adaptive variants.
                for strength in strengths:
                    factor = freq_factors[strength]
                    tag = f"{strength:g}"

                    # A1: frequency-adaptive KG weight.
                    alpha_a1 = args.base_alpha * factor

                    fused_a1 = (
                        kgsf_vote + alpha_a1 * kg_vote
                    )

                    order_a1 = np.lexsort(
                        (movie_ids, -fused_a1)
                    )[:50]

                    method_a1 = f"A1_frequency_{tag}"

                    update_stats(
                        stats[method_a1],
                        order_a1,
                        true_local,
                        frequency,
                        target_overlap,
                        group_ids,
                    )

                    event_rows.append(
                        make_event_record(
                            event_id,
                            method_a1,
                            order_a1,
                            true_local,
                            target,
                            frequency,
                            target_overlap,
                            bool(unique_context),
                            movie_ids,
                            group_ids,
                        )
                    )

                    # A2: frequency and context-confidence adaptive.
                    alpha_a2 = (
                        args.base_alpha
                        * factor
                        * context_confidence
                    )

                    fused_a2 = (
                        kgsf_vote + alpha_a2 * kg_vote
                    )

                    order_a2 = np.lexsort(
                        (movie_ids, -fused_a2)
                    )[:50]

                    method_a2 = (
                        f"A2_frequency_context_{tag}"
                    )

                    update_stats(
                        stats[method_a2],
                        order_a2,
                        true_local,
                        frequency,
                        target_overlap,
                        group_ids,
                    )

                    event_rows.append(
                        make_event_record(
                            event_id,
                            method_a2,
                            order_a2,
                            true_local,
                            target,
                            frequency,
                            target_overlap,
                            bool(unique_context),
                            movie_ids,
                            group_ids,
                        )
                    )

                n_events += 1

    # Preserve the known validation-set size and method count.
    if n_events <= 0:
        raise RuntimeError(
            f"No held-out test events found: {n_events}."
        )

    results = [
        summarize(
            name,
            stats[name],
            n_events,
            n_movies,
        )
        for name in variant_names
    ]

    expected_rows = n_events * len(variant_names)

    if len(event_rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} event-method records, "
            f"observed {len(event_rows)}."
        )

    # Validate event/method pairing before writing.
    expected_methods = set(variant_names)
    rows_by_event = {}

    for row in event_rows:
        eid = row["event_id"]
        rows_by_event.setdefault(eid, []).append(row)

    if len(rows_by_event) != n_events:
        raise RuntimeError(
            "Unique event count does not match validation event count."
        )

    for eid, rows in rows_by_event.items():
        methods = [row["method"] for row in rows]

        if len(methods) != len(set(methods)):
            raise RuntimeError(
                f"Duplicate method rows for event {eid}."
            )

        if set(methods) != expected_methods:
            raise RuntimeError(
                f"Incomplete method set for event {eid}."
            )

    # Write event-level records using the original output naming convention.
    target_path = Path(args.out)
    if target_path.resolve().parent != Path(
        "results_final_test_m2"
    ).resolve():
        raise RuntimeError(
            "Final test outputs must be inside results_final_test_m2."
        )
    if target_path.exists() or target_path.with_name(
        target_path.stem + "_events.csv"
    ).exists():
        raise FileExistsError(
            "Test output already exists; refusing to overwrite."
        )

    event_path = target_path.with_name(
        target_path.stem + "_events.csv"
    )

    target_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    with event_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as event_file:
        writer = csv.DictWriter(
            event_file,
            fieldnames=list(event_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(event_rows)

    report = {
        "split": "held_out_test",
        "method": (
            "Fixed versus candidate-frequency "
            "and context-adaptive RRF"
        ),
        "events": n_events,
        "candidate_movies": n_movies,
        "methods": variant_names,
        "base_alpha": args.base_alpha,
        "rrf_k": args.rrf_k,
        "hub_cutoff": args.hub_cutoff,
        "frequency_strengths": strengths,
        "context_confidence": (
            "Fraction of unique context entities with at least "
            "one raw KG neighbor."
        ),
        "top10_exposure_unit": "one recommended movie in one event",
        "top10_movie_ids_saved": True,
        "top10_group_ids_saved": True,
        "target_used_for_ranking": False,
        "checkpoint_changed": False,
        "test_split_evaluated": True,
        "event_level_csv": str(event_path),
        "event_level_rows": len(event_rows),
        "results": results,
        "caution": (
            "Exploratory validation comparison, not a novelty claim. "
            "Configuration selected and locked on validation before "
            "this held-out test evaluation. Fairness uncertainty should "
            "be estimated by resampling conversations and recomputing "
            "the fairness metrics from event-level recommendation lists."
        ),
    }

    target_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print("Saved:", target_path.resolve())
    print("Saved event-level records:", event_path.resolve())
    print(
        f"Validation events: {n_events}; "
        f"methods: {len(variant_names)}; "
        f"event rows: {len(event_rows)}"
    )


if __name__ == "__main__":
    main()
