#!/usr/bin/env python
"""
Phase 1 diagnostic for KG-FairCRS category metadata.

This script does NOT silently assign genres. It:
1) verifies the official movie catalog against entity2id,
2) enumerates direct DBpedia KG edges incident to catalog movies,
3) records relation-ID frequencies and neighboring entity URIs,
4) flags URI-name-based genre candidates for manual/predicate validation.

Run from the repository root:
    python audit_category_metadata.py --data-dir data/dataset/redial/nltk \
        --out-dir results_category_audit

Important:
The ReDial DBpedia subgraph stores integer relation IDs. Unless a trustworthy
relation-ID-to-property mapping is available, a genre-looking neighbor is only
a candidate, not a validated genre assignment.
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote


GENRE_TERMS = (
    "genre", "film_genre", "movie_genre", "filmgenre",
    "comedy", "drama", "thriller", "horror", "romance",
    "science_fiction", "science-fiction", "documentary",
    "animation", "action_film", "adventure_film", "crime_fiction",
    "western", "musical", "war_film", "fantasy"
)


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def uri_local_name(uri):
    value = unquote(str(uri)).rstrip("/").rsplit("/", 1)[-1]
    value = value.rsplit("#", 1)[-1]
    return value.replace("_", " ")


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/dataset/redial/nltk")
    ap.add_argument("--out-dir", default="results_category_audit")
    args = ap.parse_args()

    root = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    required = ["movie_ids.json", "entity2id.json", "dbpedia_subkg.json"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required ReDial files: " + ", ".join(missing)
        )

    movie_ids_raw = load_json(root / "movie_ids.json")
    entity2id_raw = load_json(root / "entity2id.json")
    kg = load_json(root / "dbpedia_subkg.json")

    movie_ids = [int(x) for x in movie_ids_raw]
    if len(movie_ids) != len(set(movie_ids)):
        raise ValueError("movie_ids.json contains duplicate entity IDs")

    entity2id = {str(uri): int(eid) for uri, eid in entity2id_raw.items()}
    id2entity = {}
    for uri, eid in entity2id.items():
        if eid in id2entity and id2entity[eid] != uri:
            raise ValueError(f"Entity ID collision at {eid}")
        id2entity[eid] = uri

    movie_set = set(movie_ids)
    movie_to_uri = {mid: id2entity.get(mid) for mid in movie_ids}
    missing_movie_entities = [mid for mid, uri in movie_to_uri.items() if uri is None]

    # Keep directed source records and separately build undirected incident edges.
    direct_edges = []
    relation_counts = Counter()
    movie_degree = Counter()
    candidate_neighbor_edges = []
    all_neighbor_ids = set()

    for head_text, entries in kg.items():
        head = int(head_text)
        for entry in entries:
            if len(entry) != 2:
                raise ValueError(f"Unexpected KG edge format: {entry!r}")
            relation, tail = int(entry[0]), int(entry[1])
            relation_counts[relation] += 1

            # CRSLab's ReDial graph processor reserves relation 185 for
            # self-loops and excludes it from ordinary KG relationships.
            if head == tail or relation == 185:
                continue

            if head in movie_set or tail in movie_set:
                movie_id = head if head in movie_set else tail
                neighbor_id = tail if head in movie_set else head
                direct_edges.append({
                    "movie_entity_id": movie_id,
                    "movie_uri": movie_to_uri.get(movie_id, ""),
                    "relation_id": relation,
                    "neighbor_entity_id": neighbor_id,
                    "neighbor_uri": id2entity.get(neighbor_id, ""),
                    "neighbor_local_name": uri_local_name(
                        id2entity.get(neighbor_id, f"entity_{neighbor_id}")
                    ),
                    "edge_direction_in_source": (
                        "movie_to_neighbor" if head in movie_set
                        else "neighbor_to_movie"
                    )
                })
                movie_degree[movie_id] += 1
                all_neighbor_ids.add(neighbor_id)

    # URI text is only a discovery heuristic. No row is called a validated genre.
    for row in direct_edges:
        local = row["neighbor_local_name"].lower().replace(" ", "_")
        uri = row["neighbor_uri"].lower()
        if any(term in local or term in uri for term in GENRE_TERMS):
            candidate_neighbor_edges.append({
                **row,
                "candidate_reason": "genre-related URI/local-name keyword",
                "validated_genre": "NO"
            })

    movie_rows = []
    for mid in movie_ids:
        movie_rows.append({
            "movie_entity_id": mid,
            "movie_uri": movie_to_uri.get(mid, ""),
            "has_entity_mapping": int(movie_to_uri.get(mid) is not None),
            "direct_kg_degree_records": int(movie_degree.get(mid, 0))
        })

    relation_rows = [
        {"relation_id": rid, "raw_edge_record_count": count}
        for rid, count in relation_counts.most_common()
    ]

    write_csv(
        out / "movie_entity_mapping.csv", movie_rows,
        ["movie_entity_id", "movie_uri", "has_entity_mapping",
         "direct_kg_degree_records"]
    )
    write_csv(
        out / "movie_direct_kg_edges.csv", direct_edges,
        ["movie_entity_id", "movie_uri", "relation_id",
         "neighbor_entity_id", "neighbor_uri", "neighbor_local_name",
         "edge_direction_in_source"]
    )
    write_csv(
        out / "relation_id_frequency.csv", relation_rows,
        ["relation_id", "raw_edge_record_count"]
    )
    write_csv(
        out / "genre_candidate_edges_unverified.csv", candidate_neighbor_edges,
        ["movie_entity_id", "movie_uri", "relation_id",
         "neighbor_entity_id", "neighbor_uri", "neighbor_local_name",
         "edge_direction_in_source", "candidate_reason", "validated_genre"]
    )

    summary = {
        "catalog_movies": len(movie_ids),
        "catalog_ids_unique": len(movie_ids) == len(set(movie_ids)),
        "movie_ids_missing_from_entity2id": len(missing_movie_entities),
        "movie_ids_missing_examples": missing_movie_entities[:20],
        "raw_kg_head_entities": len(kg),
        "raw_kg_relation_ids": len(relation_counts),
        "raw_kg_edge_records": int(sum(relation_counts.values())),
        "catalog_movies_with_at_least_one_direct_kg_edge": len(movie_degree),
        "catalog_movies_without_direct_kg_edge": len(movie_set - set(movie_degree)),
        "direct_edge_records_incident_to_catalog_movies": len(direct_edges),
        "distinct_neighbor_entities_of_catalog_movies": len(all_neighbor_ids),
        "genre_keyword_candidate_edge_records_unverified": len(candidate_neighbor_edges),
        "genre_keyword_candidate_neighbor_entities_unverified": len({
            r["neighbor_entity_id"] for r in candidate_neighbor_edges
        }),
        "critical_limitation": (
            "The raw subgraph contains integer relation IDs, but this archive "
            "does not include a verified relation-ID-to-DBpedia-property mapping. "
            "URI keyword matches are discovery candidates only, not validated genres."
        )
    }
    (out / "category_metadata_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print("\nSaved files:")
    for p in sorted(out.glob("*")):
        print(" -", p)


if __name__ == "__main__":
    main()
