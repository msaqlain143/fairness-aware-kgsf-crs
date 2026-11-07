from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import hashlib
from datetime import datetime, timezone

BASE = Path("data/dataset/redial/nltk")
OUT = Path("results_category_audit/experiment_21_18A5")

KG_FILE = BASE / "dbpedia_subkg.json"
ENTITY_FILE = BASE / "entity2id.json"
MOVIE_FILE = BASE / "movie_ids.json"

OUT.mkdir(parents=True, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def uri_label(uri):
    if not uri:
        return ""
    value = uri.strip("<>")
    return value.rsplit("/", 1)[-1].replace("_", " ")


def main():

    print("=" * 75)
    print("EXPERIMENT 21.18A.5: COMPLETE MOVIE-CENTRIC RELATIONSHIP DISCOVERY")
    print("=" * 75)

    print("\n[1] LOADING ORIGINAL SOURCES")

    kg = load_json(KG_FILE)
    entity2id = load_json(ENTITY_FILE)
    movie_ids_raw = load_json(MOVIE_FILE)

    id2uri = {int(v): k for k, v in entity2id.items()}
    movie_ids = set(map(int, movie_ids_raw))

    assert len(id2uri) == len(entity2id)
    assert len(movie_ids) == 6924
    assert all(mid in id2uri for mid in movie_ids)

    print("KG heads:", len(kg))
    print("Entity mappings:", len(entity2id))
    print("Original movie population:", len(movie_ids))

    print("\n[2] RECONSTRUCTING ORIGINAL TRIPLES")

    all_triples = []
    malformed = []
    global_relation_counts = Counter()
    exact_counts = Counter()

    for head_raw, edges in kg.items():

        try:
            head = int(head_raw)
        except (ValueError, TypeError):
            malformed.append({
                "head": str(head_raw),
                "relation": "",
                "tail": "",
                "error": "INVALID_HEAD_ID"
            })
            continue

        if not isinstance(edges, list):
            malformed.append({
                "head": head,
                "relation": "",
                "tail": "",
                "error": "INVALID_EDGE_COLLECTION"
            })
            continue

        for edge_index, edge in enumerate(edges):

            if not isinstance(edge, list) or len(edge) != 2:
                malformed.append({
                    "head": head,
                    "relation": "",
                    "tail": str(edge),
                    "error": "MALFORMED_TRIPLE"
                })
                continue

            try:
                relation, tail = map(int, edge)
            except (ValueError, TypeError):
                malformed.append({
                    "head": head,
                    "relation": str(edge[0]),
                    "tail": str(edge[1]),
                    "error": "INVALID_RELATION_OR_TAIL_ID"
                })
                continue

            triple = (head, relation, tail)

            all_triples.append({
                "head": head,
                "relation": relation,
                "tail": tail,
                "source_edge_index": edge_index
            })

            global_relation_counts[relation] += 1
            exact_counts[triple] += 1

    print("Valid original triples:", len(all_triples))
    print("Unique relations:", len(global_relation_counts))
    print("Malformed records:", len(malformed))

    print("\n[3] MOVIE-CENTRIC EDGE EXTRACTION")

    movie_edges = []
    participation = {
        mid: Counter() for mid in movie_ids
    }

    for record in all_triples:

        h = record["head"]
        r = record["relation"]
        t = record["tail"]

        head_is_movie = h in movie_ids
        tail_is_movie = t in movie_ids

        if not head_is_movie and not tail_is_movie:
            continue

        if head_is_movie and tail_is_movie:
            direction = "SELF_LOOP" if h == t else "MOVIE_TO_MOVIE"
            participation[h]["movie_to_movie"] += 1
            if h != t:
                participation[t]["movie_to_movie"] += 1

        elif head_is_movie:
            direction = "MOVIE_AS_HEAD"
            participation[h]["outgoing"] += 1

        else:
            direction = "MOVIE_AS_TAIL"
            participation[t]["incoming"] += 1

        if h == t:
            participation[h]["self_loop"] += 1

        movie_edges.append({
            "head_id": h,
            "head_uri": id2uri[h],
            "head_label": uri_label(id2uri[h]),
            "relation_id": r,
            "tail_id": t,
            "tail_uri": id2uri.get(t, ""),
            "tail_label": uri_label(id2uri.get(t, "")),
            "movie_direction": direction,
            "source_edge_index": record["source_edge_index"]
        })

    print("Movie-centric triple records:", len(movie_edges))

    print("\n[4] RELATION SUMMARY")

    relation_summary = []

    for rid in sorted(global_relation_counts):

        subset = [
            x for x in movie_edges
            if x["relation_id"] == rid
        ]

        relation_summary.append({
            "relation_id": rid,
            "global_triple_count": global_relation_counts[rid],
            "movie_centric_edge_count": len(subset),
            "movie_as_head_count": sum(
                x["movie_direction"] == "MOVIE_AS_HEAD" for x in subset
            ),
            "movie_as_tail_count": sum(
                x["movie_direction"] == "MOVIE_AS_TAIL" for x in subset
            ),
            "movie_to_movie_count": sum(
                x["movie_direction"] == "MOVIE_TO_MOVIE" for x in subset
            ),
            "self_loop_count": sum(
                x["movie_direction"] == "SELF_LOOP" for x in subset
            )
        })

    print("Original relation IDs:", len(relation_summary))

    print("\n[5] NEIGHBOR ENTITY INVENTORY")

    neighbors = {}

    for row in movie_edges:
        for entity_id in (row["head_id"], row["tail_id"]):
            uri = id2uri.get(entity_id, "")
            neighbors[entity_id] = {
                "entity_id": entity_id,
                "entity_uri": uri,
                "entity_label": uri_label(uri),
                "is_movie": entity_id in movie_ids
            }

    print("Unique neighboring entities:", len(neighbors))

    print("\n[6] MOVIE PARTICIPATION")

    participation_rows = []

    for mid in sorted(movie_ids):

        c = participation[mid]

        participation_rows.append({
            "movie_entity_id": mid,
            "movie_uri": id2uri[mid],
            "movie_label": uri_label(id2uri[mid]),
            "outgoing_edge_count": c["outgoing"],
            "incoming_edge_count": c["incoming"],
            "movie_to_movie_count": c["movie_to_movie"],
            "self_loop_count": c["self_loop"],
            "total_incident_edges": sum(c.values())
        })

    print("Movies with no incident edges:", sum(
        x["total_incident_edges"] == 0 for x in participation_rows
    ))

    print("\n[7] DUPLICATE EDGE AUDIT")

    duplicate_rows = []

    for triple, count in exact_counts.items():
        if count > 1:
            h, r, t = triple
            duplicate_rows.append({
                "head_id": h,
                "relation_id": r,
                "tail_id": t,
                "duplicate_count": count,
                "head_uri": id2uri.get(h, ""),
                "tail_uri": id2uri.get(t, "")
            })

    movie_duplicate_rows = [
        x for x in duplicate_rows
        if x["head_id"] in movie_ids or x["tail_id"] in movie_ids
    ]

    print("Global duplicate groups:", len(duplicate_rows))
    print("Movie-centric duplicate groups:", len(movie_duplicate_rows))

    print("\n[8] INVALID EDGE AUDIT")

    for row in all_triples:
        h, t = row["head"], row["tail"]

        if h not in id2uri:
            malformed.append({
                "head": h,
                "relation": row["relation"],
                "tail": t,
                "error": "MISSING_HEAD_MAPPING"
            })

        if t not in id2uri:
            malformed.append({
                "head": h,
                "relation": row["relation"],
                "tail": t,
                "error": "MISSING_TAIL_MAPPING"
            })

    print("Invalid records:", len(malformed))

    print("\n[9] EXPORTING RESULTS")

    write_csv(
        OUT / "movie_centric_relationship_inventory.csv",
        movie_edges,
        [
            "head_id", "head_uri", "head_label",
            "relation_id", "tail_id", "tail_uri",
            "tail_label", "movie_direction", "source_edge_index"
        ]
    )

    write_csv(
        OUT / "movie_relation_summary.csv",
        relation_summary,
        [
            "relation_id", "global_triple_count",
            "movie_centric_edge_count",
            "movie_as_head_count", "movie_as_tail_count",
            "movie_to_movie_count", "self_loop_count"
        ]
    )

    write_csv(
        OUT / "movie_neighbor_entity_inventory.csv",
        list(neighbors.values()),
        ["entity_id", "entity_uri", "entity_label", "is_movie"]
    )

    write_csv(
        OUT / "movie_participation_summary.csv",
        participation_rows,
        [
            "movie_entity_id", "movie_uri", "movie_label",
            "outgoing_edge_count", "incoming_edge_count",
            "movie_to_movie_count", "self_loop_count",
            "total_incident_edges"
        ]
    )

    write_csv(
        OUT / "duplicate_edge_register.csv",
        movie_duplicate_rows,
        [
            "head_id", "relation_id", "tail_id",
            "duplicate_count", "head_uri", "tail_uri"
        ]
    )

    write_csv(
        OUT / "invalid_edge_register.csv",
        malformed,
        ["head", "relation", "tail", "error"]
    )

    print("\n[10] FINAL VALIDATION")

    validation = {
        "original_movie_population_6924": len(movie_ids) == 6924,
        "all_movies_mapped": all(mid in id2uri for mid in movie_ids),
        "all_original_triples_preserved": len(all_triples) == 165696,
        "original_relation_population_214": len(global_relation_counts) == 214,
        "movie_edge_population_consistent":
            len(movie_edges) == sum(
                x["movie_direction"] in (
                    "MOVIE_AS_HEAD", "MOVIE_AS_TAIL",
                    "MOVIE_TO_MOVIE", "SELF_LOOP"
                ) for x in movie_edges
            ),
        "participation_population_6924": len(participation_rows) == 6924,
        "neighbor_inventory_unique": len(neighbors) == len(set(neighbors)),
        "no_missing_entity_mappings": len(malformed) == 0
    }

    for name, passed in validation.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    manifest = {
        "experiment": "21.18A.5",
        "name": "Complete Movie-Centric Relationship Structure Discovery",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": {
            "dbpedia_subkg": sha256(KG_FILE),
            "entity2id": sha256(ENTITY_FILE),
            "movie_ids": sha256(MOVIE_FILE)
        },
        "counts": {
            "movie_population": len(movie_ids),
            "previous_audit_population": 6112,
            "additional_movie_population": len(movie_ids) - 6112,
            "kg_heads": len(kg),
            "global_triples": len(all_triples),
            "unique_relations": len(global_relation_counts),
            "movie_centric_triples": len(movie_edges),
            "unique_neighbor_entities": len(neighbors),
            "global_duplicate_groups": len(duplicate_rows),
            "movie_duplicate_groups": len(movie_duplicate_rows),
            "invalid_records": len(malformed)
        },
        "direction_definition": {
            "MOVIE_AS_HEAD": "Movie is original triple head",
            "MOVIE_AS_TAIL": "Movie is original triple tail",
            "MOVIE_TO_MOVIE": "Distinct movie entities at both endpoints",
            "SELF_LOOP": "Original head and tail are identical"
        },
        "validation": validation,
        "completion_status": (
            "COMPLETED_ALL_VALIDATIONS_PASSED"
            if all(validation.values())
            else "COMPLETED_WITH_VALIDATION_FAILURES"
        ),
        "methodological_constraints": [
            "All 6924 original movie IDs retained",
            "Original triple direction preserved",
            "No relation assumed to represent genre",
            "No automatic semantic classification",
            "No graph modification",
            "No automatic entity merging"
        ]
    }

    with open(
        OUT / "experiment_21_18A5_manifest.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 75)
    print("EXPERIMENT 21.18A.5 FINISHED")
    print("Output:", OUT.resolve())
    print("Completion:", manifest["completion_status"])
    print("=" * 75)


if __name__ == "__main__":
    main()
