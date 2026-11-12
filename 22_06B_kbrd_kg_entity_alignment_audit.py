
import json
from pathlib import Path
from collections import Counter

ROOT = Path("data/dataset/redial/nltk")

FILES = {
    "entity2id": ROOT / "entity2id.json",
    "movie_ids": ROOT / "movie_ids.json",
    "dbpedia": ROOT / "dbpedia_subkg.json",
    "conceptnet": ROOT / "conceptnet_subkg.txt",
}


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def inspect_object(name, obj):
    print(f"\n========== {name.upper()} ==========")
    print("Type:", type(obj).__name__)

    if isinstance(obj, dict):
        print("Number of entries:", len(obj))
        print("First 5 entries:")
        for k, v in list(obj.items())[:5]:
            print(repr(k)[:150], "=>", repr(v)[:200])

    elif isinstance(obj, list):
        print("Number of entries:", len(obj))
        print("First 5 entries:")
        for x in obj[:5]:
            print(repr(x)[:300])

    else:
        print("Sample:", repr(obj)[:500])


def normalize_mapping(obj):
    """Extract possible entity-ID pairs without assuming mapping direction."""
    pairs = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (int, float)):
                pairs.append((str(k), int(v)))
            elif isinstance(v, str) and v.isdigit():
                pairs.append((str(k), int(v)))

    return pairs


def main():

    print("=" * 75)
    print("EXPERIMENT 22.6B — KG ENTITY ALIGNMENT AUDIT")
    print("=" * 75)

    for name, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(path)

    entity2id = load_json(FILES["entity2id"])
    movie_ids = load_json(FILES["movie_ids"])
    dbpedia = load_json(FILES["dbpedia"])

    inspect_object("ENTITY2ID", entity2id)
    inspect_object("MOVIE_IDS", movie_ids)
    inspect_object("DBPEDIA SUBKG", dbpedia)

    print("\n========== ENTITY MAPPING AUDIT ==========")

    entity_pairs = normalize_mapping(entity2id)
    movie_pairs = normalize_mapping(movie_ids)

    print("Entity mapping pairs:", len(entity_pairs))
    print("Movie mapping pairs:", len(movie_pairs))

    entity_names = {x[0] for x in entity_pairs}
    entity_values = {x[1] for x in entity_pairs}

    movie_names = {x[0] for x in movie_pairs}
    movie_values = {x[1] for x in movie_pairs}

    print("Movie keys matching entity keys:",
          len(movie_names & entity_names))

    print("Movie values matching entity values:",
          len(movie_values & entity_values))

    print("\n========== DBPEDIA GRAPH STRUCTURE ==========")

    if isinstance(dbpedia, dict):
        print("Top-level keys:", list(dbpedia.keys())[:20])

    elif isinstance(dbpedia, list):
        print("Number of graph records:", len(dbpedia))

    print("\n========== CONCEPTNET STRUCTURE ==========")

    with FILES["conceptnet"].open(
        "r", encoding="utf-8"
    ) as f:
        concept_lines = [next(f, None) for _ in range(5)]

    for i, line in enumerate(concept_lines):
        print(f"Line {i + 1}:", repr(line))

    print("\n========== MOVIE ENTITY OVERLAP ==========")

    if movie_pairs and entity_pairs:
        print("First movie mapping pairs:")
        for pair in movie_pairs[:10]:
            print(pair)

    print("\n========== AUDIT SUMMARY ==========")

    print("Entity2ID entries:", len(entity2id))
    print("Movie ID entries:", len(movie_ids))
    print("DBpedia records:",
          len(dbpedia) if hasattr(dbpedia, "__len__") else "Unknown")

    print("\nAudit completed.")
    print("No recommendation ranking or model was modified.")


if __name__ == "__main__":
    main()
