import pickle

path = "catalog_audit_corrected.pkl"

with open(path, "rb") as f:
    meta = pickle.load(f)

print("Catalog metadata type:", type(meta))

if isinstance(meta, dict):
    print("Metadata keys:", list(meta.keys()))

    for key in ["movie_ids", "freq_dict", "group_ids"]:
        if key in meta:
            value = meta[key]
            print("\n", key)
            print("Type:", type(value))
            print("Length:", len(value))
            print("Sample:", list(value.items())[:5]
                  if isinstance(value, dict)
                  else value[:5])
