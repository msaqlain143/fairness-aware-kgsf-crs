#!/usr/bin/env python
"""Inspect CRSLab's restored ReDial all_data.pkl without modifying it."""
import pickle
from pathlib import Path

P = Path("data/dataset/redial/nltk/all_data.pkl")

def describe(name, obj, depth=0):
    pad = " " * depth
    print(f"\n{pad}--- {name} ---")
    print(f"{pad}type: {type(obj)}")
    try:
        print(f"{pad}len: {len(obj)}")
    except Exception:
        pass

    if isinstance(obj, dict):
        print(f"{pad}keys: {list(obj.keys())[:80]}")
        for key, value in list(obj.items())[:8]:
            if isinstance(value, (list, tuple)) and value:
                print(f"{pad}{key}: {type(value).__name__}, len={len(value)}")
                first = value[0]
                print(f"{pad}  first item type: {type(first)}")
                if isinstance(first, dict):
                    print(f"{pad}  first item keys: {list(first.keys())}")
                    print(f"{pad}  first item preview: {str(first)[:1200]}")
            elif isinstance(value, dict):
                print(f"{pad}{key}: dict len={len(value)}, keys={list(value.keys())[:20]}")
            else:
                print(f"{pad}{key}: {str(value)[:300]}")
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj[:12]):
            print(f"{pad}[{i}] type={type(value)}")
            try:
                print(f"{pad}    len={len(value)}")
            except Exception:
                pass
            if isinstance(value, dict):
                print(f"{pad}    keys={list(value.keys())[:80]}")
                for k, v in list(value.items())[:5]:
                    print(f"{pad}    {k}: {str(v)[:250]}")
            elif isinstance(value, (list, tuple)) and value:
                first = value[0]
                print(f"{pad}    first type={type(first)}")
                if isinstance(first, dict):
                    print(f"{pad}    first keys={list(first.keys())[:80]}")
                    print(f"{pad}    first preview={str(first)[:1000]}")

def main():
    if not P.is_file():
        raise FileNotFoundError(P)
    print("FILE:", P.resolve())
    print("SIZE:", P.stat().st_size)
    with P.open("rb") as f:
        data = pickle.load(f)

    describe("ROOT", data)
    if isinstance(data, tuple):
        print("\n========== TUPLE ELEMENT SUMMARY ==========")
        for i, element in enumerate(data):
            describe(f"TUPLE ELEMENT {i}", element, depth=2)

    print("\n========== RAW DATASET DIRECTORY ==========")
    raw_dir = Path("data/dataset/redial")
    for p in sorted(raw_dir.rglob("*")):
        if p.is_file() and p != P:
            print(f"{p} | {p.stat().st_size} bytes")

if __name__ == "__main__":
    main()


