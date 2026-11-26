
import subprocess
import hashlib

targets = {
    "22_16": "6555144efe624658eb32e8de48ee9e95ef94baded9486a8f6280d848ecbef751",
    "22_35F": "fc3658bc607d13ec6c3be631fb707dd34a449c4b5fbd280e03bfcb0fb015457f",
    "22_35J": "42ba8ee6de15e100614d4d61b1635fee80701df59631fadae2efe3f9a48dcb3c",
    "22_35C": "cb0c3b8115b561a26d2e76775bf9904468cc477d89bc266b09d253c459eca69a",
    "22_18": "651050b50c34cc9ebbaa1cf63217f5c82a6b9276c915e4440435ed436f4a28e2",
}

reverse = {v: k for k, v in targets.items()}
found = {}

print("Searching all Git objects, including unreachable objects...")

process = subprocess.Popen(
    ["git", "cat-file", "--batch-all-objects", "--batch"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

count = 0

while True:
    header = process.stdout.readline()

    if not header:
        break

    parts = header.strip().split()

    if len(parts) != 3:
        raise RuntimeError(f"Unexpected Git object header: {header!r}")

    oid, obj_type, size_text = parts
    size = int(size_text)

    digest = hashlib.sha256()
    remaining = size

    while remaining:
        chunk = process.stdout.read(min(1024 * 1024, remaining))

        if not chunk:
            raise RuntimeError("Unexpected end of Git object.")

        if obj_type == b"blob":
            digest.update(chunk)

        remaining -= len(chunk)

    process.stdout.read(1)

    count += 1

    if obj_type == b"blob":
        sha = digest.hexdigest()

        if sha in reverse:
            name = reverse[sha]
            found[name] = oid.decode()

            print("EXACT MATCH:", name, oid.decode())

    if count % 10000 == 0:
        print("Objects scanned:", count)

return_code = process.wait()

if return_code != 0:
    raise RuntimeError("Git object scan failed.")

print("\n========== FINAL RECOVERY RESULTS ==========")

for name in targets:
    print(
        name,
        "FOUND" if name in found else "NOT FOUND",
        found.get(name, "")
    )

print("\nTotal objects scanned:", count)
print("Exact historical versions recovered:", len(found))
