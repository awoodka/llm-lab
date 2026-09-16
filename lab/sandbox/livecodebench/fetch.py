"""Build step: fetch LiveCodeBench's code and one release file, and refuse anything that isn't pinned.

The code comes from GitHub's archive of the pinned commit. The archive's own bytes can change when GitHub
recompresses, so the check is on every extracted file (lcb_files.sha256), not on the archive.
"""

import hashlib
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
ARCHIVE = f"https://codeload.github.com/LiveCodeBench/LiveCodeBench/tar.gz/{COMMIT}"
DATASET_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
RELEASE_FILE = "test6.jsonl"
RELEASE_SHA256 = "bb4c364f71921c4495a6ad15abe1a927350b720009f4933e2e71f8af0f6fd1f5"
RELEASE_URL = f"https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/{DATASET_REVISION}/{RELEASE_FILE}"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main(dest: Path, manifest: Path) -> None:
    expected = {}
    for line in manifest.read_text().splitlines():
        if line.strip():
            digest, path = line.split(None, 1)
            expected[path.strip()] = digest
    prefix = f"LiveCodeBench-{COMMIT}/"
    with urllib.request.urlopen(ARCHIVE, timeout=120) as r:
        archive = tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz")
    found = {}
    for member in archive.getmembers():
        name = member.name.removeprefix(prefix)
        if not member.isfile() or name not in expected:
            continue
        data = archive.extractfile(member).read()
        if sha256(data) != expected[name]:
            sys.exit(f"{name}: checksum mismatch")
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        found[name] = True
    missing = sorted(set(expected) - set(found))
    if missing:
        sys.exit(f"missing from the archive: {missing}")

    data_dir = dest / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with urllib.request.urlopen(RELEASE_URL, timeout=600) as r, open(data_dir / RELEASE_FILE, "wb") as out:
        while chunk := r.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != RELEASE_SHA256:
        sys.exit(f"{RELEASE_FILE}: checksum mismatch")
    print(f"LiveCodeBench {COMMIT[:12]}: {len(found)} files; {RELEASE_FILE} @ {DATASET_REVISION[:12]}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
