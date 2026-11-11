"""Verify the official HybridKla archive and extract its two feature tables."""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path


ARCHIVE_SHA256 = "73ddb128c441d7b213c4636e6bb6ce34501e82143c2b9bc06ec329d68a422993"
RESOURCE_NAMES = ("top10.txt", "BLOSUM62R.txt")


def extract(archive_path: Path, output_dir: Path) -> None:
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    checksum = hashlib.sha256()
    with archive_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    digest = checksum.hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError("Official HybridKla archive checksum mismatch")

    with zipfile.ZipFile(archive_path) as archive:
        payloads = {
            name: archive.read(f"HybridKla/train/{name}")
            for name in RESOURCE_NAMES
        }
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        (output_dir / name).write_bytes(payload)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: extract_hybridkla_resources.py ARCHIVE OUTPUT_DIR")
    extract(Path(sys.argv[1]), Path(sys.argv[2]))
