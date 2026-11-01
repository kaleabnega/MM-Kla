from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from pcbert_kla_clean.source_study import lock_external_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the screened rice external-test inventory")
    parser.add_argument("--gao-manifest", type=Path, default=PROJECT_DIR / "review_data/gao_2020_source_split/source_split_manifest.csv")
    parser.add_argument("--rice-candidates", type=Path, default=PROJECT_DIR / "review_data/source_study_audit_2026_11_01/rice_homology_screened_candidates.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "review_data/locked_source_study_v1")
    args = parser.parse_args()
    print(json.dumps(lock_external_manifest(args.gao_manifest, args.rice_candidates, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
