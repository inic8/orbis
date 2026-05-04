#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pruning.metadata import extract_pruned_metadata  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract minimal reconstruction metadata from an Orbis checkpoint."
    )
    parser.add_argument("checkpoint", help="Path to the checkpoint file (.ckpt or .pt)")
    parser.add_argument("--config", default=None, help="Optional path to config.yaml")
    parser.add_argument("--orbis-repo", default=None, help="Optional path to the Orbis checkout or its parent workspace")
    parser.add_argument("--output", default=None, help="Optional output JSON path; defaults to stdout")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = extract_pruned_metadata(
        args.checkpoint,
        config_path=args.config,
        orbis_repo_path=args.orbis_repo,
    )
    payload = json.dumps(metadata, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
        print(output_path)
        return
    print(payload)


if __name__ == "__main__":
    main()