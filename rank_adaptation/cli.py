# SPDX-License-Identifier: MIT
# Author: Arunachalam Thirunavukkarasu
# Contributor: Dr Shashank Pathak
# Email: arunachalam.thirunavukkarasu@dlr.de
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

from __future__ import annotations

import argparse

try:
    from .api import OrbisRankAdaptationOptions, rank_adapt_orbis_checkpoint
except ImportError:
    from api import OrbisRankAdaptationOptions, rank_adapt_orbis_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank adapt an Orbis checkpoint into a repo-local run directory")
    parser.add_argument("--checkpoint", required=True, help="Path to the source checkpoint")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output run directory; defaults to logs_wm/<source_run>_rank_adapted",
    )
    parser.add_argument("--config", default=None, help="Optional path to config.yaml")
    parser.add_argument("--orbis-repo", default=None, help="Optional path to an Orbis checkout or its parent repo root")
    parser.add_argument("--acc-budget-pct", type=float, default=2.0, help="Global accuracy loss budget as a percent")
    parser.add_argument("--comp-target", type=float, default=2.0, help="Target overall compression ratio for compressible Linear layers")
    parser.add_argument("--rank-step-fraction", type=float, default=0.20, help="Rank sweep step size as a fraction of min(in_features, out_features)")
    parser.add_argument("--min-features", type=int, default=64, help="Minimum Linear feature size to consider for compression")
    parser.add_argument("--batch-size", type=int, default=4, help="Synthetic evaluation batch size carried in the saved options")
    parser.add_argument("--vit-attr", default="vit", help="Attribute name of the ViT backbone on the loaded Orbis model")
    parser.add_argument("--checkpoint-dir", default=None, help="Optional directory with cached phase-1 sweep JSON checkpoints")
    parser.add_argument("--skip-phase1", action="store_true", help="Skip the phase-1 rank sweep and rely on checkpoint-dir only")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip the synthetic before/after latency and memory benchmark")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = rank_adapt_orbis_checkpoint(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        config_path=args.config,
        orbis_repo_path=args.orbis_repo,
        checkpoint_dir=args.checkpoint_dir,
        skip_phase1=args.skip_phase1,
        vit_attr=args.vit_attr,
        options=OrbisRankAdaptationOptions(
            acc_budget_pct=args.acc_budget_pct,
            comp_target=args.comp_target,
            rank_step_fraction=args.rank_step_fraction,
            min_features=args.min_features,
            batch_size=args.batch_size,
            run_benchmark=not args.skip_benchmark,
        ),
    )

    print(f"Rank-adapted checkpoint: {result.checkpoint_path}")
    print(f"Config: {result.config_path}")
    print(f"Stats: {result.stats_path}")
    print(f"Summary: {result.summary_path}")
    if result.benchmark_path is not None:
        print(f"Benchmark: {result.benchmark_path}")
    print(f"Parameter reduction: {result.stats['reduction_pct']:.2f}%")
    print(f"Linear parameter reduction: {result.stats['linear_reduction_pct']:.2f}%")
    print(f"Layers compressed: {result.stats['layers_compressed']}")
    print(f"Layers skipped: {result.stats['layers_skipped']}")


if __name__ == "__main__":
    main()