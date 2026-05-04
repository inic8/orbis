# SPDX-License-Identifier: MIT
# Author: Dr Shashank Pathak
# Email: shashank@computer.org
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

from __future__ import annotations

import argparse

try:
    from .api import OrbisPruningOptions, prune_orbis_checkpoint
except ImportError:
    from api import OrbisPruningOptions, prune_orbis_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prune an Orbis checkpoint into a repo-local run directory")
    parser.add_argument("--checkpoint", required=True, help="Path to the source checkpoint")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output run directory; defaults to logs_wm/<source_run>_pruned",
    )
    parser.add_argument("--config", default=None, help="Optional path to config.yaml")
    parser.add_argument("--orbis-repo", default=None, help="Optional path to an Orbis checkout or its parent repo root")
    parser.add_argument("--mlp-prune-ratio", type=float, default=0.2, help="Fraction of MLP hidden width to prune")
    parser.add_argument("--mlp-round-to", type=int, default=128, help="Round the pruned MLP width to a multiple of this value")
    parser.add_argument("--head-prune-ratio", type=float, default=0.0, help="Fraction of attention heads to prune")
    parser.add_argument("--mlp-prune-layers", default="all", help="Which MLP layers to prune: all, space, or time")
    parser.add_argument("--head-prune-layers", default="all", help="Which attention layers to prune: all, space, or time")
    parser.add_argument(
        "--importance-metric",
        default="l1_weight",
        choices=["l1_weight", "l2_weight", "random"],
        help="Importance metric used by the structured pruning logic",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip the synthetic before/after latency and memory benchmark",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = prune_orbis_checkpoint(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        config_path=args.config,
        orbis_repo_path=args.orbis_repo,
        run_benchmark=not args.skip_benchmark,
        options=OrbisPruningOptions(
            mlp_prune_ratio=args.mlp_prune_ratio,
            mlp_round_to=args.mlp_round_to,
            head_prune_ratio=args.head_prune_ratio,
            mlp_prune_layers=args.mlp_prune_layers,
            head_prune_layers=args.head_prune_layers,
            importance_metric=args.importance_metric,
        ),
    )
    print(f"Pruned checkpoint: {result.checkpoint_path}")
    print(f"Config: {result.config_path}")
    print(f"Stats: {result.stats_path}")
    if result.benchmark_path is not None:
        print(f"Benchmark: {result.benchmark_path}")
    print(f"Parameter reduction: {result.stats['reduction_pct']:.2f}%")

    checkpoint_reduction = result.stats.get("checkpoint_size_reduction_pct")
    if checkpoint_reduction is not None:
        print(f"Checkpoint size reduction: {checkpoint_reduction:.2f}%")

    benchmark_summary = result.stats.get("benchmark")
    if benchmark_summary is not None:
        latency_reduction = benchmark_summary.get("latency_reduction_pct")
        latency_speedup = benchmark_summary.get("latency_speedup")
        memory_reduction = benchmark_summary.get("parameter_memory_reduction_pct")
        peak_memory_reduction = benchmark_summary.get("peak_memory_reduction_pct")

        if latency_reduction is not None and latency_speedup is not None:
            print(f"Latency change: {latency_reduction:.2f}% ({latency_speedup:.2f}x speedup)")
        if memory_reduction is not None:
            print(f"Parameter memory reduction: {memory_reduction:.2f}%")
        if peak_memory_reduction is not None:
            print(f"Peak memory reduction: {peak_memory_reduction:.2f}%")


if __name__ == "__main__":
    main()