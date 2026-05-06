# SPDX-License-Identifier: MIT
# Author: Dr Shashank Pathak
# See LICENSE for the full MIT license text.

from __future__ import annotations

import argparse
from pathlib import Path

from common.pipeline import (
    PruningPipelineComponent,
    RankAdaptationPipelineComponent,
    build_pipeline_evaluation_summary,
    evaluate_orbis_checkpoint,
    run_pipeline,
    write_pipeline_summary,
)


def _default_output_dir(checkpoint_path: str | Path) -> Path:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if checkpoint.parent.name == "checkpoints" and checkpoint.parent.parent.parent.name == "logs_wm":
        source_run = checkpoint.parent.parent
        return source_run.parent / f"{source_run.name}_optimized"
    return checkpoint.parent / f"{checkpoint.stem}_optimized"


def _parse_csv_steps(value: str) -> list[str]:
    steps = [item.strip() for item in value.split(",") if item.strip()]
    valid = {"rank_adaptation", "pruning"}
    invalid = [step for step in steps if step not in valid]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported step(s): {', '.join(invalid)}")
    if not steps:
        raise argparse.ArgumentTypeError("At least one step must be provided")
    return steps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Orbis optimization steps such as pruning and rank adaptation in a configurable order")
    parser.add_argument("--checkpoint", required=True, help="Path to the source checkpoint")
    parser.add_argument("--config", default=None, help="Optional config.yaml path")
    parser.add_argument("--output-dir", default=None, help="Pipeline output root; defaults next to the source run")
    parser.add_argument("--orbis-repo", default=None, help="Optional path to an Orbis checkout or repo root")
    parser.add_argument(
        "--steps",
        type=_parse_csv_steps,
        default=["rank_adaptation", "pruning"],
        help="Comma-separated optimization order, for example rank_adaptation,pruning or pruning,rank_adaptation",
    )
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip before/after latency and memory evaluation")

    parser.add_argument("--acc-budget-pct", type=float, default=2.0)
    parser.add_argument("--comp-target", type=float, default=2.0)
    parser.add_argument("--rank-step-fraction", type=float, default=0.20)
    parser.add_argument("--rank-layer-pattern", action="append", default=None, help="Repeat to limit rank adaptation to matching layer names")
    parser.add_argument("--rank-skip-pattern", action="append", default=None, help="Repeat to exclude matching layer names from rank adaptation")
    parser.add_argument("--min-features", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vit-attr", default="vit")
    parser.add_argument("--checkpoint-dir", default=None, help="Optional Phase-1 checkpoint cache for rank adaptation")
    parser.add_argument("--skip-phase1", action="store_true", help="Skip fresh Phase-1 sweep generation for rank adaptation")

    parser.add_argument("--mlp-prune-ratio", type=float, default=0.2)
    parser.add_argument("--mlp-round-to", type=int, default=128)
    parser.add_argument("--head-prune-ratio", type=float, default=0.0)
    parser.add_argument("--mlp-prune-layers", default="all")
    parser.add_argument("--head-prune-layers", default="all")
    parser.add_argument(
        "--importance-metric",
        default="l1_weight",
        choices=["l1_weight", "l2_weight", "random"],
    )
    return parser


def _build_steps(args: argparse.Namespace):
    steps = []
    for index, step_name in enumerate(args.steps, start=1):
        if step_name == "rank_adaptation":
            steps.append(
                RankAdaptationPipelineComponent(
                    "rank_adaptation",
                    {
                        "step_index": index,
                        "orbis_repo_path": args.orbis_repo,
                        "acc_budget_pct": args.acc_budget_pct,
                        "comp_target": args.comp_target,
                        "rank_step_fraction": args.rank_step_fraction,
                        "layer_patterns": args.rank_layer_pattern,
                        "skip_patterns": args.rank_skip_pattern,
                        "min_features": args.min_features,
                        "batch_size": args.batch_size,
                        "vit_attr": args.vit_attr,
                        "checkpoint_dir": args.checkpoint_dir,
                        "skip_phase1": args.skip_phase1,
                        "run_benchmark": False,
                    },
                )
            )
        elif step_name == "pruning":
            steps.append(
                PruningPipelineComponent(
                    "pruning",
                    {
                        "step_index": index,
                        "orbis_repo_path": args.orbis_repo,
                        "mlp_prune_ratio": args.mlp_prune_ratio,
                        "mlp_round_to": args.mlp_round_to,
                        "head_prune_ratio": args.head_prune_ratio,
                        "mlp_prune_layers": args.mlp_prune_layers,
                        "head_prune_layers": args.head_prune_layers,
                        "importance_metric": args.importance_metric,
                        "run_benchmark": False,
                    },
                )
            )
    return steps


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else _default_output_dir(args.checkpoint)
    steps = _build_steps(args)

    before_eval = None
    if not args.skip_evaluation:
        before_eval = evaluate_orbis_checkpoint(
            args.checkpoint,
            config_path=args.config,
            orbis_repo_path=args.orbis_repo,
        )

    state = run_pipeline(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=output_dir,
        steps=steps,
        metadata={
            "requested_steps": args.steps,
            "orbis_repo_path": args.orbis_repo,
        },
    )

    after_eval = None
    evaluation_summary = None
    if before_eval is not None:
        after_eval = evaluate_orbis_checkpoint(
            state.model_path,
            config_path=state.config_path,
            orbis_repo_path=args.orbis_repo,
        )
        evaluation_summary = build_pipeline_evaluation_summary(before_eval, after_eval)

    payload = {
        "pipeline": {
            "name": "optimization_pipeline",
            "steps": args.steps,
            "output_dir": str(output_dir),
            "final_checkpoint": state.model_path,
            "final_config": state.config_path,
        },
        "history": state.history,
        "artifacts": state.metadata.get("artifacts", {}),
        "evaluation": evaluation_summary,
    }
    summary_path = write_pipeline_summary(output_dir / "pipeline_summary.json", payload)

    print(f"Pipeline output: {output_dir}")
    print(f"Final checkpoint: {state.model_path}")
    if state.config_path is not None:
        print(f"Final config: {state.config_path}")
    print(f"Summary: {summary_path}")
    if evaluation_summary is not None:
        summary = evaluation_summary["summary"]
        if summary.get("latency_reduction_pct") is not None:
            print(f"Latency reduction: {summary['latency_reduction_pct']:.2f}%")
        if summary.get("parameter_memory_reduction_pct") is not None:
            print(f"Parameter memory reduction: {summary['parameter_memory_reduction_pct']:.2f}%")
        if summary.get("peak_memory_reduction_pct") is not None:
            print(f"Peak memory reduction: {summary['peak_memory_reduction_pct']:.2f}%")
        if summary.get("parameter_reduction_pct") is not None:
            print(f"Parameter reduction: {summary['parameter_reduction_pct']:.2f}%")


if __name__ == "__main__":
    main()