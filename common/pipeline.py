from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import uuid

from .base import BaseComponent, PipelineState
from .registry import register_component
from .utils import ensure_dir, write_json
from pruning.api import OrbisPruningOptions, prune_orbis_checkpoint
from rank_adaptation.api import OrbisRankAdaptationOptions, rank_adapt_orbis_checkpoint
from rank_adaptation.api import _absolute_path as _rank_absolute_path
from rank_adaptation.api import _benchmark_vit, _detect_config_path, _load_orbis_model
from rank_adaptation.bootstrap import resolve_orbis_modules
from rank_adaptation.transformer_layers import count_all_params, count_linear_params, count_vit_params


@dataclass(frozen=True)
class CheckpointEvaluation:
    checkpoint_path: str
    config_path: str
    params: int
    vit_params: int
    linear_params: int
    checkpoint_size_mb: float
    benchmark: dict[str, Any]


def evaluate_orbis_checkpoint(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path | None = None,
    orbis_repo_path: str | Path | None = None,
) -> CheckpointEvaluation:
    checkpoint = _rank_absolute_path(checkpoint_path)
    resolved_config_path = _detect_config_path(checkpoint, config_path)
    modules = resolve_orbis_modules(orbis_repo_path=orbis_repo_path, checkpoint_path=checkpoint)
    model, model_config = _load_orbis_model(
        checkpoint,
        resolved_config_path,
        modules,
        orbis_repo_path=orbis_repo_path,
    )
    benchmark = _benchmark_vit(model, model_config)
    return CheckpointEvaluation(
        checkpoint_path=str(checkpoint),
        config_path=str(resolved_config_path),
        params=count_all_params(model),
        vit_params=count_vit_params(model),
        linear_params=count_linear_params(model),
        checkpoint_size_mb=float(checkpoint.stat().st_size / (1024 * 1024)),
        benchmark=benchmark,
    )


def _safe_pct(before: float | int | None, after: float | int | None) -> float | None:
    if before is None or after is None:
        return None
    before_value = float(before)
    after_value = float(after)
    if before_value == 0.0:
        return None
    return float((1.0 - after_value / before_value) * 100.0)


def build_pipeline_evaluation_summary(
    before: CheckpointEvaluation,
    after: CheckpointEvaluation,
) -> dict[str, Any]:
    before_latency = before.benchmark.get("latency_ms", {}).get("mean")
    after_latency = after.benchmark.get("latency_ms", {}).get("mean")
    before_param_memory = before.benchmark.get("parameter_memory_mb")
    after_param_memory = after.benchmark.get("parameter_memory_mb")
    before_peak = before.benchmark.get("peak_memory_mb")
    after_peak = after.benchmark.get("peak_memory_mb")

    return {
        "before": asdict(before),
        "after": asdict(after),
        "summary": {
            "latency_before_ms": before_latency,
            "latency_after_ms": after_latency,
            "latency_reduction_pct": _safe_pct(before_latency, after_latency),
            "parameter_memory_before_mb": before_param_memory,
            "parameter_memory_after_mb": after_param_memory,
            "parameter_memory_reduction_pct": _safe_pct(before_param_memory, after_param_memory),
            "peak_memory_before_mb": before_peak,
            "peak_memory_after_mb": after_peak,
            "peak_memory_reduction_pct": _safe_pct(before_peak, after_peak),
            "params_before": before.params,
            "params_after": after.params,
            "parameter_reduction_pct": _safe_pct(before.params, after.params),
            "linear_params_before": before.linear_params,
            "linear_params_after": after.linear_params,
            "linear_parameter_reduction_pct": _safe_pct(before.linear_params, after.linear_params),
            "checkpoint_size_before_mb": before.checkpoint_size_mb,
            "checkpoint_size_after_mb": after.checkpoint_size_mb,
            "checkpoint_size_reduction_pct": _safe_pct(before.checkpoint_size_mb, after.checkpoint_size_mb),
        },
    }


def _step_output_dir(root_output_dir: str | Path, step_index: int, step_name: str) -> Path:
    return Path(root_output_dir).expanduser().resolve() / f"{step_index:02d}_{step_name}"


@register_component("rank_adaptation")
class RankAdaptationPipelineComponent(BaseComponent):
    def run(self, state: PipelineState) -> PipelineState:
        step_index = int(self.params.get("step_index", len(state.history) + 1))
        step_output_dir = _step_output_dir(state.output_dir, step_index, self.name)
        default_options = OrbisRankAdaptationOptions()
        layer_patterns = self.params.get("layer_patterns")
        skip_patterns = self.params.get("skip_patterns")
        options = OrbisRankAdaptationOptions(
            acc_budget_pct=float(self.params.get("acc_budget_pct", 2.0)),
            comp_target=float(self.params.get("comp_target", 2.0)),
            rank_step_fraction=float(self.params.get("rank_step_fraction", 0.20)),
            layer_patterns=list(default_options.layer_patterns if layer_patterns is None else layer_patterns),
            skip_patterns=list(default_options.skip_patterns if skip_patterns is None else skip_patterns),
            min_features=int(self.params.get("min_features", 64)),
            batch_size=int(self.params.get("batch_size", 4)),
            run_benchmark=bool(self.params.get("run_benchmark", False)),
        )
        result = rank_adapt_orbis_checkpoint(
            checkpoint_path=state.model_path,
            output_dir=step_output_dir,
            options=options,
            config_path=state.config_path,
            orbis_repo_path=self.params.get("orbis_repo_path"),
            vit_attr=str(self.params.get("vit_attr", "vit")),
            checkpoint_dir=self.params.get("checkpoint_dir"),
            skip_phase1=bool(self.params.get("skip_phase1", False)),
        )
        state.model_path = str(result.checkpoint_path)
        state.config_path = str(result.config_path)
        state.current_run_dir = str(result.output_dir)
        state.metadata.setdefault("artifacts", {})[self.name] = {
            "checkpoint": str(result.checkpoint_path),
            "config": str(result.config_path),
            "stats": str(result.stats_path),
            "summary": str(result.summary_path),
            "benchmark": str(result.benchmark_path) if result.benchmark_path is not None else None,
        }
        state.record(
            self.name,
            "success",
            {
                "output_dir": str(result.output_dir),
                "checkpoint_path": str(result.checkpoint_path),
                "config_path": str(result.config_path),
                "stats_path": str(result.stats_path),
                "summary_path": str(result.summary_path),
                "metrics": result.stats,
            },
        )
        return state


@register_component("pruning")
class PruningPipelineComponent(BaseComponent):
    def run(self, state: PipelineState) -> PipelineState:
        step_index = int(self.params.get("step_index", len(state.history) + 1))
        step_output_dir = _step_output_dir(state.output_dir, step_index, self.name)
        options = OrbisPruningOptions(
            mlp_prune_ratio=float(self.params.get("mlp_prune_ratio", 0.2)),
            mlp_round_to=int(self.params.get("mlp_round_to", 128)),
            head_prune_ratio=float(self.params.get("head_prune_ratio", 0.0)),
            mlp_prune_layers=str(self.params.get("mlp_prune_layers", "all")),
            head_prune_layers=str(self.params.get("head_prune_layers", "all")),
            importance_metric=str(self.params.get("importance_metric", "l1_weight")),
        )
        result = prune_orbis_checkpoint(
            checkpoint_path=state.model_path,
            output_dir=step_output_dir,
            options=options,
            config_path=state.config_path,
            orbis_repo_path=self.params.get("orbis_repo_path"),
            run_benchmark=bool(self.params.get("run_benchmark", False)),
        )
        state.model_path = str(result.checkpoint_path)
        state.config_path = str(result.config_path)
        state.current_run_dir = str(result.output_dir)
        state.metadata.setdefault("artifacts", {})[self.name] = {
            "checkpoint": str(result.checkpoint_path),
            "config": str(result.config_path),
            "stats": str(result.stats_path),
            "summary": str(result.summary_path),
            "benchmark": str(result.benchmark_path) if result.benchmark_path is not None else None,
        }
        state.record(
            self.name,
            "success",
            {
                "output_dir": str(result.output_dir),
                "checkpoint_path": str(result.checkpoint_path),
                "config_path": str(result.config_path),
                "stats_path": str(result.stats_path),
                "summary_path": str(result.summary_path),
                "metrics": result.stats,
            },
        )
        return state


def run_pipeline(
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    config_path: str | Path | None,
    steps: list[BaseComponent],
    pipeline_name: str = "optimization_pipeline",
    metadata: dict[str, Any] | None = None,
) -> PipelineState:
    root_output_dir = Path(output_dir).expanduser().resolve()
    ensure_dir(str(root_output_dir))
    source_checkpoint_path = Path(checkpoint_path).expanduser().absolute()
    source_config_path = Path(config_path).expanduser().absolute() if config_path is not None else None
    state = PipelineState(
        model_path=str(source_checkpoint_path),
        config_path=str(source_config_path) if source_config_path is not None else None,
        output_dir=str(root_output_dir),
        current_run_dir=None,
        metadata={
            "pipeline_name": pipeline_name,
            "run_id": str(uuid.uuid4()),
            **(metadata or {}),
        },
    )
    for step in steps:
        state = step.run(state)
    return state


def write_pipeline_summary(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path).expanduser().resolve()
    write_json(str(output_path), payload)
    return output_path