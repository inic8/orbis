from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Literal

import torch
from omegaconf import OmegaConf

from .bootstrap import OrbisModules, resolve_orbis_modules
from .contracts import (
    INTERFACE_VERSION,
    ArtifactDescriptor,
    ArtifactType,
    ComponentInterface,
    ComponentResult,
    LatencyImprovement,
    MemoryImprovement,
    PruningComponentResult,
    PruningMetrics,
    ComponentStatus,
    PipelineState,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRUNED_RUN_SUFFIX = "_pruned"

ImportanceMetric = Literal["l1_weight", "l2_weight", "random"]


@dataclass(frozen=True)
class OrbisPruningOptions:
    mlp_prune_ratio: float = 0.2
    mlp_round_to: int = 128
    head_prune_ratio: float = 0.0
    mlp_prune_layers: str = "all"
    head_prune_layers: str = "all"
    importance_metric: ImportanceMetric = "l1_weight"

    def build_structured_config(self, structured_config_cls: type) -> Any:
        return structured_config_cls(
            enabled=True,
            mlp_prune_ratio=self.mlp_prune_ratio,
            mlp_round_to=self.mlp_round_to,
            head_prune_ratio=self.head_prune_ratio,
            mlp_prune_layers=self.mlp_prune_layers,
            head_prune_layers=self.head_prune_layers,
            importance_metric=self.importance_metric,
            recovery_steps=0,
            recovery_lr_multiplier=0.1,
        )


@dataclass(frozen=True)
class OrbisPruningResult:
    output_dir: Path
    checkpoint_path: Path
    config_path: Path
    stats_path: Path
    summary_path: Path
    benchmark_path: Path | None
    stats: dict[str, Any]
    output_artifacts: list[ArtifactDescriptor]
    component_result: PruningComponentResult

    def to_component_result(self) -> PruningComponentResult:
        return self.component_result


def _artifact_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "directory"


def _build_output_artifacts(
    *,
    output_checkpoint_path: Path,
    output_config_path: Path,
    output_stats_path: Path,
    output_summary_path: Path,
    output_benchmark_path: Path | None,
    producer: str,
    stats: dict[str, Any],
) -> list[ArtifactDescriptor]:
    artifacts = [
        ArtifactDescriptor(
            name="pruned_checkpoint",
            type=ArtifactType.MODEL,
            path=str(output_checkpoint_path),
            format=_artifact_format(output_checkpoint_path),
            producer=producer,
            metadata={
                "reduction_pct": stats.get("reduction_pct"),
                "params_before": stats.get("params_before"),
                "params_after": stats.get("params_after"),
            },
        ),
        ArtifactDescriptor(
            name="pruned_config",
            type=ArtifactType.CONFIG,
            path=str(output_config_path),
            format=_artifact_format(output_config_path),
            producer=producer,
        ),
        ArtifactDescriptor(
            name="pruning_stats",
            type=ArtifactType.METRICS,
            path=str(output_stats_path),
            format=_artifact_format(output_stats_path),
            producer=producer,
        ),
        ArtifactDescriptor(
            name="model_structure_summary",
            type=ArtifactType.REPORT,
            path=str(output_summary_path),
            format=_artifact_format(output_summary_path),
            producer=producer,
        ),
    ]

    if output_benchmark_path is not None:
        artifacts.append(
            ArtifactDescriptor(
                name="benchmark_stats",
                type=ArtifactType.METRICS,
                path=str(output_benchmark_path),
                format=_artifact_format(output_benchmark_path),
                producer=producer,
                metadata={"benchmark": stats.get("benchmark")},
            )
        )

    return artifacts


def _build_component_result(
    *,
    output_artifacts: list[ArtifactDescriptor],
    stats: dict[str, Any],
    producer: str,
    output_dir: Path,
    benchmark_enabled: bool,
) -> PruningComponentResult:
    pruning_metrics = _build_pruning_metrics(stats)

    return PruningComponentResult(
        component_name=producer,
        status=ComponentStatus.SUCCESS,
        message=f"Pruning completed successfully in {output_dir}",
        output_artifacts=output_artifacts,
        metrics=asdict(pruning_metrics),
        metadata={
            "interface_version": INTERFACE_VERSION,
            "output_dir": str(output_dir),
            "benchmark_enabled": benchmark_enabled,
            "checkpoint_path": stats.get("checkpoint_path"),
            "config_path": stats.get("config_path"),
            "options": stats.get("options", {}),
        },
        pruning_metrics=pruning_metrics,
    )


def _build_latency_improvement(benchmark_summary: dict[str, Any] | None) -> LatencyImprovement | None:
    if benchmark_summary is None:
        return None

    latency = benchmark_summary.get("latency")
    if latency is None:
        return None

    return LatencyImprovement(
        before_ms=latency.get("before_ms"),
        after_ms=latency.get("after_ms"),
        absolute_ms=latency.get("absolute_ms"),
        factor=latency.get("factor"),
        reduction_pct=latency.get("reduction_pct"),
    )


def _build_memory_improvement(
    benchmark_summary: dict[str, Any] | None,
    key: str,
) -> MemoryImprovement | None:
    if benchmark_summary is None:
        return None

    memory = benchmark_summary.get(key)
    if memory is None:
        return None

    return MemoryImprovement(
        before_mb=memory.get("before_mb"),
        after_mb=memory.get("after_mb"),
        absolute_mb=memory.get("absolute_mb"),
        reduction_pct=memory.get("reduction_pct"),
    )


def _build_pruning_metrics(stats: dict[str, Any]) -> PruningMetrics:
    benchmark_summary = stats.get("benchmark")
    return PruningMetrics(
        params_before=int(stats.get("params_before", 0)),
        params_after=int(stats.get("params_after", 0)),
        parameter_reduction_pct=float(stats.get("reduction_pct", 0.0)),
        checkpoint_size_reduction_pct=stats.get("checkpoint_size_reduction_pct"),
        latency=_build_latency_improvement(benchmark_summary),
        parameter_memory=_build_memory_improvement(benchmark_summary, "parameter_memory"),
        peak_memory=_build_memory_improvement(benchmark_summary, "peak_memory"),
    )


def _build_pruning_options(params: dict[str, Any]) -> OrbisPruningOptions:
    return OrbisPruningOptions(
        mlp_prune_ratio=float(params.get("mlp_prune_ratio", 0.2)),
        mlp_round_to=int(params.get("mlp_round_to", 128)),
        head_prune_ratio=float(params.get("head_prune_ratio", 0.0)),
        mlp_prune_layers=str(params.get("mlp_prune_layers", "all")),
        head_prune_layers=str(params.get("head_prune_layers", "all")),
        importance_metric=str(params.get("importance_metric", "l1_weight")),
    )


class OrbisPruningComponent(ComponentInterface):
    def validate_params(self) -> None:
        output_dir = self.params.get("output_dir")
        checkpoint_path = self.params.get("checkpoint_path")
        if output_dir is None and checkpoint_path is None:
            return

        if output_dir is not None and not str(output_dir).strip():
            raise ValueError("output_dir must not be empty")
        if checkpoint_path is not None and not str(checkpoint_path).strip():
            raise ValueError("checkpoint_path must not be empty")

    def validate_inputs(self, state: PipelineState) -> None:
        checkpoint_path = self.params.get("checkpoint_path")
        if checkpoint_path is None and state.get_latest_artifact(ArtifactType.MODEL) is None:
            raise ValueError("A MODEL artifact or checkpoint_path parameter is required")

        if checkpoint_path is not None and not Path(checkpoint_path).expanduser().exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        config_path = self.params.get("config_path")
        if config_path is not None and not Path(config_path).expanduser().exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

    def run(self, state: PipelineState) -> PruningComponentResult:
        self.validate_params()
        self.validate_inputs(state)

        checkpoint_artifact = state.get_latest_artifact(ArtifactType.MODEL)
        config_artifact = state.get_latest_artifact(ArtifactType.CONFIG)

        checkpoint_path = self.params.get("checkpoint_path") or (checkpoint_artifact.path if checkpoint_artifact else None)
        config_path = self.params.get("config_path") or (config_artifact.path if config_artifact else None)
        output_dir = self.params.get("output_dir")
        orbis_repo_path = self.params.get("orbis_repo_path") or state.global_metadata.get("orbis_repo_path")
        run_benchmark = bool(self.params.get("run_benchmark", True))

        result = prune_orbis_checkpoint(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            options=_build_pruning_options(self.params),
            config_path=config_path,
            orbis_repo_path=orbis_repo_path,
            run_benchmark=run_benchmark,
        )
        state.add_result(result.component_result)
        return result.component_result

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "interface_version": INTERFACE_VERSION,
            "component": "pruning",
            "produces": [artifact_type.value for artifact_type in (ArtifactType.MODEL, ArtifactType.CONFIG, ArtifactType.METRICS, ArtifactType.REPORT)],
        }


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any] | Any:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _absolute_path(path: str | Path) -> Path:
    return Path(path).expanduser().absolute()


def _detect_config_path(checkpoint_path: Path, config_path: str | Path | None) -> Path:
    if config_path is not None:
        resolved = _absolute_path(config_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Config not found: {resolved}")
        return resolved

    candidates = [
        checkpoint_path.parent / "config.yaml",
        checkpoint_path.parent.parent / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.absolute()

    raise FileNotFoundError(
        f"Could not find config.yaml near checkpoint: {checkpoint_path}. "
        "Pass config_path explicitly."
    )


def _normalize_orbis_checkout(candidate: str | Path | None) -> Path | None:
    if candidate is None:
        return None

    path = Path(candidate).expanduser().resolve()
    if (path / "orbis" / "util.py").exists():
        return path / "orbis"
    if (path / "util.py").exists():
        return path
    return None


def _candidate_orbis_workspace_roots(
    *,
    checkpoint_path: Path,
    config_path: Path,
    orbis_repo_path: str | Path | None,
) -> list[Path]:
    candidates: list[Path] = []

    for anchor in [config_path, checkpoint_path]:
        for parent in anchor.parents:
            if parent.name in {"logs_tk", "logs_wm"}:
                workspace_root = parent.parent
                if workspace_root not in candidates:
                    candidates.append(workspace_root)

            detected_checkout = _normalize_orbis_checkout(parent)
            if detected_checkout is not None and detected_checkout not in candidates:
                candidates.append(detected_checkout)

    explicit_checkout = _normalize_orbis_checkout(orbis_repo_path)
    if explicit_checkout is not None and explicit_checkout not in candidates:
        candidates.append(explicit_checkout)

    env_checkout = _normalize_orbis_checkout(os.getenv("ORBIS_REPO_PATH"))
    if env_checkout is not None and env_checkout not in candidates:
        candidates.append(env_checkout)

    return candidates


def _strip_leading_current_dir(path: Path) -> Path:
    parts = list(path.parts)
    while parts and parts[0] == ".":
        parts.pop(0)
    return Path(*parts) if parts else Path()


def _resolve_workspace_path(raw_path: str | Path, workspace_roots: list[Path]) -> Path:
    expanded = Path(os.path.expandvars(str(raw_path))).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()

    relative_path = _strip_leading_current_dir(expanded)
    candidates = [root / relative_path for root in workspace_roots]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    if workspace_roots:
        return (workspace_roots[0] / relative_path).resolve()
    return expanded.resolve()


def _normalize_orbis_config_paths(config: Any, workspace_roots: list[Path]) -> Any:
    tokenizer_config = config.model.params.get("tokenizer_config")
    if tokenizer_config is None or not getattr(tokenizer_config, "folder", None):
        return config

    tokenizer_config.folder = str(_resolve_workspace_path(tokenizer_config.folder, workspace_roots))
    return config


def _configure_orbis_environment(
    *,
    checkpoint_path: Path,
    config_path: Path,
    orbis_repo_path: str | Path | None,
) -> None:
    candidates = _candidate_orbis_workspace_roots(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        orbis_repo_path=orbis_repo_path,
    )

    for checkout in candidates:
        os.environ.setdefault("ORBIS_REPO_PATH", str(checkout))

        logs_tk = checkout / "logs_tk"
        if logs_tk.exists():
            os.environ.setdefault("TK_WORK_DIR", str(logs_tk))

        logs_wm = checkout / "logs_wm"
        if logs_wm.exists():
            os.environ.setdefault("WM_WORK_DIR", str(logs_wm))

        data_dir = checkout / "data"
        if data_dir.exists():
            os.environ.setdefault("ORBIS_DATA_DIR", str(data_dir))

        break


def _detect_source_run_dir(checkpoint_path: Path, config_path: Path) -> Path | None:
    if checkpoint_path.parent.name == "checkpoints":
        candidate = checkpoint_path.parent.parent
        if candidate == config_path.parent and config_path.name == "config.yaml":
            return candidate

    if checkpoint_path.parent == config_path.parent and config_path.name == "config.yaml":
        return config_path.parent

    return None


def _build_pruned_run_name(source_name: str) -> str:
    if source_name.endswith(DEFAULT_PRUNED_RUN_SUFFIX):
        return source_name
    return f"{source_name}{DEFAULT_PRUNED_RUN_SUFFIX}"


def _resolve_output_dir(
    output_dir: str | Path | None,
    *,
    checkpoint_path: Path,
    config_path: Path,
    orbis_repo_path: str | Path | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()

    workspace_roots = _candidate_orbis_workspace_roots(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        orbis_repo_path=orbis_repo_path,
    )
    workspace_root = workspace_roots[0] if workspace_roots else REPO_ROOT
    source_run_dir = config_path.parent if config_path.name == "config.yaml" else _detect_source_run_dir(checkpoint_path, config_path)
    source_name = source_run_dir.name if source_run_dir is not None else checkpoint_path.stem
    return (workspace_root / "logs_wm" / _build_pruned_run_name(source_name)).resolve()


def _load_orbis_model(
    checkpoint_path: Path,
    config_path: Path,
    modules: OrbisModules,
    *,
    orbis_repo_path: str | Path | None = None,
):
    config = OmegaConf.load(config_path)
    workspace_roots = _candidate_orbis_workspace_roots(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        orbis_repo_path=orbis_repo_path,
    )
    _configure_orbis_environment(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        orbis_repo_path=orbis_repo_path,
    )
    config = _normalize_orbis_config_paths(config, workspace_roots)
    model = modules.instantiate_from_config(config.model)

    checkpoint = _load_checkpoint(checkpoint_path)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    return model, config


def _update_config_for_pruning(config: Any, stats: dict[str, Any]) -> Any:
    config_copy = deepcopy(config)
    gen_params = config_copy.model.params.generator_config.params

    target_space_heads = stats.get("target_space_heads", stats.get("target_num_heads"))
    target_time_heads = stats.get("target_time_heads", stats.get("target_num_heads"))
    original_heads = stats.get("original_num_heads")

    if target_space_heads != target_time_heads:
        gen_params._pruning_incompatible = True
        gen_params._space_heads = target_space_heads
        gen_params._time_heads = target_time_heads
    elif original_heads is not None and target_space_heads != original_heads:
        gen_params.num_heads = target_space_heads

    target_mlp_dim = stats.get("target_mlp_dim")
    original_mlp_dim = stats.get("original_mlp_dim")
    hidden_size = gen_params.get("hidden_size", 768)
    if target_mlp_dim is not None and original_mlp_dim is not None and target_mlp_dim != original_mlp_dim:
        gen_params.mlp_ratio = target_mlp_dim / hidden_size

    return config_copy


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = int(round((len(sorted_values) - 1) * percentile))
    return float(sorted_values[index])


def _parameter_memory_mb(module: Any) -> float:
    total_bytes = sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())
    return total_bytes / (1024 * 1024)


def _build_benchmark_inputs(model_config: Any, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    generator_params = model_config.model.params.generator_config.params
    input_size = generator_params.get("input_size", [16, 16])
    if isinstance(input_size, int):
        height = width = input_size
    else:
        height = int(input_size[0])
        width = int(input_size[1])

    in_channels = int(generator_params.get("in_channels", 32))
    max_num_frames = int(generator_params.get("max_num_frames", 2))
    context_frames = max(1, min(max_num_frames - 1, 4))

    torch.manual_seed(42)
    return {
        "target": torch.randn(1, 1, in_channels, height, width, device=device, dtype=dtype),
        "context": torch.randn(1, context_frames, in_channels, height, width, device=device, dtype=dtype),
        "t": torch.rand(1, device=device, dtype=dtype),
        "frame_rate": torch.ones(1, device=device, dtype=dtype),
    }


def _benchmark_vit(module: Any, model_config: Any) -> dict[str, Any]:
    def run_on_device(device: torch.device) -> dict[str, Any]:
        warmup_runs = 5 if device.type == "cuda" else 1
        timed_runs = 20 if device.type == "cuda" else 3

        module.to(device)
        module.eval()
        dtype = next(module.parameters()).dtype if any(True for _ in module.parameters()) else torch.float32
        inputs = _build_benchmark_inputs(model_config, device, dtype)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        with torch.inference_mode():
            for _ in range(warmup_runs):
                _ = module(inputs["target"], inputs["context"], inputs["t"], frame_rate=inputs["frame_rate"])
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            latencies_ms: list[float] = []
            for _ in range(timed_runs):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                _ = module(inputs["target"], inputs["context"], inputs["t"], frame_rate=inputs["frame_rate"])
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                latencies_ms.append((time.perf_counter() - start) * 1000.0)

        benchmark = {
            "device": device.type,
            "warmup_runs": warmup_runs,
            "timed_runs": timed_runs,
            "latency_ms": {
                "mean": float(mean(latencies_ms)),
                "median": float(median(latencies_ms)),
                "min": float(min(latencies_ms)),
                "max": float(max(latencies_ms)),
                "p95": _percentile(latencies_ms, 0.95),
            },
            "parameter_memory_mb": _parameter_memory_mb(module),
        }

        if device.type == "cuda":
            benchmark["peak_memory_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
            module.to("cpu")
            torch.cuda.empty_cache()
        else:
            benchmark["peak_memory_mb"] = None

        return benchmark

    if torch.cuda.is_available():
        try:
            return run_on_device(torch.device("cuda"))
        except Exception as error:
            if "cuda" not in str(error).lower() and "kernel image" not in str(error).lower():
                raise
            module.to("cpu")
            torch.cuda.empty_cache()
            cpu_benchmark = run_on_device(torch.device("cpu"))
            cpu_benchmark["fallback_reason"] = str(error)
            return cpu_benchmark

    return run_on_device(torch.device("cpu"))


def _build_improvement_summary(
    before: float | None,
    after: float | None,
    *,
    unit: str,
    include_factor: bool,
) -> dict[str, Any] | None:
    if before is None or after is None:
        return None

    summary = {
        f"before_{unit}": float(before),
        f"after_{unit}": float(after),
        f"absolute_{unit}": float(before - after),
        "reduction_pct": float((1 - after / before) * 100.0) if before > 0 else None,
    }
    if include_factor:
        summary["factor"] = float(before / after) if after > 0 else None

    return summary


def _build_benchmark_summary(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    latency_before = before["latency_ms"]["mean"]
    latency_after = after["latency_ms"]["mean"]
    parameter_memory_before = before["parameter_memory_mb"]
    parameter_memory_after = after["parameter_memory_mb"]

    peak_before = before.get("peak_memory_mb")
    peak_after = after.get("peak_memory_mb")

    latency_summary = _build_improvement_summary(latency_before, latency_after, unit="ms", include_factor=True)
    parameter_memory_summary = _build_improvement_summary(
        parameter_memory_before,
        parameter_memory_after,
        unit="mb",
        include_factor=False,
    )
    peak_memory_summary = _build_improvement_summary(peak_before, peak_after, unit="mb", include_factor=False)

    return {
        "benchmark_scope": "Synthetic model.vit forward pass",
        "before": before,
        "after": after,
        "summary": {
            "latency": latency_summary,
            "parameter_memory": parameter_memory_summary,
            "peak_memory": peak_memory_summary,
            "latency_speedup": latency_summary.get("factor") if latency_summary is not None else None,
            "latency_reduction_pct": latency_summary.get("reduction_pct") if latency_summary is not None else None,
            "parameter_memory_reduction_pct": parameter_memory_summary.get("reduction_pct")
            if parameter_memory_summary is not None
            else None,
            "peak_memory_reduction_pct": peak_memory_summary.get("reduction_pct") if peak_memory_summary is not None else None,
        },
    }


def prune_orbis_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    options: OrbisPruningOptions | None = None,
    config_path: str | Path | None = None,
    orbis_repo_path: str | Path | None = None,
    run_benchmark: bool = True,
) -> OrbisPruningResult:
    checkpoint_path = _absolute_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    resolved_config_path = _detect_config_path(checkpoint_path, config_path)
    modules = resolve_orbis_modules(
        orbis_repo_path=orbis_repo_path,
        checkpoint_path=checkpoint_path,
    )
    pruning_options = options or OrbisPruningOptions()

    model, model_config = _load_orbis_model(
        checkpoint_path,
        resolved_config_path,
        modules,
        orbis_repo_path=orbis_repo_path,
    )

    if not hasattr(model, "vit"):
        raise AttributeError("Loaded model does not expose a vit attribute required for Orbis structured pruning.")

    params_before = sum(parameter.numel() for parameter in model.vit.parameters())
    checkpoint_size_before_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    benchmark_results = None
    if run_benchmark:
        benchmark_results = {"before": _benchmark_vit(model.vit, model_config)}

    prune_config = pruning_options.build_structured_config(modules.StructuredPruningConfig)
    model, stats = modules.apply_structured_pruning(model, prune_config)

    params_after = sum(parameter.numel() for parameter in model.vit.parameters())
    reduction_pct = (1 - params_after / params_before) * 100.0 if params_before > 0 else 0.0

    if benchmark_results is not None:
        benchmark_results = _build_benchmark_summary(benchmark_results["before"], _benchmark_vit(model.vit, model_config))

    stats = dict(stats)
    stats.update(
        {
            "checkpoint_path": str(checkpoint_path),
            "config_path": str(resolved_config_path),
            "params_before": int(params_before),
            "params_after": int(params_after),
            "reduction_pct": float(reduction_pct),
            "checkpoint_size_mb_before": float(checkpoint_size_before_mb),
            "options": asdict(pruning_options),
        }
    )

    output_dir = _resolve_output_dir(
        output_dir,
        checkpoint_path=checkpoint_path,
        config_path=resolved_config_path,
        orbis_repo_path=orbis_repo_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    onnx_dir = output_dir / "onnx"
    pruning_dir = output_dir / "pruning"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir.mkdir(parents=True, exist_ok=True)
    pruning_dir.mkdir(parents=True, exist_ok=True)

    output_checkpoint_path = checkpoints_dir / "last.ckpt"
    output_config_path = output_dir / "config.yaml"
    output_stats_path = pruning_dir / "pruning_stats.json"
    output_summary_path = pruning_dir / "model_structure_summary.json"
    output_benchmark_path = pruning_dir / "benchmark_stats.json"

    model = model.to("cpu")

    torch.save(
        {
            "state_dict": model.state_dict(),
            "pruning_stats": stats,
        },
        output_checkpoint_path,
    )
    OmegaConf.save(_update_config_for_pruning(model_config, stats), output_config_path)
    checkpoint_size_after_mb = output_checkpoint_path.stat().st_size / (1024 * 1024)
    stats["checkpoint_size_mb_after"] = float(checkpoint_size_after_mb)
    stats["checkpoint_size_reduction_pct"] = float((1 - checkpoint_size_after_mb / checkpoint_size_before_mb) * 100.0)
    if benchmark_results is not None:
        stats["benchmark"] = benchmark_results["summary"]
    _write_json(output_stats_path, stats)
    _write_json(output_summary_path, modules.get_pruning_summary(model))
    if benchmark_results is not None:
        _write_json(output_benchmark_path, benchmark_results)

    output_artifacts = _build_output_artifacts(
        output_checkpoint_path=output_checkpoint_path,
        output_config_path=output_config_path,
        output_stats_path=output_stats_path,
        output_summary_path=output_summary_path,
        output_benchmark_path=output_benchmark_path if benchmark_results is not None else None,
        producer="pruning",
        stats=stats,
    )
    component_result = _build_component_result(
        output_artifacts=output_artifacts,
        stats=stats,
        producer="pruning",
        output_dir=output_dir,
        benchmark_enabled=benchmark_results is not None,
    )

    return OrbisPruningResult(
        output_dir=output_dir,
        checkpoint_path=output_checkpoint_path,
        config_path=output_config_path,
        stats_path=output_stats_path,
        summary_path=output_summary_path,
        benchmark_path=output_benchmark_path if benchmark_results is not None else None,
        stats=stats,
        output_artifacts=output_artifacts,
        component_result=component_result,
    )
