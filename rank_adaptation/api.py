# SPDX-License-Identifier: MIT
# Author: Arunachalam Thirunavukkarasu
# Contributor: Dr Shashank Pathak
# Email: arunachalam.thirunavukkarasu@dlr.de
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch
from omegaconf import OmegaConf

from .bootstrap import OrbisModules, resolve_orbis_modules
from .checkpoints_linear import (
    load_layer_results_from_checkpoints,
    save_layer_checkpoint,
)
from .compat import apply_low_rank_metadata, collect_low_rank_metadata, extract_low_rank_metadata_from_checkpoint
from .compress_linear import (
    distribute_budget,
    sequential_compress_linear,
)
from .contracts import (
    INTERFACE_VERSION,
    ArtifactDescriptor,
    ArtifactType,
    ComponentInterface,
    ComponentStatus,
    LatencyImprovement,
    MemoryImprovement,
    PipelineState,
    RankAdaptationComponentResult,
    RankAdaptationMetrics,
)
from .polynomial import fit_biquadratic_polynomial
from .serialization import make_json_serializable
from .sweep_linear import (
    _orbis_forward,
    build_synthetic_vit_inputs,
    evaluate_accuracy_vit,
    sweep_linear_layer,
)
from .transformer_layers import (
    count_all_params,
    count_linear_params,
    count_vit_params,
    get_compressible_linear_layers,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RANK_ADAPTED_RUN_SUFFIX = "_rank_adapted"


@dataclass(frozen=True)
class OrbisRankAdaptationOptions:
    acc_budget_pct: float = 2.0
    comp_target: float = 2.0
    rank_step_fraction: float = 0.20
    layer_patterns: list[str] = field(
        default_factory=lambda: [
            "attn.qkv",
            "attn.proj",
            "mlp.fc1",
            "mlp.fc2",
            "attn.q_proj",
            "attn.k_proj",
            "attn.v_proj",
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.out_proj",
            "in_proj",
            "out_proj",
        ]
    )
    skip_patterns: list[str] = field(
        default_factory=lambda: ["head", "cls_token", "pos_embed", "patch_embed", "norm"]
    )
    min_features: int = 64
    batch_size: int = 4
    run_benchmark: bool = True


@dataclass(frozen=True)
class OrbisRankAdaptationResult:
    output_dir: Path
    checkpoint_path: Path
    config_path: Path
    stats_path: Path
    summary_path: Path
    benchmark_path: Path | None
    stats: dict[str, Any]
    output_artifacts: list[ArtifactDescriptor]
    component_result: RankAdaptationComponentResult

    def to_component_result(self) -> RankAdaptationComponentResult:
        return self.component_result


def _artifact_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "directory"


def _absolute_path(path: str | Path) -> Path:
    return Path(path).expanduser().absolute()


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any] | Any:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


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
        f"Could not find config.yaml near checkpoint: {checkpoint_path}. Pass config_path explicitly."
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
    if expanded.is_absolute() and expanded.exists():
        return expanded.resolve()

    relative_path = _strip_leading_current_dir(expanded)
    if expanded.is_absolute():
        parts = expanded.parts
        for anchor in ("logs_tk", "logs_wm", "data"):
            if anchor in parts:
                relative_path = Path(*parts[parts.index(anchor):])
                break

    candidates = [root / relative_path for root in workspace_roots]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    if workspace_roots:
        return (workspace_roots[0] / relative_path).resolve()
    return expanded.resolve()


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


def _normalize_orbis_config_paths(
    config: Any,
    checkpoint_path: Path,
    config_path: Path,
    orbis_repo_path: str | Path | None,
) -> Any:
    workspace_roots = _candidate_orbis_workspace_roots(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        orbis_repo_path=orbis_repo_path,
    )
    try:
        tokenizer_config = config.model.params.get("tokenizer_config")
        if tokenizer_config and getattr(tokenizer_config, "folder", None):
            tokenizer_config.folder = str(_resolve_workspace_path(tokenizer_config.folder, workspace_roots))
    except Exception:
        pass
    return config


def _select_orbis_runtime_cwd(
    *,
    checkpoint_path: Path,
    config_path: Path,
    orbis_repo_path: str | Path | None,
) -> Path:
    candidates = _candidate_orbis_workspace_roots(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        orbis_repo_path=orbis_repo_path,
    )

    for candidate in candidates:
        if (candidate / "logs_tk").exists() or (candidate / "logs_wm").exists():
            return candidate

    return config_path.parent


def _load_orbis_model(
    checkpoint_path: Path,
    config_path: Path,
    modules: OrbisModules,
    *,
    orbis_repo_path: str | Path | None = None,
):
    config = OmegaConf.load(config_path)
    _configure_orbis_environment(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        orbis_repo_path=orbis_repo_path,
    )
    config = _normalize_orbis_config_paths(config, checkpoint_path, config_path, orbis_repo_path)
    previous_cwd = Path.cwd()
    runtime_cwd = _select_orbis_runtime_cwd(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        orbis_repo_path=orbis_repo_path,
    )

    try:
        os.chdir(runtime_cwd)
        model = modules.instantiate_from_config(config.model)
    finally:
        os.chdir(previous_cwd)

    checkpoint = _load_checkpoint(checkpoint_path)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    low_rank_metadata = extract_low_rank_metadata_from_checkpoint(checkpoint)
    if low_rank_metadata:
        apply_low_rank_metadata(model, low_rank_metadata)
    model.load_state_dict(state_dict, strict=False)
    return model, config


def _detect_source_run_dir(checkpoint_path: Path, config_path: Path) -> Path | None:
    if checkpoint_path.parent.name == "checkpoints":
        candidate = checkpoint_path.parent.parent
        if candidate == config_path.parent and config_path.name == "config.yaml":
            return candidate

    if checkpoint_path.parent == config_path.parent and config_path.name == "config.yaml":
        return config_path.parent

    return None


def _build_rank_adapted_run_name(source_name: str) -> str:
    if source_name.endswith(DEFAULT_RANK_ADAPTED_RUN_SUFFIX):
        return source_name
    return f"{source_name}{DEFAULT_RANK_ADAPTED_RUN_SUFFIX}"


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
    return (workspace_root / "logs_wm" / _build_rank_adapted_run_name(source_name)).resolve()


def _parameter_memory_mb(module: Any) -> float:
    total_bytes = sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())
    return total_bytes / (1024 * 1024)


def _benchmark_vit(module: Any, model_config: Any) -> dict[str, Any]:
    def run_on_device(device: torch.device) -> dict[str, Any]:
        warmup_runs = 5 if device.type == "cuda" else 1
        timed_runs = 20 if device.type == "cuda" else 3

        module.to(device)
        module.eval()
        dtype = next(module.parameters()).dtype if any(True for _ in module.parameters()) else torch.float32
        inputs = build_synthetic_vit_inputs(model_config, device, dtype)

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        with torch.inference_mode():
            for _ in range(warmup_runs):
                _ = _orbis_forward(module, inputs)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            latencies_ms: list[float] = []
            for _ in range(timed_runs):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                _ = _orbis_forward(module, inputs)
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
        except Exception:
            return run_on_device(torch.device("cpu"))

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
    latency_summary = _build_improvement_summary(
        before["latency_ms"]["mean"],
        after["latency_ms"]["mean"],
        unit="ms",
        include_factor=True,
    )
    parameter_memory_summary = _build_improvement_summary(
        before["parameter_memory_mb"],
        after["parameter_memory_mb"],
        unit="mb",
        include_factor=False,
    )
    peak_memory_summary = _build_improvement_summary(
        before.get("peak_memory_mb"),
        after.get("peak_memory_mb"),
        unit="mb",
        include_factor=False,
    )
    return {
        "benchmark_scope": "Synthetic Orbis forward pass",
        "before": before,
        "after": after,
        "summary": {
            "latency": latency_summary,
            "parameter_memory": parameter_memory_summary,
            "peak_memory": peak_memory_summary,
            "latency_speedup": latency_summary.get("factor") if latency_summary is not None else None,
            "latency_reduction_pct": latency_summary.get("reduction_pct") if latency_summary is not None else None,
            "parameter_memory_reduction_pct": parameter_memory_summary.get("reduction_pct") if parameter_memory_summary is not None else None,
            "peak_memory_reduction_pct": peak_memory_summary.get("reduction_pct") if peak_memory_summary is not None else None,
        },
    }


def _fit_poly(sweep: list[dict[str, Any]]) -> list[float] | None:
    import math

    if len(sweep) < 3:
        return None
    valid = [result for result in sweep if not math.isnan(result.get("noise_percent", float("nan")))]
    if len(valid) < 3:
        return None
    noise_vals = [result["noise_percent"] for result in valid]
    acc_vals = [result.get("accuracy", 1.0 / (1.0 + result["noise_percent"] / 100.0)) for result in valid]
    return fit_biquadratic_polynomial(noise_vals, acc_vals)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(make_json_serializable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _build_rank_adaptation_summary(
    *,
    compressible: list[dict[str, Any]],
    layer_results: list[dict[str, Any]],
    distributions: list[dict[str, Any]],
    compression_log: dict[str, Any],
    vit_attr: str,
) -> dict[str, Any]:
    return {
        "method": "svd_low_rank",
        "vit_attr": vit_attr,
        "compressible_layers": [
            {
                "name": layer["name"],
                "in_features": int(layer["in_features"]),
                "out_features": int(layer["out_features"]),
            }
            for layer in compressible
        ],
        "phase1": [
            {
                "name": layer["name"],
                "sweep_points": len(layer.get("sweep_results", [])),
                "poly_coeffs": layer.get("poly_coeffs"),
                "sampled_ranks": [point.get("rank") for point in layer.get("sweep_results", [])],
            }
            for layer in layer_results
        ],
        "budget_distribution": distributions,
        "compression_log": compression_log,
    }


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
            name="rank_adapted_checkpoint",
            type=ArtifactType.MODEL,
            path=str(output_checkpoint_path),
            format=_artifact_format(output_checkpoint_path),
            producer=producer,
            metadata={
                "reduction_pct": stats.get("reduction_pct"),
                "linear_reduction_pct": stats.get("linear_reduction_pct"),
                "params_before": stats.get("params_before"),
                "params_after": stats.get("params_after"),
            },
        ),
        ArtifactDescriptor(
            name="rank_adapted_config",
            type=ArtifactType.CONFIG,
            path=str(output_config_path),
            format=_artifact_format(output_config_path),
            producer=producer,
        ),
        ArtifactDescriptor(
            name="rank_adaptation_stats",
            type=ArtifactType.METRICS,
            path=str(output_stats_path),
            format=_artifact_format(output_stats_path),
            producer=producer,
        ),
        ArtifactDescriptor(
            name="rank_adaptation_summary",
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


def _build_memory_improvement(benchmark_summary: dict[str, Any] | None, key: str) -> MemoryImprovement | None:
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


def _build_rank_adaptation_metrics(stats: dict[str, Any]) -> RankAdaptationMetrics:
    benchmark_summary = stats.get("benchmark")
    return RankAdaptationMetrics(
        params_before=int(stats.get("params_before", 0)),
        params_after=int(stats.get("params_after", 0)),
        parameter_reduction_pct=float(stats.get("reduction_pct", 0.0)),
        linear_params_before=int(stats.get("linear_params_before", 0)),
        linear_params_after=int(stats.get("linear_params_after", 0)),
        linear_reduction_pct=float(stats.get("linear_reduction_pct", 0.0)),
        checkpoint_size_reduction_pct=stats.get("checkpoint_size_reduction_pct"),
        latency=_build_latency_improvement(benchmark_summary),
        parameter_memory=_build_memory_improvement(benchmark_summary, "parameter_memory"),
        peak_memory=_build_memory_improvement(benchmark_summary, "peak_memory"),
    )


def _build_component_result(
    *,
    output_artifacts: list[ArtifactDescriptor],
    stats: dict[str, Any],
    producer: str,
    output_dir: Path,
    benchmark_enabled: bool,
) -> RankAdaptationComponentResult:
    metrics = _build_rank_adaptation_metrics(stats)
    return RankAdaptationComponentResult(
        component_name=producer,
        status=ComponentStatus.SUCCESS,
        message=f"Rank adaptation completed successfully in {output_dir}",
        output_artifacts=output_artifacts,
        metrics=asdict(metrics),
        metadata={
            "interface_version": INTERFACE_VERSION,
            "output_dir": str(output_dir),
            "benchmark_enabled": benchmark_enabled,
            "checkpoint_path": stats.get("checkpoint_path"),
            "config_path": stats.get("config_path"),
            "options": stats.get("options", {}),
            "method": stats.get("method", "svd_low_rank"),
        },
        rank_adaptation_metrics=metrics,
    )


def _build_rank_adaptation_options(params: dict[str, Any]) -> OrbisRankAdaptationOptions:
    return OrbisRankAdaptationOptions(
        acc_budget_pct=float(params.get("acc_budget_pct", 2.0)),
        comp_target=float(params.get("comp_target", 2.0)),
        rank_step_fraction=float(params.get("rank_step_fraction", 0.20)),
        layer_patterns=list(params.get("layer_patterns", OrbisRankAdaptationOptions().layer_patterns)),
        skip_patterns=list(params.get("skip_patterns", OrbisRankAdaptationOptions().skip_patterns)),
        min_features=int(params.get("min_features", 64)),
        batch_size=int(params.get("batch_size", 4)),
        run_benchmark=bool(params.get("run_benchmark", True)),
    )


class OrbisRankAdaptationComponent(ComponentInterface):
    def validate_params(self) -> None:
        output_dir = self.params.get("output_dir")
        checkpoint_path = self.params.get("checkpoint_path")
        if output_dir is None and checkpoint_path is None:
            return

        if output_dir is not None and not str(output_dir).strip():
            raise ValueError("output_dir must not be empty")
        if checkpoint_path is not None and not str(checkpoint_path).strip():
            raise ValueError("checkpoint_path must not be empty")

        acc_budget_pct = float(self.params.get("acc_budget_pct", 2.0))
        if not (0.0 < acc_budget_pct < 100.0):
            raise ValueError("acc_budget_pct must be between 0 and 100")

        comp_target = float(self.params.get("comp_target", 2.0))
        if comp_target <= 1.0:
            raise ValueError("comp_target must be greater than 1.0")

        rank_step_fraction = float(self.params.get("rank_step_fraction", 0.20))
        if not (0.0 < rank_step_fraction < 1.0):
            raise ValueError("rank_step_fraction must be between 0 and 1")

        min_features = int(self.params.get("min_features", 64))
        if min_features < 1:
            raise ValueError("min_features must be positive")

    def validate_inputs(self, state: PipelineState) -> None:
        checkpoint_path = self.params.get("checkpoint_path")
        if checkpoint_path is None:
            latest_model = state.get_latest_artifact(ArtifactType.MODEL)
            if latest_model is None:
                raise ValueError("A MODEL artifact or checkpoint_path parameter is required")
            checkpoint_path = latest_model.path

        if not Path(checkpoint_path).expanduser().exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        config_path = self.params.get("config_path")
        if config_path is not None and not Path(config_path).expanduser().exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

    def run(self, state: PipelineState) -> RankAdaptationComponentResult:
        self.validate_params()
        self.validate_inputs(state)

        checkpoint_artifact = state.get_latest_artifact(ArtifactType.MODEL)
        config_artifact = state.get_latest_artifact(ArtifactType.CONFIG)

        checkpoint_path = self.params.get("checkpoint_path") or (checkpoint_artifact.path if checkpoint_artifact else None)
        config_path = self.params.get("config_path") or (config_artifact.path if config_artifact else None)
        output_dir = self.params.get("output_dir")
        orbis_repo_path = self.params.get("orbis_repo_path") or state.global_metadata.get("orbis_repo_path")

        result = rank_adapt_orbis_checkpoint(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            options=_build_rank_adaptation_options(self.params),
            config_path=config_path,
            orbis_repo_path=orbis_repo_path,
            vit_attr=str(self.params.get("vit_attr", "vit")),
            checkpoint_dir=self.params.get("checkpoint_dir"),
            skip_phase1=bool(self.params.get("skip_phase1", False)),
        )
        state.add_result(result.component_result)
        return result.component_result

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "interface_version": INTERFACE_VERSION,
            "component": "rank_adaptation",
            "method": "svd_low_rank",
            "produces": [artifact_type.value for artifact_type in (ArtifactType.MODEL, ArtifactType.CONFIG, ArtifactType.METRICS, ArtifactType.REPORT)],
        }


def rank_adapt_orbis_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    options: OrbisRankAdaptationOptions | None = None,
    config_path: str | Path | None = None,
    orbis_repo_path: str | Path | None = None,
    vit_attr: str = "vit",
    checkpoint_dir: str | Path | None = None,
    skip_phase1: bool = False,
) -> OrbisRankAdaptationResult:
    checkpoint_path = _absolute_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    source_checkpoint_path = checkpoint_path.absolute()
    resolved_checkpoint_path = checkpoint_path.resolve()

    resolved_config_path = _detect_config_path(checkpoint_path, config_path)
    resolved_config_path = resolved_config_path.absolute()
    rank_adaptation_options = options or OrbisRankAdaptationOptions()
    modules = resolve_orbis_modules(
        orbis_repo_path=orbis_repo_path,
        checkpoint_path=source_checkpoint_path,
    )
    model, model_config = _load_orbis_model(
        source_checkpoint_path,
        resolved_config_path,
        modules,
        orbis_repo_path=orbis_repo_path,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    params_before = count_all_params(model)
    vit_params_before = count_vit_params(model, vit_attr)
    linear_params_before = count_linear_params(model)
    checkpoint_size_before_mb = source_checkpoint_path.stat().st_size / (1024 * 1024)

    benchmark_results = None
    if rank_adaptation_options.run_benchmark:
        benchmark_results = {"before": _benchmark_vit(model, model_config)}
        model = model.to(device)

    dtype = next(model.parameters()).dtype if any(True for _ in model.parameters()) else torch.float32
    inputs = build_synthetic_vit_inputs(model_config, device, dtype)
    compressible = get_compressible_linear_layers(
        model,
        vit_attr=vit_attr,
        layer_patterns=rank_adaptation_options.layer_patterns if rank_adaptation_options.layer_patterns else None,
        skip_patterns=rank_adaptation_options.skip_patterns if rank_adaptation_options.skip_patterns else None,
        min_features=rank_adaptation_options.min_features,
    )

    output_dir = _resolve_output_dir(
        output_dir,
        checkpoint_path=source_checkpoint_path,
        config_path=resolved_config_path,
        orbis_repo_path=orbis_repo_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    onnx_dir = output_dir / "onnx"
    rank_adaptation_dir = output_dir / "rank_adaptation"
    phase1_dir = rank_adaptation_dir / "phase1_checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir.mkdir(parents=True, exist_ok=True)
    rank_adaptation_dir.mkdir(parents=True, exist_ok=True)
    phase1_dir.mkdir(parents=True, exist_ok=True)

    global_acc_budget = rank_adaptation_options.acc_budget_pct / 100.0
    layer_results: list[dict[str, Any]] = []

    if checkpoint_dir and Path(checkpoint_dir).expanduser().is_dir():
        layer_results, missing = load_layer_results_from_checkpoints(str(Path(checkpoint_dir).expanduser().resolve()), compressible)
        for layer_info in missing:
            sweep = sweep_linear_layer(
                model,
                layer_info,
                inputs,
                device,
                rank_step_fraction=rank_adaptation_options.rank_step_fraction,
                forward_fn=_orbis_forward,
            )
            poly_coeffs = _fit_poly(sweep)
            layer_results.append({
                **layer_info,
                "sweep_results": sweep,
                "poly_coeffs": poly_coeffs,
            })
    elif not skip_phase1 and compressible:
        for index, layer_info in enumerate(compressible, start=1):
            sweep = sweep_linear_layer(
                model,
                layer_info,
                inputs,
                device,
                rank_step_fraction=rank_adaptation_options.rank_step_fraction,
                forward_fn=_orbis_forward,
            )
            poly_coeffs = _fit_poly(sweep)
            if poly_coeffs is not None:
                checkpoint_json_path = phase1_dir / f"svd_ckpt_layer{index}of{len(compressible)}.json"
                save_layer_checkpoint(
                    str(checkpoint_json_path),
                    "orbis_vit",
                    layer_info["name"],
                    sweep,
                    poly_coeffs,
                )
            layer_results.append({
                **layer_info,
                "sweep_results": sweep,
                "poly_coeffs": poly_coeffs,
            })

    if layer_results:
        distributions = distribute_budget(layer_results, global_acc_budget, rank_adaptation_options.comp_target)
        baseline_quality = evaluate_accuracy_vit(model, inputs, forward_fn=_orbis_forward)
        compressed_model, compression_log = sequential_compress_linear(
            model,
            layer_results,
            distributions,
            baseline_quality,
            global_acc_budget,
            inputs,
            device,
            forward_fn=_orbis_forward,
        )
    else:
        compressed_model = model
        compression_log = {}
        distributions = []

    params_after = count_all_params(compressed_model)
    vit_params_after = count_vit_params(compressed_model, vit_attr)
    linear_params_after = count_linear_params(compressed_model)
    reduction_pct = (1 - params_after / params_before) * 100.0 if params_before > 0 else 0.0
    linear_reduction_pct = (1 - linear_params_after / linear_params_before) * 100.0 if linear_params_before > 0 else 0.0

    if benchmark_results is not None:
        benchmark_results = _build_benchmark_summary(benchmark_results["before"], _benchmark_vit(compressed_model, model_config))

    output_checkpoint_path = checkpoints_dir / "last.ckpt"
    output_config_path = output_dir / "config.yaml"
    output_stats_path = rank_adaptation_dir / "rank_adaptation_stats.json"
    output_summary_path = rank_adaptation_dir / "rank_adaptation_summary.json"
    output_benchmark_path = rank_adaptation_dir / "benchmark_stats.json"

    compressed_model = compressed_model.to("cpu")
    low_rank_metadata = collect_low_rank_metadata(compressed_model)
    torch.save(
        {
            "state_dict": compressed_model.state_dict(),
            "rank_adaptation_stats": {
                "method": "svd_low_rank",
                "source_checkpoint_path": str(source_checkpoint_path),
                "resolved_checkpoint_path": str(resolved_checkpoint_path),
                "params_before": params_before,
                "params_after": params_after,
                "reduction_pct": reduction_pct,
                "linear_params_before": linear_params_before,
                "linear_params_after": linear_params_after,
                "linear_reduction_pct": linear_reduction_pct,
                "low_rank_modules": low_rank_metadata,
            },
            "rank_adaptation_metadata": low_rank_metadata,
        },
        output_checkpoint_path,
    )
    OmegaConf.save(model_config, output_config_path)

    checkpoint_size_after_mb = output_checkpoint_path.stat().st_size / (1024 * 1024)
    stats = {
        "method": "svd_low_rank",
        "checkpoint_path": str(source_checkpoint_path),
        "source_checkpoint_path": str(source_checkpoint_path),
        "resolved_checkpoint_path": str(resolved_checkpoint_path),
        "output_checkpoint_path": str(output_checkpoint_path),
        "config_path": str(resolved_config_path),
        "params_before": int(params_before),
        "params_after": int(params_after),
        "reduction_pct": float(reduction_pct),
        "vit_params_before": int(vit_params_before),
        "vit_params_after": int(vit_params_after),
        "linear_params_before": int(linear_params_before),
        "linear_params_after": int(linear_params_after),
        "linear_reduction_pct": float(linear_reduction_pct),
        "low_rank_modules": low_rank_metadata,
        "checkpoint_size_mb_before": float(checkpoint_size_before_mb),
        "checkpoint_size_mb_after": float(checkpoint_size_after_mb),
        "checkpoint_size_reduction_pct": float((1 - checkpoint_size_after_mb / checkpoint_size_before_mb) * 100.0) if checkpoint_size_before_mb > 0 else 0.0,
        "options": asdict(rank_adaptation_options),
        "compression_log": compression_log,
        "distributions": distributions,
        "layers_compressed": sum(1 for value in compression_log.values() if value.get("status") == "compressed"),
        "layers_skipped": sum(1 for value in compression_log.values() if value.get("status", "").startswith("skipped")),
    }
    if benchmark_results is not None:
        stats["benchmark"] = benchmark_results.get("summary", {})

    summary = _build_rank_adaptation_summary(
        compressible=compressible,
        layer_results=layer_results,
        distributions=distributions,
        compression_log=compression_log,
        vit_attr=vit_attr,
    )

    _write_json(output_stats_path, stats)
    _write_json(output_summary_path, summary)
    if benchmark_results is not None:
        _write_json(output_benchmark_path, benchmark_results)

    output_artifacts = _build_output_artifacts(
        output_checkpoint_path=output_checkpoint_path,
        output_config_path=output_config_path,
        output_stats_path=output_stats_path,
        output_summary_path=output_summary_path,
        output_benchmark_path=output_benchmark_path if benchmark_results is not None else None,
        producer="rank_adaptation",
        stats=stats,
    )
    component_result = _build_component_result(
        output_artifacts=output_artifacts,
        stats=stats,
        producer="rank_adaptation",
        output_dir=output_dir,
        benchmark_enabled=benchmark_results is not None,
    )

    return OrbisRankAdaptationResult(
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