from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from timm.layers.mlp import SwiGLU
from timm.models.vision_transformer import Mlp

from .api import _absolute_path, _detect_config_path, _load_checkpoint, _load_orbis_model
from .bootstrap import resolve_orbis_modules


METADATA_SCHEMA_NAME = "orbis_pruned_model_metadata"
METADATA_SCHEMA_VERSION = "1.0"


def _default_orbis_repo_path() -> Path | None:
    repo_root = Path(__file__).resolve().parents[1]
    if (repo_root / "util.py").exists():
        return repo_root

    workspace_checkout = repo_root / "external" / "orbis"
    if workspace_checkout.exists():
        return workspace_checkout
    return None


def _generator_signature(config: Any) -> dict[str, Any]:
    generator_config = config.model.params.get("generator_config")
    if generator_config is None:
        return {}

    params = generator_config.get("params", {})
    return {
        "target": generator_config.get("target"),
        "hidden_size": params.get("hidden_size"),
        "input_size": list(params.get("input_size", [])) if params.get("input_size") is not None else None,
        "in_channels": params.get("in_channels"),
        "num_heads": params.get("num_heads"),
        "depth": params.get("depth"),
        "mlp_ratio": params.get("mlp_ratio"),
        "max_num_frames": params.get("max_num_frames"),
    }


def _top_level_summary(raw_checkpoint: Any) -> dict[str, Any]:
    if not isinstance(raw_checkpoint, dict):
        return {
            "container_type": type(raw_checkpoint).__name__,
            "top_level_keys": [],
            "has_state_dict_key": False,
            "has_pruning_stats_key": False,
        }

    return {
        "container_type": type(raw_checkpoint).__name__,
        "top_level_keys": sorted(str(key) for key in raw_checkpoint.keys()),
        "has_state_dict_key": "state_dict" in raw_checkpoint,
        "has_pruning_stats_key": "pruning_stats" in raw_checkpoint,
    }


def _state_dict_view(raw_checkpoint: Any) -> dict[str, Any]:
    if isinstance(raw_checkpoint, dict) and "state_dict" in raw_checkpoint:
        state_dict = raw_checkpoint["state_dict"]
    else:
        state_dict = raw_checkpoint

    if not hasattr(state_dict, "items"):
        raise TypeError("Checkpoint does not contain a state_dict-like mapping")

    tensor_count = 0
    parameter_count = 0
    sample_shapes: dict[str, list[int]] = {}
    for name, tensor in state_dict.items():
        tensor_count += 1
        if hasattr(tensor, "numel"):
            parameter_count += int(tensor.numel())
        if len(sample_shapes) < 12 and hasattr(tensor, "shape"):
            sample_shapes[str(name)] = [int(dim) for dim in tensor.shape]

    return {
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "sample_tensor_shapes": sample_shapes,
    }


def _module_shape_summary(model: Any) -> list[dict[str, Any]]:
    vit = model.vit if hasattr(model, "vit") else model
    summary: list[dict[str, Any]] = []
    for module_path, module in vit.named_modules():
        if isinstance(module, Mlp):
            summary.append(
                {
                    "name": module_path,
                    "kind": "mlp",
                    "layer_group": "space" if "space_mlp" in module_path else "time" if "time_mlp" in module_path else "all",
                    "in_features": int(module.fc1.in_features),
                    "hidden_features": int(module.fc1.out_features),
                    "out_features": int(module.fc2.out_features),
                    "effective_hidden_dim": int(module.fc1.out_features),
                }
            )
        elif isinstance(module, SwiGLU):
            summary.append(
                {
                    "name": module_path,
                    "kind": "swiglu",
                    "layer_group": "space" if "space_mlp" in module_path else "time" if "time_mlp" in module_path else "all",
                    "in_features": int(module.fc1_x.in_features),
                    "hidden_features": int(module.fc1_x.out_features),
                    "out_features": int(module.fc2.out_features),
                    "effective_hidden_dim": int((module.fc1_x.out_features * 3) // 2),
                }
            )
    return summary


def extract_pruned_metadata(
    checkpoint_path: str | Path,
    *,
    config_path: str | Path | None = None,
    orbis_repo_path: str | Path | None = None,
) -> dict[str, Any]:
    checkpoint_path = _absolute_path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    resolved_config_path = _detect_config_path(checkpoint_path, config_path)
    raw_checkpoint = _load_checkpoint(checkpoint_path)
    resolved_orbis_repo_path = orbis_repo_path or _default_orbis_repo_path()
    modules = resolve_orbis_modules(
        orbis_repo_path=resolved_orbis_repo_path,
        checkpoint_path=checkpoint_path,
    )
    model, config = _load_orbis_model(
        checkpoint_path,
        resolved_config_path,
        modules,
        orbis_repo_path=resolved_orbis_repo_path,
    )

    structure_summary = modules.get_pruning_summary(model)
    module_shapes = _module_shape_summary(model)
    embedded_stats = raw_checkpoint.get("pruning_stats") if isinstance(raw_checkpoint, dict) else None

    metadata = {
        "schema_name": METADATA_SCHEMA_NAME,
        "schema_version": METADATA_SCHEMA_VERSION,
        "source": {
            "checkpoint_path": str(checkpoint_path),
            "config_path": str(resolved_config_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "orbis_repo_path": str(resolved_orbis_repo_path) if resolved_orbis_repo_path is not None else None,
        },
        "checkpoint": {
            **_top_level_summary(raw_checkpoint),
            **_state_dict_view(raw_checkpoint),
        },
        "model": {
            "target": config.model.get("target"),
            "generator": _generator_signature(config),
            "has_vit": hasattr(model, "vit"),
        },
        "structure": {
            "mlp_module_count": int(structure_summary.get("mlp_module_count", len(module_shapes))),
            "mlp_modules": module_shapes,
        },
        "pruning": {
            "summary": structure_summary,
            "embedded_stats": embedded_stats,
            "options": embedded_stats.get("options") if isinstance(embedded_stats, dict) else None,
        },
    }

    return metadata


def write_pruned_metadata(
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    config_path: str | Path | None = None,
    orbis_repo_path: str | Path | None = None,
) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    metadata = extract_pruned_metadata(
        checkpoint_path,
        config_path=config_path,
        orbis_repo_path=orbis_repo_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(OmegaConf.to_yaml(metadata, resolve=True), encoding="utf-8")
    return output_path