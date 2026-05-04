# SPDX-License-Identifier: MIT
# Author: Dr Shashank Pathak
# Email: shashank@computer.org
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from timm.layers.mlp import SwiGLU
from timm.models.vision_transformer import Mlp


@dataclass(frozen=True)
class StructuredPruningConfig:
    enabled: bool = True
    mlp_prune_ratio: float = 0.2
    mlp_round_to: int = 128
    head_prune_ratio: float = 0.0
    mlp_prune_layers: str = "all"
    head_prune_layers: str = "all"
    importance_metric: str = "l1_weight"
    recovery_steps: int = 0
    recovery_lr_multiplier: float = 0.1


@dataclass(frozen=True)
class _PrunePlanItem:
    module_path: str
    module_kind: str
    layer_group: str
    keep_indices: tuple[int, ...]
    original_hidden_features: int
    target_hidden_features: int
    original_effective_dim: int
    target_effective_dim: int


def _act_layer_factory(module: nn.Module):
    if isinstance(module, nn.GELU):
        approximate = getattr(module, "approximate", "none")
        return lambda: nn.GELU(approximate=approximate)
    return type(module)


def _norm_layer_factory(module: nn.Module):
    if isinstance(module, nn.Identity):
        return None
    return type(module)


def _round_target_dim(original_dim: int, prune_ratio: float, round_to: int) -> int:
    remaining = max(1, int(round(original_dim * (1.0 - prune_ratio))))
    if round_to > 1:
        rounded = int(round(remaining / round_to)) * round_to
        remaining = max(round_to, rounded)
    return min(original_dim, max(1, remaining))


def _resolve_module(root: nn.Module, module_path: str) -> nn.Module:
    current: Any = root
    for part in module_path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def _resolve_parent(root: nn.Module, module_path: str) -> tuple[Any, str]:
    parts = module_path.split(".")
    current: Any = root
    for part in parts[:-1]:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current, parts[-1]


def _set_child_module(root: nn.Module, module_path: str, module: nn.Module) -> None:
    parent, child_name = _resolve_parent(root, module_path)
    if child_name.isdigit():
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def _infer_layer_group(module_path: str) -> str:
    if "space_mlp" in module_path:
        return "space"
    if "time_mlp" in module_path:
        return "time"
    return "all"


def _layer_selected(layer_group: str, requested_layers: str) -> bool:
    if requested_layers == "all":
        return True
    return layer_group == requested_layers


def _effective_hidden_dim(module: nn.Module) -> int:
    if isinstance(module, Mlp):
        return module.fc1.out_features
    if isinstance(module, SwiGLU):
        return (module.fc1_x.out_features * 3) // 2
    raise TypeError(f"Unsupported MLP module type: {type(module)!r}")


def _hidden_features(module: nn.Module) -> int:
    if isinstance(module, Mlp):
        return module.fc1.out_features
    if isinstance(module, SwiGLU):
        return module.fc1_x.out_features
    raise TypeError(f"Unsupported MLP module type: {type(module)!r}")


def _target_hidden_features(module: nn.Module, target_effective_dim: int) -> int:
    if isinstance(module, Mlp):
        return target_effective_dim
    if isinstance(module, SwiGLU):
        return max(1, (target_effective_dim * 2) // 3)
    raise TypeError(f"Unsupported MLP module type: {type(module)!r}")


def _importance_scores(module: nn.Module, metric: str) -> torch.Tensor:
    if metric == "random":
        return torch.rand(_hidden_features(module), dtype=torch.float32)

    power = 1 if metric == "l1_weight" else 2

    if isinstance(module, Mlp):
        fc1 = module.fc1.weight.detach().float()
        fc2 = module.fc2.weight.detach().float()
        if power == 1:
            return fc1.abs().sum(dim=1) + fc2.abs().sum(dim=0)
        return fc1.pow(2).sum(dim=1) + fc2.pow(2).sum(dim=0)

    if isinstance(module, SwiGLU):
        gate = module.fc1_g.weight.detach().float()
        value = module.fc1_x.weight.detach().float()
        proj = module.fc2.weight.detach().float()
        if power == 1:
            return gate.abs().sum(dim=1) + value.abs().sum(dim=1) + proj.abs().sum(dim=0)
        return gate.pow(2).sum(dim=1) + value.pow(2).sum(dim=1) + proj.pow(2).sum(dim=0)

    raise TypeError(f"Unsupported MLP module type: {type(module)!r}")


def _build_pruned_mlp(module: Mlp, keep_indices: tuple[int, ...]) -> Mlp:
    act_layer = _act_layer_factory(module.act)
    norm_layer = _norm_layer_factory(module.norm)
    pruned = Mlp(
        in_features=module.fc1.in_features,
        hidden_features=len(keep_indices),
        out_features=module.fc2.out_features,
        act_layer=act_layer,
        norm_layer=norm_layer,
        bias=(module.fc1.bias is not None, module.fc2.bias is not None),
        drop=(module.drop1.p, module.drop2.p),
        use_conv=False,
        device=module.fc1.weight.device,
        dtype=module.fc1.weight.dtype,
    )
    index_tensor = torch.tensor(keep_indices, device=module.fc1.weight.device, dtype=torch.long)
    pruned.fc1.weight.data.copy_(module.fc1.weight.data.index_select(0, index_tensor))
    if module.fc1.bias is not None:
        pruned.fc1.bias.data.copy_(module.fc1.bias.data.index_select(0, index_tensor))
    if not isinstance(module.norm, nn.Identity):
        pruned.norm.weight.data.copy_(module.norm.weight.data.index_select(0, index_tensor))
        pruned.norm.bias.data.copy_(module.norm.bias.data.index_select(0, index_tensor))
    pruned.fc2.weight.data.copy_(module.fc2.weight.data.index_select(1, index_tensor))
    if module.fc2.bias is not None:
        pruned.fc2.bias.data.copy_(module.fc2.bias.data)
    return pruned


def _build_pruned_swiglu(module: SwiGLU, keep_indices: tuple[int, ...]) -> SwiGLU:
    act_layer = _act_layer_factory(module.act)
    norm_layer = _norm_layer_factory(module.norm)
    pruned = SwiGLU(
        in_features=module.fc1_x.in_features,
        hidden_features=len(keep_indices),
        out_features=module.fc2.out_features,
        act_layer=act_layer,
        norm_layer=norm_layer,
        bias=(
            module.fc1_g.bias is not None,
            module.fc2.bias is not None,
        ),
        drop=(module.drop1.p, module.drop2.p),
        align_to=0,
        device=module.fc1_x.weight.device,
        dtype=module.fc1_x.weight.dtype,
    )
    index_tensor = torch.tensor(keep_indices, device=module.fc1_x.weight.device, dtype=torch.long)
    pruned.fc1_g.weight.data.copy_(module.fc1_g.weight.data.index_select(0, index_tensor))
    pruned.fc1_x.weight.data.copy_(module.fc1_x.weight.data.index_select(0, index_tensor))
    if module.fc1_g.bias is not None:
        pruned.fc1_g.bias.data.copy_(module.fc1_g.bias.data.index_select(0, index_tensor))
    if module.fc1_x.bias is not None:
        pruned.fc1_x.bias.data.copy_(module.fc1_x.bias.data.index_select(0, index_tensor))
    if not isinstance(module.norm, nn.Identity):
        pruned.norm.weight.data.copy_(module.norm.weight.data.index_select(0, index_tensor))
        pruned.norm.bias.data.copy_(module.norm.bias.data.index_select(0, index_tensor))
    pruned.fc2.weight.data.copy_(module.fc2.weight.data.index_select(1, index_tensor))
    if module.fc2.bias is not None:
        pruned.fc2.bias.data.copy_(module.fc2.bias.data)
    return pruned


def _build_pruned_module(module: nn.Module, keep_indices: tuple[int, ...]) -> nn.Module:
    if isinstance(module, Mlp):
        return _build_pruned_mlp(module, keep_indices)
    if isinstance(module, SwiGLU):
        return _build_pruned_swiglu(module, keep_indices)
    raise TypeError(f"Unsupported MLP module type: {type(module)!r}")


def _collect_prune_plan(vit: nn.Module, config: StructuredPruningConfig) -> list[_PrunePlanItem]:
    supported_metrics = {"l1_weight", "l2_weight", "random"}
    if config.importance_metric not in supported_metrics:
        raise ValueError(f"Unsupported importance metric: {config.importance_metric}")
    if config.mlp_prune_layers not in {"all", "space", "time"}:
        raise ValueError(f"Unsupported mlp_prune_layers value: {config.mlp_prune_layers}")

    plan: list[_PrunePlanItem] = []
    for module_path, module in vit.named_modules():
        if not isinstance(module, (Mlp, SwiGLU)):
            continue

        layer_group = _infer_layer_group(module_path)
        if not _layer_selected(layer_group, config.mlp_prune_layers):
            continue

        original_effective_dim = _effective_hidden_dim(module)
        target_effective_dim = _round_target_dim(
            original_effective_dim,
            config.mlp_prune_ratio,
            config.mlp_round_to,
        )
        original_hidden_features = _hidden_features(module)
        target_hidden_features = _target_hidden_features(module, target_effective_dim)

        if target_hidden_features >= original_hidden_features:
            keep_indices = tuple(range(original_hidden_features))
        else:
            scores = _importance_scores(module, config.importance_metric)
            kept = torch.topk(scores, k=target_hidden_features, largest=True, sorted=False).indices.tolist()
            keep_indices = tuple(sorted(int(index) for index in kept))

        module_kind = "mlp" if isinstance(module, Mlp) else "swiglu"
        plan.append(
            _PrunePlanItem(
                module_path=module_path,
                module_kind=module_kind,
                layer_group=layer_group,
                keep_indices=keep_indices,
                original_hidden_features=original_hidden_features,
                target_hidden_features=len(keep_indices),
                original_effective_dim=original_effective_dim,
                target_effective_dim=target_effective_dim if len(keep_indices) < original_hidden_features else original_effective_dim,
            )
        )

    return plan


def _apply_prune_plan(root: nn.Module, plan: list[_PrunePlanItem]) -> None:
    for item in plan:
        module = _resolve_module(root, item.module_path)
        pruned = _build_pruned_module(module, item.keep_indices)
        _set_child_module(root, item.module_path, pruned)


def _build_stats(plan: list[_PrunePlanItem], config: StructuredPruningConfig) -> dict[str, Any]:
    pruned_items = [item for item in plan if item.target_hidden_features < item.original_hidden_features]
    if not plan:
        return {
            "pruner": "local_mlp_pruner",
            "pruned_module_count": 0,
            "supported_head_pruning": False,
            "requested_head_prune_ratio": config.head_prune_ratio,
        }

    original_dims = sorted({item.original_effective_dim for item in plan})
    target_dims = sorted({item.target_effective_dim for item in plan})
    return {
        "pruner": "local_mlp_pruner",
        "pruned_module_count": len(pruned_items),
        "total_prunable_module_count": len(plan),
        "supported_head_pruning": False,
        "requested_head_prune_ratio": config.head_prune_ratio,
        "original_mlp_dim": plan[0].original_effective_dim,
        "target_mlp_dim": plan[0].target_effective_dim,
        "distinct_original_mlp_dims": original_dims,
        "distinct_target_mlp_dims": target_dims,
        "mlp_prune_layers": config.mlp_prune_layers,
        "mlp_modules": [
            {
                "name": item.module_path,
                "kind": item.module_kind,
                "layer_group": item.layer_group,
                "original_hidden_features": item.original_hidden_features,
                "target_hidden_features": item.target_hidden_features,
                "original_effective_dim": item.original_effective_dim,
                "target_effective_dim": item.target_effective_dim,
            }
            for item in plan
        ],
    }


def apply_structured_pruning(model: nn.Module, config: StructuredPruningConfig):
    if not config.enabled:
        return model, {"pruner": "local_mlp_pruner", "enabled": False}
    if config.head_prune_ratio > 0:
        raise NotImplementedError(
            "The local fallback pruner supports MLP pruning only. Set head_prune_ratio=0 or use an Orbis checkout "
            "that provides orbis.pruning.* modules."
        )
    if not hasattr(model, "vit"):
        raise AttributeError("Expected model to expose a vit module for pruning.")

    plan = _collect_prune_plan(model.vit, config)
    if not plan:
        raise ValueError("No prunable MLP modules were found in model.vit.")

    _apply_prune_plan(model.vit, plan)
    if hasattr(model, "ema_vit") and isinstance(model.ema_vit, nn.Module):
        _apply_prune_plan(model.ema_vit, plan)

    return model, _build_stats(plan, config)


def get_pruning_summary(model: nn.Module) -> dict[str, Any]:
    vit = model.vit if hasattr(model, "vit") else model
    mlp_modules: list[dict[str, Any]] = []
    for module_path, module in vit.named_modules():
        if not isinstance(module, (Mlp, SwiGLU)):
            continue
        mlp_modules.append(
            {
                "name": module_path,
                "kind": "mlp" if isinstance(module, Mlp) else "swiglu",
                "layer_group": _infer_layer_group(module_path),
                "hidden_features": _hidden_features(module),
                "effective_hidden_dim": _effective_hidden_dim(module),
            }
        )

    return {
        "pruner": "local_mlp_pruner",
        "mlp_module_count": len(mlp_modules),
        "mlp_modules": mlp_modules,
    }