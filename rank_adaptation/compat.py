# SPDX-License-Identifier: MIT
# Author: Arunachalam Thirunavukkarasu
# Contributor: Dr Shashank Pathak
# Email: arunachalam.thirunavukkarasu@dlr.de
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .svd_decompose import LowRankLinear, make_low_rank_linear


def collect_low_rank_metadata(model: nn.Module) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for module_path, module in model.named_modules():
        if not isinstance(module, LowRankLinear):
            continue
        metadata.append(
            {
                "path": module_path,
                "rank": int(module.down_proj.out_features),
                "in_features": int(module.down_proj.in_features),
                "out_features": int(module.up_proj.out_features),
                "has_bias": bool(module.up_proj.bias is not None),
            }
        )
    return metadata


def extract_low_rank_metadata_from_checkpoint(raw_checkpoint: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_checkpoint, dict):
        return []

    metadata = raw_checkpoint.get("rank_adaptation_metadata")
    if isinstance(metadata, list):
        return metadata

    stats = raw_checkpoint.get("rank_adaptation_stats")
    if isinstance(stats, dict):
        fallback = stats.get("low_rank_modules")
        if isinstance(fallback, list):
            return fallback

    return []


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


def apply_low_rank_metadata(model: nn.Module, metadata: list[dict[str, Any]]) -> nn.Module:
    for item in metadata:
        module_path = str(item["path"])
        replacement = make_low_rank_linear(
            in_features=int(item["in_features"]),
            out_features=int(item["out_features"]),
            rank=int(item["rank"]),
            has_bias=bool(item.get("has_bias", True)),
        )
        _set_child_module(model, module_path, replacement)
    return model