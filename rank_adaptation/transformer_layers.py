# SPDX-License-Identifier: MIT
# Author: Arunachalam Thirunavukkarasu
# Contributor: Dr Shashank Pathak
# Email: arunachalam.thirunavukkarasu@dlr.de
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

from __future__ import annotations

from typing import Any

import torch.nn as nn


def get_compressible_linear_layers(
    model: nn.Module,
    *,
    vit_attr: str = "vit",
    layer_patterns: list[str] | None = None,
    skip_patterns: list[str] | None = None,
    min_features: int = 64,
) -> list[dict[str, Any]]:
    search_root = model
    prefix = ""
    if vit_attr:
        if hasattr(model, vit_attr):
            search_root = getattr(model, vit_attr)
            prefix = f"{vit_attr}."
        else:
            print(f"[Layers] Warning: model has no '{vit_attr}' attribute, searching entire model")

    if skip_patterns is None:
        skip_patterns = ["head", "cls_token", "pos_embed", "patch_embed", "norm"]

    result: list[dict[str, Any]] = []
    for name, module in search_root.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        full_name = f"{prefix}{name}" if prefix else name
        in_features = module.in_features
        out_features = module.out_features
        if min(in_features, out_features) < min_features:
            continue
        if any(pattern in full_name for pattern in skip_patterns):
            continue
        if layer_patterns and not any(pattern in full_name for pattern in layer_patterns):
            continue

        result.append({
            "name": full_name,
            "in_features": in_features,
            "out_features": out_features,
            "module": module,
        })

    print(f"[Layers] Found {len(result)} compressible Linear layers in {'model.' + vit_attr if vit_attr else 'model'}")
    for index, layer in enumerate(result, start=1):
        print(f"  [{index:2d}] {layer['name']:<50} in={layer['in_features']:>5}  out={layer['out_features']:>5}")
    return result


def count_linear_params(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            total += sum(parameter.numel() for parameter in module.parameters())
    return total


def count_all_params(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def count_vit_params(model: nn.Module, vit_attr: str = "vit") -> int:
    if hasattr(model, vit_attr):
        vit = getattr(model, vit_attr)
        return sum(parameter.numel() for parameter in vit.parameters())
    return count_all_params(model)