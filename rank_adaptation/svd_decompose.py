# SPDX-License-Identifier: MIT
# Author: Arunachalam Thirunavukkarasu
# Contributor: Dr Shashank Pathak
# Email: arunachalam.thirunavukkarasu@dlr.de
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    def __init__(
        self,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        orig_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        rank = down_weight.shape[0]
        in_features = down_weight.shape[1]
        out_features = up_weight.shape[0]

        self.down_proj = nn.Linear(in_features, rank, bias=False)
        self.up_proj = nn.Linear(rank, out_features, bias=(orig_bias is not None))

        with torch.no_grad():
            self.down_proj.weight.copy_(down_weight)
            self.up_proj.weight.copy_(up_weight)
            if orig_bias is not None and self.up_proj.bias is not None:
                self.up_proj.bias.copy_(orig_bias)

        self._rank = rank
        self._in_features = in_features
        self._out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up_proj(self.down_proj(x))

    def extra_repr(self) -> str:
        return f"in={self._in_features}, rank={self._rank}, out={self._out_features}"


def make_low_rank_linear(
    *,
    in_features: int,
    out_features: int,
    rank: int,
    has_bias: bool,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> LowRankLinear:
    if device is None:
        device = "cpu"
    if dtype is None:
        dtype = torch.float32

    down_weight = torch.zeros(rank, in_features, device=device, dtype=dtype)
    up_weight = torch.zeros(out_features, rank, device=device, dtype=dtype)
    orig_bias = torch.zeros(out_features, device=device, dtype=dtype) if has_bias else None
    return LowRankLinear(down_weight=down_weight, up_weight=up_weight, orig_bias=orig_bias)


def svd_decompose_linear(linear_layer: nn.Linear, rank: int) -> LowRankLinear:
    weight = linear_layer.weight.data.float()
    out_features, in_features = weight.shape

    max_rank = min(in_features, out_features)
    rank = min(rank, max_rank - 1)
    rank = max(rank, 1)

    u_matrix, singular_values, vh_matrix = torch.linalg.svd(weight, full_matrices=False)
    u_reduced = u_matrix[:, :rank]
    singular_values_reduced = singular_values[:rank]
    vh_reduced = vh_matrix[:rank, :]

    up_weight = u_reduced * singular_values_reduced.unsqueeze(0)
    down_weight = vh_reduced

    orig_dtype = linear_layer.weight.dtype
    up_weight = up_weight.to(orig_dtype)
    down_weight = down_weight.to(orig_dtype)

    orig_bias = linear_layer.bias.data.clone() if linear_layer.bias is not None else None
    return LowRankLinear(down_weight=down_weight, up_weight=up_weight, orig_bias=orig_bias)


def compute_compression_ratio_linear(in_features: int, out_features: int, rank: int) -> float:
    original = in_features * out_features
    decomposed = rank * (in_features + out_features)
    return original / decomposed if decomposed > 0 else 0.0


def _find_parent_and_attr(model: nn.Module, target_name: str):
    parts = target_name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def replace_linear_module(model: nn.Module, target_name: str, replacement: nn.Module) -> nn.Module:
    new_model = copy.deepcopy(model)
    parent, attr = _find_parent_and_attr(new_model, target_name)

    if attr.isdigit():
        parent[int(attr)] = replacement
    else:
        setattr(parent, attr, replacement)

    return new_model