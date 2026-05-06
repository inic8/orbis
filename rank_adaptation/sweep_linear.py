# SPDX-License-Identifier: MIT
# Author: Arunachalam Thirunavukkarasu
# Contributor: Dr Shashank Pathak
# Email: arunachalam.thirunavukkarasu@dlr.de
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

from __future__ import annotations

import gc
import time
import traceback
from typing import Any, Callable

import torch
import torch.nn as nn

from .noise_linear import compute_noise_percent_linear
from .svd_decompose import compute_compression_ratio_linear, replace_linear_module, svd_decompose_linear


def build_synthetic_vit_inputs(
    model_config: Any,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    try:
        gen_params = model_config.model.params.generator_config.params
        input_size = gen_params.input_size
        if input_size is None:
            height, width = 18, 32
        elif isinstance(input_size, int):
            height = width = input_size
        else:
            height = int(input_size[0])
            width = int(input_size[1])
        in_channels = int(gen_params.in_channels) if hasattr(gen_params, "in_channels") else 32
        max_num_frames = int(gen_params.max_num_frames) if hasattr(gen_params, "max_num_frames") else 6
        context_frames = max(1, min(max_num_frames - 1, 4))
    except Exception:
        height, width = 18, 32
        in_channels = 32
        context_frames = 4

    torch.manual_seed(42)
    return {
        "target": torch.randn(1, 1, in_channels, height, width, device=device, dtype=dtype),
        "context": torch.randn(1, context_frames, in_channels, height, width, device=device, dtype=dtype),
        "t": torch.rand(1, device=device, dtype=dtype),
        "frame_rate": torch.ones(1, device=device, dtype=dtype),
    }


def _orbis_forward(model, inputs, **kwargs):
    backbone = getattr(model, "vit", model)
    return backbone(inputs["target"], inputs["context"], inputs["t"], frame_rate=inputs["frame_rate"])


def evaluate_accuracy_vit(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    *,
    forward_fn: Callable | None = None,
) -> float:
    model.eval()
    try:
        with torch.no_grad():
            if forward_fn is not None:
                output = forward_fn(model, inputs)
            elif isinstance(inputs, dict) and "target" in inputs:
                output = model(inputs["target"], inputs["context"], inputs["t"], frame_rate=inputs["frame_rate"])
            else:
                output = model(inputs)

            if isinstance(output, tuple):
                output = output[0]
            return float(output.norm().item())
    except Exception as error:
        print(f"    [Eval] Forward pass failed: {error}")
        return 0.0


def sweep_linear_layer(
    orig_model: nn.Module,
    layer_info: dict[str, Any],
    inputs: Any,
    device: torch.device,
    *,
    rank_step_fraction: float = 0.20,
    forward_fn: Callable | None = None,
    forward_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    name = layer_info["name"]
    in_features = layer_info["in_features"]
    out_features = layer_info["out_features"]

    max_rank = min(in_features, out_features)
    step = max(1, int(round(rank_step_fraction * max_rank)))

    rank_values = list(range(step, max_rank, step))
    if rank_values and rank_values[-1] != max_rank - 1:
        rank_values.append(max_rank - 1)
    if rank_values and rank_values[0] > 1:
        rank_values.insert(0, max(1, step // 2))

    total = len(rank_values)
    results: list[dict[str, Any]] = []

    print(f"\n  Layer : {name}  in={in_features}  out={out_features}  max_rank={max_rank}  step={step}  sweeps={total}")

    for index, rank in enumerate(rank_values, start=1):
        started = time.time()
        tmp_model = None
        try:
            target = orig_model
            for part in name.split("."):
                target = target[int(part)] if part.isdigit() else getattr(target, part)

            low_rank = svd_decompose_linear(target, rank).to(device)
            tmp_model = replace_linear_module(orig_model, name, low_rank).to(device)
            tmp_model.eval()

            noise = compute_noise_percent_linear(
                orig_model,
                tmp_model,
                name,
                inputs,
                forward_fn=forward_fn,
                forward_kwargs=forward_kwargs,
            )
            compression_ratio = compute_compression_ratio_linear(in_features, out_features, rank)
            output_norm = evaluate_accuracy_vit(tmp_model, inputs, forward_fn=forward_fn)

            results.append(
                {
                    "rank": int(rank),
                    "noise_percent": float(noise),
                    "compression_ratio": float(compression_ratio),
                    "output_norm": float(output_norm),
                    "r_in": int(rank),
                    "r_out": int(rank),
                    "accuracy": float(1.0 / (1.0 + noise / 100.0)),
                }
            )

            elapsed = time.time() - started
            print(
                f"    [{index:3d}/{total}] rank={rank:4d} | noise={noise:6.2f}%  CR={compression_ratio:.2f}x  norm={output_norm:.4f}  ({elapsed:.1f}s)"
            )
        except Exception as error:
            print(f"    [{index:3d}/{total}] rank={rank}  FAILED: {error}")
            traceback.print_exc()
        finally:
            if tmp_model is not None:
                del tmp_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results