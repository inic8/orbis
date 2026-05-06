# SPDX-License-Identifier: MIT
# Author: Arunachalam Thirunavukkarasu
# Contributor: Dr Shashank Pathak
# Email: arunachalam.thirunavukkarasu@dlr.de
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _get_layer_output_hook(
    model: nn.Module,
    layer_name: str,
    x: Any,
    forward_fn: Any = None,
    forward_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    activation: dict[str, torch.Tensor] = {}

    def hook_fn(module, inp, output):
        if isinstance(output, tuple):
            activation["out"] = output[0].detach()
        else:
            activation["out"] = output.detach()

    target = model
    for part in layer_name.split("."):
        if part.isdigit():
            target = target[int(part)]
        else:
            target = getattr(target, part)

    handle = target.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            if forward_fn is not None:
                forward_fn(model, x, **(forward_kwargs or {}))
            elif isinstance(x, dict):
                model(**x)
            else:
                model(x)
    finally:
        handle.remove()

    return activation["out"]


def compute_noise_percent_linear(
    orig_model: nn.Module,
    modified_model: nn.Module,
    layer_name: str,
    inputs: Any,
    *,
    batch_size: int = 4,
    forward_fn: Any = None,
    forward_kwargs: dict[str, Any] | None = None,
) -> float:
    try:
        orig_model.eval()
        modified_model.eval()

        out_orig = _get_layer_output_hook(
            orig_model,
            layer_name,
            inputs,
            forward_fn=forward_fn,
            forward_kwargs=forward_kwargs,
        )
        out_modified = _get_layer_output_hook(
            modified_model,
            layer_name,
            inputs,
            forward_fn=forward_fn,
            forward_kwargs=forward_kwargs,
        )

        out_orig_np = out_orig.cpu().float().numpy()
        out_mod_np = out_modified.cpu().float().numpy()

        err = out_orig_np - out_mod_np
        err_centered = err - np.mean(err)
        std_err = np.std(err_centered)
        std_orig = np.std(out_orig_np)
        return float((std_err / (std_orig + 1e-12)) * 100.0)
    except Exception as error:
        print(f"    [Noise] Failed for {layer_name}: {error}")
        return float("nan")