# SPDX-License-Identifier: MIT
# Author: Arunachalam Thirunavukkarasu
# Contributor: Dr Shashank Pathak
# Email: arunachalam.thirunavukkarasu@dlr.de
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

from __future__ import annotations

import json
import os
from typing import Any

from .polynomial import compute_sensitivity
from .serialization import make_json_serializable


def save_layer_checkpoint(
    ckpt_path: str,
    model_name: str,
    layer_name: str,
    sweep: list[dict[str, Any]],
    poly_coeffs: list[float] | None,
) -> None:
    sensitivity = compute_sensitivity(poly_coeffs) if poly_coeffs else None
    data = {
        "model": model_name,
        "layer": layer_name,
        "method": "svd_low_rank",
        "sweep": sweep,
        "poly_coeffs": poly_coeffs,
        "sensitivity": sensitivity,
    }
    with open(ckpt_path, "w", encoding="utf-8") as handle:
        json.dump(make_json_serializable(data), handle, indent=2)
    print(f"  [Checkpoint] Saved to {ckpt_path}")


def load_layer_results_from_checkpoints(
    ckpt_dir: str,
    compressible: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_files = sorted([name for name in os.listdir(ckpt_dir) if name.endswith(".json") and "svd_ckpt" in name])

    if not all_files:
        return [], compressible

    print(f"\n[Checkpoints] Found {len(all_files)} file(s) in {ckpt_dir}:")

    ckpt_by_layer: dict[str, dict[str, Any]] = {}
    for file_name in all_files:
        file_path = os.path.join(ckpt_dir, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            layer_name = data["layer"]
            ckpt_by_layer[layer_name] = data
            print(f"  [Loaded] {file_name} → {layer_name}  sweep: {len(data.get('sweep', []))}")
        except Exception as error:
            print(f"  [Warning] Could not load {file_name}: {error}")

    layer_results: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for layer_info in compressible:
        name = layer_info["name"]
        if name not in ckpt_by_layer:
            missing.append(layer_info)
            continue

        checkpoint = ckpt_by_layer[name]
        layer_results.append(
            {
                **layer_info,
                "sweep_results": checkpoint.get("sweep", []),
                "poly_coeffs": checkpoint.get("poly_coeffs"),
            }
        )

    return layer_results, missing