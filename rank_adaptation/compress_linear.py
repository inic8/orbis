# SPDX-License-Identifier: MIT
# Author: Arunachalam Thirunavukkarasu
# Contributor: Dr Shashank Pathak
# Email: arunachalam.thirunavukkarasu@dlr.de
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

from __future__ import annotations

import copy
import gc
import traceback
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import brentq

from .noise_linear import compute_noise_percent_linear
from .polynomial import compute_sensitivity
from .sweep_linear import evaluate_accuracy_vit
from .svd_decompose import LowRankLinear, compute_compression_ratio_linear, replace_linear_module, svd_decompose_linear


def distribute_budget(
    layer_results: list[dict],
    global_acc_budget: float,
    global_comp_target: float,
) -> list[dict[str, Any]]:
    valid_layers = [layer_result for layer_result in layer_results if layer_result.get("poly_coeffs") is not None]
    if not valid_layers:
        print("[Budget] Warning: No layers with valid polynomial fits")
        return []

    sensitivities = [max(compute_sensitivity(layer_result["poly_coeffs"]), 1e-8) for layer_result in valid_layers]
    inv_sensitivities = [1.0 / sensitivity for sensitivity in sensitivities]
    total_inv = sum(inv_sensitivities)
    mean_inv = total_inv / len(inv_sensitivities)

    distributions = []
    for index, layer_result in enumerate(valid_layers):
        norm_weight = inv_sensitivities[index] / total_inv
        acc_budget = global_acc_budget * norm_weight
        comp_target = global_comp_target * (inv_sensitivities[index] / mean_inv)
        distributions.append(
            {
                "name": layer_result["name"],
                "sensitivity": float(sensitivities[index]),
                "inv_sensitivity": float(inv_sensitivities[index]),
                "norm_weight": float(norm_weight),
                "acc_budget": float(acc_budget),
                "comp_target": float(comp_target),
            }
        )

    print(f"\n[Budget] Global accuracy budget   : {global_acc_budget * 100:.2f}%")
    print(f"[Budget] Global compression target : {global_comp_target:.1f}x")
    print(f'\n  {"Layer":<50} {"Sensitivity":>12} {"Acc Budget":>12} {"Comp Target":>12}')
    print("  " + "-" * 90)
    for distribution in distributions:
        print(
            f'  {distribution["name"]:<50} {distribution["sensitivity"]:>12.4f} {distribution["acc_budget"] * 100:>11.3f}% {distribution["comp_target"]:>12.2f}x'
        )
    return distributions


def invert_polynomial_for_noise(poly_coeffs, baseline_val, acc_budget):
    target = baseline_val - acc_budget
    poly = np.poly1d(poly_coeffs)
    search_noise = np.linspace(0, 100, 10000)
    f_values = poly(search_noise) - target

    crossings = []
    for index in range(len(f_values) - 1):
        if f_values[index] * f_values[index + 1] < 0:
            try:
                root = brentq(lambda noise: poly(noise) - target, search_noise[index], search_noise[index + 1])
                crossings.append(root)
            except Exception:
                pass

    if not crossings:
        return 100.0 if np.all(f_values >= 0) else None
    return max(crossings)


def compute_pareto_front(candidates: list[dict]) -> list[int]:
    count = len(candidates)
    is_pareto = np.ones(count, dtype=bool)
    for left in range(count):
        for right in range(count):
            if left == right:
                continue
            better_ratio = candidates[right]["compression_ratio"] >= candidates[left]["compression_ratio"]
            better_noise = candidates[right]["noise_percent"] <= candidates[left]["noise_percent"]
            strictly_better = (
                candidates[right]["compression_ratio"] > candidates[left]["compression_ratio"]
                or candidates[right]["noise_percent"] < candidates[left]["noise_percent"]
            )
            if better_ratio and better_noise and strictly_better:
                is_pareto[left] = False
                break
    return [index for index in range(count) if is_pareto[index]]


def sequential_compress_linear(
    orig_model: nn.Module,
    layer_results: list[dict],
    distributions: list[dict],
    baseline_quality: float,
    global_acc_budget: float,
    inputs: Any,
    device: torch.device,
    *,
    forward_fn: Callable = None,
    forward_kwargs: dict[str, Any] | None = None,
    noise_threshold: float = 50.0,
) -> tuple:
    working_model = copy.deepcopy(orig_model).to(device)
    working_model.eval()

    dist_map = {distribution["name"]: distribution for distribution in distributions}
    compression_log: dict[str, dict[str, Any]] = {}

    print("\n[Phase 2] Starting sequential compression (Linear SVD)")
    print(f"[Phase 2] Baseline quality     : {baseline_quality:.4f}")
    print(f"[Phase 2] Global budget        : {global_acc_budget * 100:.2f}%")
    print(f"[Phase 2] Noise threshold      : {noise_threshold:.1f}%")

    for layer_result in layer_results:
        name = layer_result["name"]
        if layer_result.get("poly_coeffs") is None:
            print(f"\n  [{name}] No polynomial — skipping")
            compression_log[name] = {"status": "skipped_no_poly"}
            continue
        if name not in dist_map:
            print(f"\n  [{name}] No budget distribution — skipping")
            compression_log[name] = {"status": "skipped_no_dist"}
            continue

        distribution = dist_map[name]
        acc_budget = distribution["acc_budget"]
        comp_target = distribution["comp_target"]

        print(f'\n  [{name}]')
        print(f'    Sensitivity  : {distribution["sensitivity"]:.4f}')
        print(f"    Acc budget   : {acc_budget * 100:.3f}%")
        print(f"    Comp target  : {comp_target:.2f}x")

        max_noise = invert_polynomial_for_noise(layer_result["poly_coeffs"], 1.0, acc_budget)
        if max_noise is None:
            print("    Could not invert polynomial — skipping")
            compression_log[name] = {"status": "skipped_poly_inversion"}
            continue

        max_noise = min(max_noise, noise_threshold)
        print(f"    Max noise    : {max_noise:.2f}%")

        sweep = layer_result.get("sweep_results", [])
        valid = [candidate for candidate in sweep if not np.isnan(candidate.get("noise_percent", float("nan"))) and candidate["noise_percent"] <= max_noise]
        if not valid:
            print("    No valid rank combos within noise budget — skipping")
            compression_log[name] = {"status": "skipped_no_valid_ranks"}
            continue

        pareto_candidates = [valid[index] for index in compute_pareto_front(valid)]
        print(f"    Valid combos : {len(valid)}  Pareto: {len(pareto_candidates)}")

        pareto_sorted = sorted(pareto_candidates, key=lambda candidate: abs(candidate["compression_ratio"] - comp_target))
        accepted = False
        for candidate in pareto_sorted:
            rank = candidate["rank"]
            tmp_model = None
            try:
                target = working_model
                for part in name.split("."):
                    target = target[int(part)] if part.isdigit() else getattr(target, part)

                low_rank = svd_decompose_linear(target, rank).to(device)
                tmp_model = replace_linear_module(working_model, name, low_rank).to(device)
                tmp_model.eval()

                cumulative_noise = compute_noise_percent_linear(
                    orig_model,
                    tmp_model,
                    name,
                    inputs,
                    forward_fn=forward_fn,
                    forward_kwargs=forward_kwargs,
                )
                quality = evaluate_accuracy_vit(tmp_model, inputs, forward_fn=forward_fn)

                if np.isnan(cumulative_noise) or cumulative_noise > noise_threshold:
                    continue

                working_model = tmp_model
                tmp_model = None
                accepted = True
                compression_log[name] = {
                    "status": "compressed",
                    "rank": int(rank),
                    "noise_percent": float(candidate["noise_percent"]),
                    "cumulative_noise_percent": float(cumulative_noise),
                    "compression_ratio": float(candidate["compression_ratio"]),
                    "quality": float(quality),
                    "acc_budget": float(acc_budget),
                    "comp_target": float(comp_target),
                }
                print(
                    f"    Accepted rank={rank}  candidate_noise={candidate['noise_percent']:.2f}%  cumulative_noise={cumulative_noise:.2f}%  CR={candidate['compression_ratio']:.2f}x"
                )
                break
            except Exception as error:
                print(f"    Candidate rank={rank} FAILED: {error}")
                traceback.print_exc()
            finally:
                if tmp_model is not None:
                    del tmp_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not accepted:
            print("    No acceptable candidate found — skipping")
            compression_log[name] = {"status": "skipped_no_accepted_candidate"}

    return working_model, compression_log