# SPDX-License-Identifier: MIT
# Author: Dr Shashank Pathak
# Email: shashank@computer.org
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from export.common import (
    ModelLoader,
    OnnxSessionFactory,
    ResolvedArtifactPaths,
    VitOnnxExporter,
    VitOnnxWrapper,
    make_vit_inputs,
)
from export.debug_runner import ParitySuiteSummary, SingleOutputParityRunner
from export.deployment import TensorRTEngineBuildRequest, TensorRTEngineBuilder, build_vit_tensorrt_shape_profiles


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"


def style(text: str, *codes: str) -> str:
    return "".join(codes) + text + RESET


def status_badge(passed: bool) -> str:
    return style(" PASS ", BOLD, GREEN) if passed else style(" FAIL ", BOLD, RED)


def section_title(index: int, title: str) -> str:
    return f"{style(f'[{index}]', BOLD, BLUE)} {style(title, BOLD)}"


@dataclass(frozen=True)
class ExportWorkflowConfig:
    artifacts: ResolvedArtifactPaths
    block_index: int
    swin_module_index: int
    window_attention_index: int
    batch_size: int
    context_frames: int
    target_frames: int
    num_samples: int
    opset: int
    atol: float
    rtol: float
    do_constant_folding: bool
    deployment_target: str
    tensorrt_engine_path: Path | None
    tensorrt_backend: str
    tensorrt_workspace_gb: float
    tensorrt_enable_fp16: bool
    tensorrt_opt_batch_size: int
    tensorrt_max_batch_size: int
    tensorrt_max_context_frames: int
    tensorrt_max_target_frames: int


@dataclass(frozen=True)
class OperationCheckResult:
    name: str
    passed: bool
    mismatches: int
    worst_max_abs_diff: float
    worst_mean_abs_diff: float
    details: str | None = None


class ExportVerificationWorkflow:
    def __init__(
        self,
        *,
        config: ExportWorkflowConfig,
        model_loader: ModelLoader,
        parity_runner: SingleOutputParityRunner,
        session_factory: OnnxSessionFactory,
        vit_exporter: VitOnnxExporter,
        tensorrt_builder: TensorRTEngineBuilder | None = None,
    ):
        self.config = config
        self.model_loader = model_loader
        self.parity_runner = parity_runner
        self.session_factory = session_factory
        self.vit_exporter = vit_exporter
        self.tensorrt_builder = tensorrt_builder or TensorRTEngineBuilder()
        self.model: nn.Module | None = None

    def _should_reuse_existing_onnx(self) -> bool:
        onnx_path = self.config.artifacts.onnx_path
        return self.config.tensorrt_engine_path is not None and onnx_path is not None and onnx_path.is_file()

    def run(self) -> int:
        self._print_header()
        self.model = self.model_loader.load(self.config.artifacts.config_path, self.config.artifacts.ckpt_path)

        reused_existing_onnx = self._should_reuse_existing_onnx()

        if reused_existing_onnx:
            onnx_path = self.config.artifacts.onnx_path
            print(f"\n{section_title(2, 'Reusing existing ONNX artifact for TensorRT build')}")
            print(f"  {style('Artifact', DIM)} {onnx_path}")
            print(f"  {style('Note    ', DIM)} Skipping ONNX preflight, export, and parity checks because the artifact already exists.")
        else:
            print(f"\n{section_title(2, 'Running preflight ONNX parity checks')}")
            preflight_results = self._run_preflight_checks()
            for result in preflight_results:
                self._print_check_result(result)

            if not all(result.passed for result in preflight_results):
                print(f"\n{status_badge(False)} {style('Export aborted because at least one preflight check failed.', BOLD)}")
                return 1

            print(f"\n{section_title(3, 'Exporting model.vit to ONNX')}")
            onnx_path = self.vit_exporter.export(
                self._require_model(),
                self.config.artifacts.onnx_path,
                batch_size=self.config.batch_size,
                context_frames=self.config.context_frames,
                target_frames=self.config.target_frames,
                opset=self.config.opset,
                do_constant_folding=self.config.do_constant_folding,
            )
            print(f"  {style('Artifact', DIM)} {onnx_path}")

            print(f"\n{section_title(4, 'Checking exported model parity with graph optimizations disabled and enabled')}")
            final_no_opt = self._check_exported_vit_parity(disable_optimizations=True)
            final_opt = self._check_exported_vit_parity(disable_optimizations=False)
            self._print_check_result(final_no_opt)
            self._print_check_result(final_opt)

            if not final_no_opt.passed:
                print(f"\n{status_badge(False)} {style('Final exported model failed parity with graph optimizations disabled.', BOLD)}")
                return 1

            if not final_opt.passed:
                print(
                    f"\n{style(' WARN ', BOLD, YELLOW)} "
                    f"{style('Export succeeded, but graph optimizations introduce mismatches. Use ORT graph optimizations disabled.', BOLD)}"
                )
            else:
                print(
                    f"\n{status_badge(True)} "
                    f"{style('Export succeeded and final parity checks passed with graph optimizations both disabled and enabled.', BOLD)}"
                )

        if self.config.tensorrt_engine_path is not None:
            section_index = 3 if reused_existing_onnx else 5
            print(f"\n{section_title(section_index, f'Building TensorRT engine for deployment target {self.config.deployment_target}')}")
            try:
                engine_result = self._build_tensorrt_engine(onnx_path)
            except Exception as exc:
                print(f"\n{status_badge(False)} {style('TensorRT engine build failed.', BOLD)}")
                print(f"  {style(str(exc), DIM)}")
                return 1
            print(f"  {style('Artifact', DIM)} {engine_result.engine_path}")
            print(f"  {style('Backend ', DIM)} {engine_result.backend}")
            print(f"  {style('Target  ', DIM)} {engine_result.deployment_target}")
            print(f"  {style('Precision', DIM)} {'fp16' if engine_result.enable_fp16 else 'fp32'}")
        return 0

    def _print_header(self) -> None:
        artifacts = self.config.artifacts
        print(style("=" * 72, DIM))
        print(style("Methodical ONNX Export Workflow", BOLD, CYAN))
        print(style("=" * 72, DIM))
        print(f"{style('Config     ', DIM)} {artifacts.config_path}")
        print(f"{style('Checkpoint', DIM)} {artifacts.ckpt_path}")
        print(f"{style('ONNX      ', DIM)} {artifacts.onnx_path}")
        print(f"{style('Target    ', DIM)} {self.config.deployment_target}")
        if self.config.tensorrt_engine_path is not None:
            print(f"{style('TensorRT  ', DIM)} {self.config.tensorrt_engine_path}")
        print(f"{style('Tolerance ', DIM)} atol={self.config.atol:.1e} rtol={self.config.rtol:.1e}")
        print(f"\n{section_title(1, 'Loading model')}")

    def _build_tensorrt_engine(self, onnx_path: Path):
        model = self._require_model()
        input_hw = tuple(int(value) for value in getattr(model.vit, 'input_size', ()))
        if len(input_hw) != 2:
            from export.common import resolve_input_hw

            input_hw = resolve_input_hw(getattr(model.vit, 'input_size', None))
        profiles = build_vit_tensorrt_shape_profiles(
            batch_size=self.config.batch_size,
            opt_batch_size=self.config.tensorrt_opt_batch_size,
            max_batch_size=self.config.tensorrt_max_batch_size,
            context_frames=self.config.context_frames,
            max_context_frames=self.config.tensorrt_max_context_frames,
            target_frames=self.config.target_frames,
            max_target_frames=self.config.tensorrt_max_target_frames,
            in_channels=int(model.vit.in_channels),
            input_hw=input_hw,
        )
        request = TensorRTEngineBuildRequest(
            onnx_path=onnx_path,
            engine_path=self.config.tensorrt_engine_path,
            deployment_target=self.config.deployment_target,
            backend=self.config.tensorrt_backend,
            workspace_gb=self.config.tensorrt_workspace_gb,
            enable_fp16=self.config.tensorrt_enable_fp16,
            min_shapes=profiles.min_shapes,
            opt_shapes=profiles.opt_shapes,
            max_shapes=profiles.max_shapes,
        )
        return self.tensorrt_builder.build(request)

    def _require_model(self) -> nn.Module:
        if self.model is None:
            raise RuntimeError("Model has not been loaded")
        return self.model

    def _print_check_result(self, result: OperationCheckResult) -> None:
        print(
            f"  {style(result.name, BOLD)} {status_badge(result.passed)} "
            f"{style('mismatches', DIM)}={result.mismatches} "
            f"{style('max', DIM)}={result.worst_max_abs_diff:.3e} "
            f"{style('mean', DIM)}={result.worst_mean_abs_diff:.3e}"
        )
        if result.details:
            print(f"    {style(result.details, DIM)}")

    def _result_from_summary(self, name: str, summary: ParitySuiteSummary, details: str | None = None) -> OperationCheckResult:
        return OperationCheckResult(
            name=name,
            passed=summary.mismatches == 0,
            mismatches=summary.mismatches,
            worst_max_abs_diff=summary.worst_max_abs_diff,
            worst_mean_abs_diff=summary.worst_mean_abs_diff,
            details=details,
        )

    def _run_preflight_checks(self) -> list[OperationCheckResult]:
        mid_level_name, mid_level_check = self._select_mid_level_check()
        print(f"  {style('Selected mid-level attention check', DIM)}: {style(mid_level_name, BOLD, CYAN)}")
        return [
            self._check_vit_wrapper(),
            self._check_stdit_block(),
            mid_level_check(),
            self._check_window_attention(),
        ]

    def _temporary_onnx_path(self) -> Path:
        handle = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        handle.close()
        return Path(handle.name)

    def _run_temp_suite(
        self,
        name: str,
        module: nn.Module,
        sample_inputs: tuple[torch.Tensor, ...],
        *,
        random_input_factory: Callable[[], tuple[torch.Tensor, ...]],
        input_names: list[str],
        details: str | None = None,
    ) -> OperationCheckResult:
        temp_onnx_path = self._temporary_onnx_path()
        try:
            _, session = self.parity_runner.export_and_create_session(
                module,
                sample_inputs,
                temp_onnx_path,
                input_names=input_names,
                output_names=["output"],
                opset=self.config.opset,
                do_constant_folding=self.config.do_constant_folding,
                disable_optimizations=True,
            )
            summary = self.parity_runner.run_same_shape_suite(
                module,
                session,
                sample_inputs,
                num_samples=self.config.num_samples,
                random_input_factory=random_input_factory,
            )
            return self._result_from_summary(name, summary, details)
        except Exception as exc:
            return OperationCheckResult(
                name=name,
                passed=False,
                mismatches=1,
                worst_max_abs_diff=float("inf"),
                worst_mean_abs_diff=float("inf"),
                details=str(exc),
            )
        finally:
            if temp_onnx_path.exists():
                os.remove(temp_onnx_path)

    def _check_vit_wrapper(self) -> OperationCheckResult:
        model = self._require_model()
        wrapper = VitOnnxWrapper(model.vit).eval()
        sample_inputs = make_vit_inputs(
            model.vit,
            batch_size=self.config.batch_size,
            context_frames=self.config.context_frames,
            target_frames=self.config.target_frames,
        )
        return self._run_temp_suite(
            "model.vit preflight",
            wrapper,
            sample_inputs,
            random_input_factory=lambda: make_vit_inputs(
                model.vit,
                batch_size=self.config.batch_size,
                context_frames=self.config.context_frames,
                target_frames=self.config.target_frames,
            ),
            input_names=["input_0", "input_1", "input_2", "input_3"],
        )

    def _capture_block_io(self) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
        model = self._require_model()
        vit_inputs = make_vit_inputs(
            model.vit,
            batch_size=self.config.batch_size,
            context_frames=self.config.context_frames,
            target_frames=self.config.target_frames,
        )
        blocks = getattr(model.vit, "blocks", None)
        if blocks is None:
            raise ValueError("model.vit does not expose blocks")
        if self.config.block_index < 0 or self.config.block_index >= len(blocks):
            raise IndexError(f"block_index must be in [0, {len(blocks) - 1}]")

        block = blocks[self.config.block_index]
        captured: dict[str, torch.Tensor] = {}

        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
            captured["x"] = inputs[0].detach().cpu().clone()
            captured["c"] = inputs[1].detach().cpu().clone()

        handle = block.register_forward_hook(hook)
        try:
            with torch.no_grad():
                target_t, context, t, frame_rate = vit_inputs
                model.vit(target_t, context, t, frame_rate=frame_rate)
        finally:
            handle.remove()

        if "x" not in captured or "c" not in captured:
            raise RuntimeError("Failed to capture block inputs from model.vit forward")
        return block, captured["x"], captured["c"]

    def _check_stdit_block(self) -> OperationCheckResult:
        try:
            block, captured_x, captured_c = self._capture_block_io()
        except Exception as exc:
            return OperationCheckResult(
                name="STDiT block preflight",
                passed=False,
                mismatches=1,
                worst_max_abs_diff=float("inf"),
                worst_mean_abs_diff=float("inf"),
                details=str(exc),
            )

        return self._run_temp_suite(
            "STDiT block preflight",
            block.eval(),
            (captured_x, captured_c),
            random_input_factory=lambda: (
                torch.randn_like(captured_x),
                torch.randn_like(captured_c),
            ),
            input_names=["input_0", "input_1"],
            details=f"block_index={self.config.block_index}",
        )

    def _find_named_module(self, class_name: str, module_index: int) -> tuple[str, nn.Module]:
        model = self._require_model()
        matches: list[tuple[str, nn.Module]] = []
        for name, module in model.vit.named_modules():
            if type(module).__name__ == class_name:
                matches.append((name, module))

        if not matches:
            raise RuntimeError(f"No {class_name} modules found")
        if module_index < 0 or module_index >= len(matches):
            raise IndexError(f"{class_name} module index must be in [0, {len(matches) - 1}]")
        return matches[module_index]

    def _first_param(self, module: nn.Module) -> torch.nn.Parameter:
        return next(module.parameters())

    def _make_like(self, module: nn.Module, *shape: int) -> torch.Tensor:
        param = self._first_param(module)
        return torch.randn(*shape, device=param.device, dtype=param.dtype)

    def _derive_swin_block_input(self, module: nn.Module, batch_size: int = 2) -> tuple[torch.Tensor, tuple[int, int], int]:
        input_resolution = getattr(module, "input_resolution", None)
        if input_resolution is None:
            raise ValueError("SwinTransformerBlock does not expose input_resolution")
        channel_count = getattr(module, "dim", None)
        if channel_count is None:
            raise ValueError("SwinTransformerBlock does not expose dim")
        token_count = int(input_resolution[0] * input_resolution[1])
        x = self._make_like(module, batch_size, token_count, channel_count)
        return x, tuple(input_resolution), channel_count

    def _check_swin_block(self) -> OperationCheckResult:
        try:
            module_name, block = self._find_named_module("SwinTransformerBlock", self.config.swin_module_index)
            sample_x, input_resolution, channel_count = self._derive_swin_block_input(block, batch_size=max(1, self.config.batch_size))
            return self._run_temp_suite(
                "SwinTransformerBlock preflight",
                block.eval(),
                (sample_x,),
                random_input_factory=lambda: (
                    torch.randn_like(sample_x),
                ),
                input_names=["input_0"],
                details=(
                    f"module={module_name} resolution={input_resolution} channels={channel_count}"
                ),
            )
        except Exception as exc:
            return OperationCheckResult(
                name="SwinTransformerBlock preflight",
                passed=False,
                mismatches=1,
                worst_max_abs_diff=float("inf"),
                worst_mean_abs_diff=float("inf"),
                details=str(exc),
            )

    def _check_swin_attention(self) -> OperationCheckResult:
        try:
            module_name, attn = self._find_named_module("SwinAttention", self.config.swin_module_index)
            sample_x, input_resolution, channel_count = self._derive_swin_block_input(attn, batch_size=max(1, self.config.batch_size))
            return self._run_temp_suite(
                "SwinAttention preflight",
                attn.eval(),
                (sample_x,),
                random_input_factory=lambda: (
                    torch.randn_like(sample_x),
                ),
                input_names=["input_0"],
                details=(
                    f"module={module_name} resolution={input_resolution} channels={channel_count}"
                ),
            )
        except Exception as exc:
            return OperationCheckResult(
                name="SwinAttention preflight",
                passed=False,
                mismatches=1,
                worst_max_abs_diff=float("inf"),
                worst_mean_abs_diff=float("inf"),
                details=str(exc),
            )

    def _select_mid_level_check(self) -> tuple[str, Callable[[], OperationCheckResult]]:
        model = self._require_model()
        class_names = {type(module).__name__ for _, module in model.vit.named_modules()}
        if "SwinTransformerBlock" in class_names:
            return "SwinTransformerBlock", self._check_swin_block
        if "SwinAttention" in class_names:
            return "SwinAttention", self._check_swin_attention
        return "skipped", self._skip_mid_level_attention_check

    def _skip_mid_level_attention_check(self) -> OperationCheckResult:
        return OperationCheckResult(
            name="mid-level attention preflight",
            passed=True,
            mismatches=0,
            worst_max_abs_diff=0.0,
            worst_mean_abs_diff=0.0,
            details="No SwinTransformerBlock or SwinAttention modules found; skipped.",
        )

    def _derive_window_attention_input(self, module: nn.Module, batch_windows: int = 2) -> tuple[torch.Tensor, int, int]:
        window_size = getattr(module, "window_size", None)
        if window_size is None:
            raise ValueError("WindowAttention module does not expose window_size")
        if isinstance(window_size, int):
            token_count = window_size * window_size
        else:
            token_count = int(window_size[0] * window_size[1])
        channel_count = getattr(module, "dim", None)
        if channel_count is None and hasattr(module, "qkv"):
            channel_count = module.qkv.in_features
        if channel_count is None:
            raise ValueError("WindowAttention module does not expose an input channel size")
        x = self._make_like(module, batch_windows, token_count, channel_count)
        return x, token_count, channel_count

    def _check_window_attention(self) -> OperationCheckResult:
        try:
            module_name, attn = self._find_named_module("WindowAttention", self.config.window_attention_index)
            sample_x, token_count, channel_count = self._derive_window_attention_input(attn)
            return self._run_temp_suite(
                "WindowAttention preflight",
                attn.eval(),
                (sample_x,),
                random_input_factory=lambda: (
                    torch.randn_like(sample_x),
                ),
                input_names=["input_0"],
                details=f"module={module_name} tokens={token_count} channels={channel_count}",
            )
        except Exception as exc:
            return OperationCheckResult(
                name="WindowAttention preflight",
                passed=False,
                mismatches=1,
                worst_max_abs_diff=float("inf"),
                worst_mean_abs_diff=float("inf"),
                details=str(exc),
            )

    def _build_named_vit_ort_inputs(
        self,
        target_t: torch.Tensor,
        context: torch.Tensor,
        t: torch.Tensor,
        frame_rate: torch.Tensor,
    ) -> dict[str, object]:
        return {
            "target_t": target_t.cpu().numpy(),
            "context": context.cpu().numpy(),
            "t": t.cpu().numpy(),
            "frame_rate": frame_rate.cpu().numpy(),
        }

    def _check_exported_vit_parity(self, *, disable_optimizations: bool) -> OperationCheckResult:
        model = self._require_model()
        wrapper = VitOnnxWrapper(model.vit).eval()
        session = self.session_factory.create_cpu_session(
            self.config.artifacts.onnx_path,
            disable_optimizations=disable_optimizations,
        )

        captured_inputs = make_vit_inputs(
            model.vit,
            batch_size=self.config.batch_size,
            context_frames=self.config.context_frames,
            target_frames=self.config.target_frames,
        )
        captured_result = self.parity_runner.compare_sample(
            wrapper,
            session,
            captured_inputs,
            ort_inputs=self._build_named_vit_ort_inputs(*captured_inputs),
        )

        worst_max_abs_diff = captured_result.max_abs_diff
        worst_mean_abs_diff = captured_result.mean_abs_diff
        mismatches = 0 if captured_result.ok else 1

        for _ in range(self.config.num_samples):
            sample_inputs = make_vit_inputs(
                model.vit,
                batch_size=self.config.batch_size,
                context_frames=self.config.context_frames,
                target_frames=self.config.target_frames,
            )
            sample_result = self.parity_runner.compare_sample(
                wrapper,
                session,
                sample_inputs,
                ort_inputs=self._build_named_vit_ort_inputs(*sample_inputs),
            )
            if not sample_result.ok:
                mismatches += 1
            worst_max_abs_diff = max(worst_max_abs_diff, sample_result.max_abs_diff)
            worst_mean_abs_diff = max(worst_mean_abs_diff, sample_result.mean_abs_diff)

        mode_label = "disabled" if disable_optimizations else "enabled"
        return OperationCheckResult(
            name=f"final model parity (graph optimizations {mode_label})",
            passed=mismatches == 0,
            mismatches=mismatches,
            worst_max_abs_diff=worst_max_abs_diff,
            worst_mean_abs_diff=worst_mean_abs_diff,
        )