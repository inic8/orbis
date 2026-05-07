# SPDX-License-Identifier: MIT
# Author: Dr Shashank Pathak
# Email: shashank@computer.org
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import onnx


@dataclass(frozen=True)
class DeploymentTargetProfile:
    name: str
    description: str
    default_workspace_gb: float
    default_opt_batch_size: int
    default_max_batch_size: int
    default_enable_fp16: bool


DEPLOYMENT_TARGET_PROFILES: dict[str, DeploymentTargetProfile] = {
    "generic": DeploymentTargetProfile(
        name="generic",
        description="Generic TensorRT deployment target with conservative defaults.",
        default_workspace_gb=8.0,
        default_opt_batch_size=1,
        default_max_batch_size=4,
        default_enable_fp16=True,
    ),
    "orin": DeploymentTargetProfile(
        name="orin",
        description="NVIDIA Orin deployment target with smaller default batch ceilings.",
        default_workspace_gb=8.0,
        default_opt_batch_size=1,
        default_max_batch_size=2,
        default_enable_fp16=True,
    ),
    "thor": DeploymentTargetProfile(
        name="thor",
        description="NVIDIA Thor deployment target with larger default workspace and batch ceilings.",
        default_workspace_gb=16.0,
        default_opt_batch_size=1,
        default_max_batch_size=8,
        default_enable_fp16=True,
    ),
}


def deployment_target_choices() -> list[str]:
    return sorted(DEPLOYMENT_TARGET_PROFILES)


def get_deployment_target_profile(name: str) -> DeploymentTargetProfile:
    try:
        return DEPLOYMENT_TARGET_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported deployment target: {name}") from exc


def default_tensorrt_engine_path(onnx_path: Path, deployment_target: str, *, enable_fp16: bool) -> Path:
    precision_tag = "fp16" if enable_fp16 else "fp32"
    base_dir = onnx_path.parent.parent / "tensorrt"
    return base_dir / f"{onnx_path.stem}_{deployment_target}_{precision_tag}.engine"


@dataclass(frozen=True)
class TensorRTShapeProfiles:
    min_shapes: dict[str, tuple[int, ...]]
    opt_shapes: dict[str, tuple[int, ...]]
    max_shapes: dict[str, tuple[int, ...]]


def build_vit_tensorrt_shape_profiles(
    *,
    batch_size: int,
    opt_batch_size: int,
    max_batch_size: int,
    context_frames: int,
    max_context_frames: int,
    target_frames: int,
    max_target_frames: int,
    in_channels: int,
    input_hw: tuple[int, int],
) -> TensorRTShapeProfiles:
    input_h, input_w = input_hw
    min_batch_size = 1
    min_context_frames = 1
    min_target_frames = 1
    opt_batch_size = max(1, int(opt_batch_size))
    max_batch_size = max(opt_batch_size, int(max_batch_size))
    opt_context_frames = max(1, int(context_frames))
    max_context_frames = max(opt_context_frames, int(max_context_frames))
    opt_target_frames = max(1, int(target_frames))
    max_target_frames = max(opt_target_frames, int(max_target_frames))

    min_shapes = {
        "target_t": (min_batch_size, min_target_frames, in_channels, input_h, input_w),
        "context": (min_batch_size, min_context_frames, in_channels, input_h, input_w),
        "t": (min_batch_size,),
        "frame_rate": (min_batch_size,),
    }
    opt_shapes = {
        "target_t": (max(1, int(batch_size), opt_batch_size), opt_target_frames, in_channels, input_h, input_w),
        "context": (max(1, int(batch_size), opt_batch_size), opt_context_frames, in_channels, input_h, input_w),
        "t": (max(1, int(batch_size), opt_batch_size),),
        "frame_rate": (max(1, int(batch_size), opt_batch_size),),
    }
    max_shapes = {
        "target_t": (max_batch_size, max_target_frames, in_channels, input_h, input_w),
        "context": (max_batch_size, max_context_frames, in_channels, input_h, input_w),
        "t": (max_batch_size,),
        "frame_rate": (max_batch_size,),
    }
    return TensorRTShapeProfiles(min_shapes=min_shapes, opt_shapes=opt_shapes, max_shapes=max_shapes)


@dataclass(frozen=True)
class TensorRTEngineBuildRequest:
    onnx_path: Path
    engine_path: Path
    deployment_target: str
    backend: str
    workspace_gb: float
    enable_fp16: bool
    min_shapes: dict[str, tuple[int, ...]]
    opt_shapes: dict[str, tuple[int, ...]]
    max_shapes: dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class TensorRTEngineBuildResult:
    engine_path: Path
    backend: str
    deployment_target: str
    enable_fp16: bool


def onnx_has_dynamic_inputs(onnx_path: Path) -> bool:
    model = onnx.load(str(onnx_path), load_external_data=False)
    for tensor in model.graph.input:
        tensor_type = tensor.type.tensor_type
        if not tensor_type.HasField("shape"):
            return True
        for dim in tensor_type.shape.dim:
            if dim.dim_param:
                return True
            if dim.HasField("dim_value") and dim.dim_value <= 0:
                return True
            if not dim.HasField("dim_value") and not dim.dim_param:
                return True
    return False


def _shape_argument(shape_map: Mapping[str, tuple[int, ...]]) -> str:
    return ",".join(f"{name}:{'x'.join(str(dim) for dim in shape)}" for name, shape in shape_map.items())


class TensorRTEngineBuilder:
    def build(self, request: TensorRTEngineBuildRequest) -> TensorRTEngineBuildResult:
        backend = self._resolve_backend(request.backend)
        request.engine_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if backend == "python":
                self._build_with_python_bindings(request)
            elif backend == "trtexec":
                self._build_with_trtexec(request)
            else:
                raise ValueError(f"Unsupported TensorRT backend: {backend}")

            if not request.engine_path.exists() or request.engine_path.stat().st_size == 0:
                raise RuntimeError("TensorRT build completed without producing a non-empty engine file")
        except Exception:
            if request.engine_path.exists() and request.engine_path.stat().st_size == 0:
                request.engine_path.unlink()
            raise
        return TensorRTEngineBuildResult(
            engine_path=request.engine_path,
            backend=backend,
            deployment_target=request.deployment_target,
            enable_fp16=request.enable_fp16,
        )

    def _resolve_backend(self, backend: str) -> str:
        if backend == "auto":
            if self._python_bindings_available():
                return "python"
            if shutil.which("trtexec"):
                return "trtexec"
            raise RuntimeError(
                "TensorRT build requested, but neither TensorRT Python bindings nor the trtexec binary are available."
            )
        if backend == "python" and not self._python_bindings_available():
            raise RuntimeError("TensorRT Python bindings are not installed in the active environment.")
        if backend == "trtexec" and shutil.which("trtexec") is None:
            raise RuntimeError("The trtexec binary is not available on PATH.")
        return backend

    def _python_bindings_available(self) -> bool:
        try:
            import tensorrt  # noqa: F401
        except ModuleNotFoundError:
            return False
        return True

    def _build_with_python_bindings(self, request: TensorRTEngineBuildRequest) -> None:
        import tensorrt as trt

        has_dynamic_inputs = onnx_has_dynamic_inputs(request.onnx_path)

        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)

        if not parser.parse(request.onnx_path.read_bytes()):
            errors = [parser.get_error(index).desc() for index in range(parser.num_errors)]
            raise RuntimeError("TensorRT failed to parse ONNX:\n" + "\n".join(errors))

        config = builder.create_builder_config()
        workspace_bytes = int(request.workspace_gb * (1024**3))
        if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        else:
            config.max_workspace_size = workspace_bytes

        if request.enable_fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        if has_dynamic_inputs:
            profile = builder.create_optimization_profile()
            for input_index in range(network.num_inputs):
                input_tensor = network.get_input(input_index)
                input_name = input_tensor.name
                if input_name not in request.min_shapes:
                    raise RuntimeError(f"Missing TensorRT shape profile for ONNX input: {input_name}")
                profile.set_shape(
                    input_name,
                    min=request.min_shapes[input_name],
                    opt=request.opt_shapes[input_name],
                    max=request.max_shapes[input_name],
                )
            config.add_optimization_profile(profile)

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("TensorRT failed to build a serialized engine")
        request.engine_path.write_bytes(bytes(serialized_engine))

    def _build_with_trtexec(self, request: TensorRTEngineBuildRequest) -> None:
        has_dynamic_inputs = onnx_has_dynamic_inputs(request.onnx_path)
        workspace_mb = max(1, int(request.workspace_gb * 1024))
        command = [
            shutil.which("trtexec") or "trtexec",
            f"--onnx={request.onnx_path}",
            f"--saveEngine={request.engine_path}",
            f"--memPoolSize=workspace:{workspace_mb}",
            "--skipInference",
        ]
        if has_dynamic_inputs:
            command.extend([
                f"--minShapes={_shape_argument(request.min_shapes)}",
                f"--optShapes={_shape_argument(request.opt_shapes)}",
                f"--maxShapes={_shape_argument(request.max_shapes)}",
            ])
        if request.enable_fp16:
            command.append("--fp16")

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            output = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
            raise RuntimeError(f"trtexec failed with exit code {result.returncode}:\n{output}")
        if not request.engine_path.exists():
            raise RuntimeError("trtexec completed without producing the requested engine file")