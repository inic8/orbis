# SPDX-License-Identifier: MIT
# Author: Dr Shashank Pathak
# Email: shashank@computer.org
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from export.common import ArtifactResolver, ModelLoader, OnnxExporter, OnnxSessionFactory, VitOnnxExporter
from export.debug_runner import SingleOutputParityRunner
from export.deployment import default_tensorrt_engine_path, deployment_target_choices, get_deployment_target_profile
from export.export_workflow import ExportVerificationWorkflow, ExportWorkflowConfig


ARTIFACT_RESOLVER = ArtifactResolver()
MODEL_LOADER = ModelLoader()
ONNX_EXPORTER = OnnxExporter()
ORT_SESSION_FACTORY = OnnxSessionFactory()
VIT_ONNX_EXPORTER = VitOnnxExporter()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model.vit to ONNX using repo-local defaults.")
    ARTIFACT_RESOLVER.add_common_arguments(parser)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--swin-module-index", type=int, default=0)
    parser.add_argument("--window-attention-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--target-frames", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--disable-constant-folding", action="store_true")
    parser.add_argument("--deployment-target", choices=deployment_target_choices(), default="generic")
    parser.add_argument("--build-tensorrt", action="store_true", help="Also build a TensorRT engine from the exported ONNX artifact.")
    parser.add_argument("--engine-path", type=str, default=None, help="Optional TensorRT engine output path override.")
    parser.add_argument("--tensorrt-backend", choices=["auto", "python", "trtexec"], default="auto")
    parser.add_argument("--trt-workspace-gb", type=float, default=None)
    parser.add_argument("--trt-opt-batch-size", type=int, default=None)
    parser.add_argument("--trt-max-batch-size", type=int, default=None)
    parser.add_argument("--trt-max-context-frames", type=int, default=None)
    parser.add_argument("--trt-max-target-frames", type=int, default=None)
    parser.add_argument("--trt-fp16", dest="trt_fp16", action="store_true", help="Force FP16 TensorRT engine building.")
    parser.add_argument("--trt-no-fp16", dest="trt_fp16", action="store_false", help="Disable FP16 TensorRT engine building.")
    parser.set_defaults(trt_fp16=None)
    args = parser.parse_args()
    args.artifacts = ARTIFACT_RESOLVER.resolve_from_args(args)
    target_profile = get_deployment_target_profile(args.deployment_target)
    args.trt_workspace_gb = target_profile.default_workspace_gb if args.trt_workspace_gb is None else args.trt_workspace_gb
    args.trt_opt_batch_size = target_profile.default_opt_batch_size if args.trt_opt_batch_size is None else args.trt_opt_batch_size
    args.trt_max_batch_size = target_profile.default_max_batch_size if args.trt_max_batch_size is None else args.trt_max_batch_size
    args.trt_enable_fp16 = target_profile.default_enable_fp16 if args.trt_fp16 is None else args.trt_fp16
    args.trt_max_context_frames = args.context_frames if args.trt_max_context_frames is None else args.trt_max_context_frames
    args.trt_max_target_frames = args.target_frames if args.trt_max_target_frames is None else args.trt_max_target_frames
    args.engine_path = (
        ARTIFACT_RESOLVER.expand_path(args.engine_path)
        if args.engine_path
        else default_tensorrt_engine_path(args.artifacts.onnx_path, args.deployment_target, enable_fp16=args.trt_enable_fp16)
    )
    return args


def main() -> None:
    args = parse_args()
    workflow = ExportVerificationWorkflow(
        config=ExportWorkflowConfig(
            artifacts=args.artifacts,
            block_index=args.block_index,
            swin_module_index=args.swin_module_index,
            window_attention_index=args.window_attention_index,
            batch_size=args.batch_size,
            context_frames=args.context_frames,
            target_frames=args.target_frames,
            num_samples=args.num_samples,
            opset=args.opset,
            atol=args.atol,
            rtol=args.rtol,
            do_constant_folding=not args.disable_constant_folding,
            deployment_target=args.deployment_target,
            tensorrt_engine_path=args.engine_path if args.build_tensorrt else None,
            tensorrt_backend=args.tensorrt_backend,
            tensorrt_workspace_gb=args.trt_workspace_gb,
            tensorrt_enable_fp16=args.trt_enable_fp16,
            tensorrt_opt_batch_size=args.trt_opt_batch_size,
            tensorrt_max_batch_size=args.trt_max_batch_size,
            tensorrt_max_context_frames=args.trt_max_context_frames,
            tensorrt_max_target_frames=args.trt_max_target_frames,
        ),
        model_loader=MODEL_LOADER,
        parity_runner=SingleOutputParityRunner(
            ONNX_EXPORTER,
            ORT_SESSION_FACTORY,
            atol=args.atol,
            rtol=args.rtol,
        ),
        session_factory=ORT_SESSION_FACTORY,
        vit_exporter=VIT_ONNX_EXPORTER,
    )
    raise SystemExit(workflow.run())


if __name__ == "__main__":
    main()