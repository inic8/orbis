#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import onnx
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_EXAMPLE_IMAGES = [
    REPO_ROOT / "imgs" / "example" / f"frame_{index:04d}.jpg"
    for index in range(5)
]


def _find_latest_file(directory: Path, pattern: str) -> Path | None:
    matches = [path for path in directory.glob(pattern) if path.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def resolve_checkpoint_path(run_dir: Path, checkpoint_arg: str | None) -> Path:
    if checkpoint_arg:
        checkpoint_path = Path(checkpoint_arg)
        return checkpoint_path.resolve() if checkpoint_path.is_absolute() else (run_dir / checkpoint_path).resolve()

    default_checkpoint = run_dir / "checkpoints" / "last.ckpt"
    if default_checkpoint.is_file():
        return default_checkpoint.resolve()
    latest_checkpoint = _find_latest_file(run_dir / "checkpoints", "*.ckpt")
    if latest_checkpoint is None:
        raise FileNotFoundError(f"No checkpoint files found under {run_dir / 'checkpoints'}")
    return latest_checkpoint.resolve()


def resolve_onnx_path(run_dir: Path, onnx_arg: str | None) -> Path:
    if onnx_arg:
        onnx_path = Path(onnx_arg)
        return onnx_path.resolve() if onnx_path.is_absolute() else (run_dir / onnx_path).resolve()

    candidate_names = ["last.onnx", "last_enhanced.onnx"]
    for candidate_name in candidate_names:
        candidate = run_dir / "onnx" / candidate_name
        if candidate.is_file():
            return candidate.resolve()
    latest_onnx = _find_latest_file(run_dir / "onnx", "*.onnx")
    if latest_onnx is None:
        raise FileNotFoundError(f"No ONNX files found under {run_dir / 'onnx'}")
    return latest_onnx.resolve()


def resolve_engine_path(run_dir: Path, engine_arg: str | None) -> Path:
    if engine_arg:
        engine_path = Path(engine_arg)
        return engine_path.resolve() if engine_path.is_absolute() else (run_dir / engine_path).resolve()

    latest_engine = _find_latest_file(run_dir / "tensorrt", "*.engine")
    if latest_engine is None:
        raise FileNotFoundError(f"No TensorRT engine files found under {run_dir / 'tensorrt'}")
    return latest_engine.resolve()


def infer_condition_frames(onnx_path: Path, num_available_images: int) -> int:
    model = onnx.load(str(onnx_path))
    for graph_input in model.graph.input:
        if graph_input.name != "context":
            continue
        dims = graph_input.type.tensor_type.shape.dim
        if len(dims) > 1 and dims[1].HasField("dim_value"):
            return int(dims[1].dim_value)
    return max(1, num_available_images - 1)


def build_validation_override(
    *,
    config_path: Path,
    image_paths: List[Path],
    output_dir: Path,
) -> Path:
    config = OmegaConf.load(config_path)
    validation_params = config.data.params.validation.params
    size = list(validation_params.size)

    validation_override = OmegaConf.create(
        {
            "data": {
                "target": "data.datamodule.DataModuleFromConfig",
                "params": {
                    "batch_size": 1,
                    "num_workers": 0,
                    "validation": {
                        "target": "data.multiframe_val.MultiFrameFromPaths",
                        "params": {
                            "image_paths": [str(path.resolve()) for path in image_paths],
                            "size": size,
                            "num_frames": len(image_paths),
                        },
                    },
                },
            }
        }
    )
    validation_config_path = output_dir / "compare_exported_models_validation.yaml"
    OmegaConf.save(validation_override, validation_config_path)
    return validation_config_path.resolve()


def run_rollout(command: List[str], *, env: Dict[str, str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True, env=env)


def load_report(report_path: Path) -> Dict[str, Any]:
    if not report_path.is_file():
        raise FileNotFoundError(f"Expected rollout report not found: {report_path}")
    return json.loads(report_path.read_text(encoding="utf-8"))


def build_summary(*, baseline: Dict[str, Any], onnx: Dict[str, Any], tensorrt: Dict[str, Any]) -> Dict[str, Any]:
    baseline_latency = baseline.get("latency_ms")
    baseline_memory = baseline.get("peak_memory_gb")

    def _decorate(report: Dict[str, Any]) -> Dict[str, Any]:
        latency = report.get("latency_ms")
        memory = report.get("peak_memory_gb")
        return {
            **report,
            "latency_pct_of_baseline": ((latency / baseline_latency) * 100.0) if baseline_latency and latency is not None else None,
            "latency_delta_pct_vs_baseline": (((latency - baseline_latency) / baseline_latency) * 100.0) if baseline_latency and latency is not None else None,
            "memory_pct_of_baseline": ((memory / baseline_memory) * 100.0) if baseline_memory and memory is not None else None,
            "memory_delta_pct_vs_baseline": (((memory - baseline_memory) / baseline_memory) * 100.0) if baseline_memory and memory is not None else None,
        }

    return {
        "baseline": _decorate(baseline),
        "onnx": _decorate(onnx),
        "tensorrt": _decorate(tensorrt),
    }


def resolve_python_executable() -> Path:
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_python.is_file():
        return venv_python
    return Path(sys.executable).resolve()


def build_rollout_environment() -> Dict[str, str]:
    env = os.environ.copy()
    logs_tk_dir = REPO_ROOT / "logs_tk"
    if logs_tk_dir.is_dir() and "TK_WORK_DIR" not in env:
        env["TK_WORK_DIR"] = str(logs_tk_dir)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequentially compare baseline, ONNX, and TensorRT exported-model rollout latency and memory against the same input frames."
    )
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory containing config.yaml, checkpoints/, onnx/, and tensorrt/.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path override, relative to run-dir if relative.")
    parser.add_argument("--onnx", type=str, default=None, help="Optional ONNX path override, relative to run-dir if relative.")
    parser.add_argument("--engine", type=str, default=None, help="Optional TensorRT engine path override, relative to run-dir if relative.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config path, relative to run-dir if relative.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for sequential rollout outputs and the comparison summary.")
    parser.add_argument("--num-gen-frames", type=int, default=5, help="Number of frames to generate in each rollout.")
    parser.add_argument("--num-steps", type=int, default=30, help="Sampler steps for each rollout.")
    parser.add_argument("--eta", type=float, default=0.0, help="Sampling stochasticity for each rollout.")
    parser.add_argument("--seed", type=int, default=42, help="Common seed used across all three rollout runs.")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device string used for each rollout.")
    parser.add_argument(
        "--example-images",
        nargs="*",
        default=None,
        help="Optional replacement image paths. By default the script uses the five files under imgs/example/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    config_path = Path(args.config)
    config_path = config_path.resolve() if config_path.is_absolute() else (run_dir / config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    image_paths = [Path(path).resolve() for path in args.example_images] if args.example_images else [path.resolve() for path in DEFAULT_EXAMPLE_IMAGES]
    if len(image_paths) != 5:
        raise ValueError(f"Expected exactly 5 input images, got {len(image_paths)}")
    missing_images = [str(path) for path in image_paths if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing input image(s): {missing_images}")

    checkpoint_path = resolve_checkpoint_path(run_dir, args.checkpoint)
    onnx_path = resolve_onnx_path(run_dir, args.onnx)
    engine_path = resolve_engine_path(run_dir, args.engine)
    num_condition_frames = infer_condition_frames(onnx_path, len(image_paths))
    if len(image_paths) < num_condition_frames:
        raise ValueError(
            f"Need at least {num_condition_frames} input image(s) for the exported model, but only {len(image_paths)} were provided."
        )

    output_dir = Path(args.output_dir).resolve() if args.output_dir else (run_dir / "compare_exported_models").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_config_path = build_validation_override(
        config_path=config_path,
        image_paths=image_paths,
        output_dir=output_dir,
    )

    python_executable = resolve_python_executable()
    rollout_env = build_rollout_environment()
    baseline_frames_dir = output_dir / "baseline"
    onnx_frames_dir = output_dir / "onnx"
    tensorrt_frames_dir = output_dir / "tensorrt"

    common_args = [
        "--exp_dir",
        str(run_dir),
        "--ckpt",
        str(checkpoint_path),
        "--config",
        str(config_path),
        "--val_config",
        str(validation_config_path),
        "--num_gen_frames",
        str(args.num_gen_frames),
        "--num_condition_frames",
        str(num_condition_frames),
        "--num_steps",
        str(args.num_steps),
        "--eta",
        str(args.eta),
        "--seed",
        str(args.seed),
        "--device",
        str(args.device),
        "--save_real",
        "false",
    ]

    run_rollout(
        [
            str(python_executable),
            str(REPO_ROOT / "evaluate" / "rollout.py"),
            *common_args,
            "--frames_dir",
            str(baseline_frames_dir),
        ],
        env=rollout_env,
    )
    baseline_report = load_report(baseline_frames_dir / "rollout_report.json")

    run_rollout(
        [
            str(python_executable),
            str(REPO_ROOT / "evaluate" / "rollout_onnx.py"),
            *common_args,
            "--onnx",
            str(onnx_path),
            "--frames_dir",
            str(onnx_frames_dir),
        ],
        env=rollout_env,
    )
    onnx_report = load_report(onnx_frames_dir / "rollout_report.json")

    run_rollout(
        [
            str(python_executable),
            str(REPO_ROOT / "evaluate" / "rollout_device.py"),
            *common_args,
            "--engine",
            str(engine_path),
            "--frames_dir",
            str(tensorrt_frames_dir),
        ],
        env=rollout_env,
    )
    tensorrt_report = load_report(tensorrt_frames_dir / "rollout_report.json")

    summary = {
        "inputs": {
            "run_dir": str(run_dir),
            "checkpoint_path": str(checkpoint_path),
            "onnx_path": str(onnx_path),
            "engine_path": str(engine_path),
            "validation_config": str(validation_config_path),
            "image_paths": [str(path) for path in image_paths],
            "num_condition_frames": int(num_condition_frames),
            "num_gen_frames": int(args.num_gen_frames),
            "num_steps": int(args.num_steps),
            "eta": float(args.eta),
            "seed": int(args.seed),
            "device": str(args.device),
            "execution_mode": "sequential",
        },
        "results": build_summary(
            baseline=baseline_report,
            onnx=onnx_report,
            tensorrt=tensorrt_report,
        ),
    }

    summary_path = output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved comparison summary to: {summary_path}")


if __name__ == "__main__":
    main()