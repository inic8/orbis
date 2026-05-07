#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torchvision.utils import save_image
from tqdm import tqdm

# Ensure project root (one level up from this file) is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from rollout_onnx import (  # noqa: E402
    build_output_dir,
    ensure_dir,
    ensure_tokenizer_env,
    gif_from_frames,
    length_of,
    logger,
    resolve_input_hw,
    str2bool,
)

try:
    from external.orbis.util import instantiate_from_config  # noqa: E402
except ModuleNotFoundError:
    from util import instantiate_from_config  # noqa: E402


def _append_system_dist_packages() -> None:
    candidate_paths = [
        Path("/usr/lib/python3/dist-packages"),
        Path(f"/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages"),
    ]
    for candidate in candidate_paths:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.append(str(candidate))


_append_system_dist_packages()

try:
    import tensorrt as trt  # type: ignore
except ModuleNotFoundError:
    trt = None


def _require_tensorrt() -> None:
    if trt is None:
        raise ModuleNotFoundError(
            "TensorRT Python bindings are not available in the active environment. "
            "Install python3-libnvinfer or another TensorRT Python package before using rollout_device.py."
        )


def get_ckpt_epoch_step(ckpt_path: Path) -> Tuple[int, int]:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    return int(ckpt.get("epoch", 0)), int(ckpt.get("global_step", 0))


def default_engine_path(exp_dir: Path) -> Path:
    engine_dir = exp_dir / "tensorrt"
    matches = [path for path in engine_dir.glob("*.engine") if path.is_file()]
    if not matches:
        raise FileNotFoundError(f"No TensorRT engine files found in {engine_dir}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def build_device_output_dir(
    exp_dir: Path,
    frames_dir_arg: Optional[str],
    ckpt_path: Path,
    val_config: Optional[str],
    num_steps: int,
    backend_tag: str = "tensorrt",
) -> Path:
    if frames_dir_arg is not None:
        return (exp_dir / frames_dir_arg).resolve() if not Path(frames_dir_arg).is_absolute() else Path(frames_dir_arg).resolve()
    epoch, global_step = get_ckpt_epoch_step(ckpt_path)
    data_tag = Path(val_config).stem if val_config is not None else "default_data"
    rel = Path(f"gen_rollout_{backend_tag}") / data_tag / f"ep{epoch}iter{global_step}_{num_steps}steps"
    return (exp_dir / rel).resolve()


class TensorRTRunner:
    def __init__(self, engine_path: Path):
        _require_tensorrt()
        runtime_logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(runtime_logger)
        self._engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Failed to create TensorRT execution context")
        self._output_names = [
            self._engine.get_tensor_name(index)
            for index in range(self._engine.num_io_tensors)
            if self._engine.get_tensor_mode(self._engine.get_tensor_name(index)) == trt.TensorIOMode.OUTPUT
        ]
        if not self._output_names:
            raise RuntimeError("TensorRT engine exposes no output tensors")

    def forward(
        self,
        target_t: torch.Tensor,
        context: torch.Tensor,
        t: torch.Tensor,
        frame_rate: torch.Tensor,
    ) -> torch.Tensor:
        if target_t.device.type != "cuda":
            raise ValueError("TensorRT inference requires CUDA tensors")

        inputs = {
            "target_t": target_t.contiguous().float(),
            "context": context.contiguous().float(),
            "t": t.contiguous().float(),
            "frame_rate": frame_rate.contiguous().float(),
        }
        for name, tensor in inputs.items():
            self._context.set_input_shape(name, tuple(int(dim) for dim in tensor.shape))

        output_name = self._output_names[0]
        output_shape = tuple(int(dim) for dim in self._context.get_tensor_shape(output_name))
        output = torch.empty(output_shape, device=target_t.device, dtype=torch.float32)

        for name, tensor in inputs.items():
            self._context.set_tensor_address(name, int(tensor.data_ptr()))
        self._context.set_tensor_address(output_name, int(output.data_ptr()))

        stream = torch.cuda.current_stream(device=target_t.device)
        if not self._context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return output.to(dtype=target_t.dtype)


@torch.no_grad()
def sample_tensorrt(
    *,
    model: torch.nn.Module,
    runner: TensorRTRunner,
    images: torch.Tensor,
    eta: float,
    nfe: int,
    num_samples: int,
    frame_rate: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    context = images.clone()

    if frame_rate is None:
        frame_rate = torch.full((num_samples,), 5.0, device=device, dtype=torch.float32)
    else:
        frame_rate = frame_rate.to(device=device, dtype=torch.float32)

    input_h, input_w = resolve_input_hw(model.vit.input_size)
    target_t = torch.randn(num_samples, 1, model.vit.in_channels, input_h, input_w, device=device)
    t_steps = torch.linspace(1, 0, nfe + 1, device=device)

    for step_index in range(nfe):
        t = t_steps[step_index].repeat(target_t.shape[0])
        neg_v = runner.forward(
            target_t,
            context,
            t=t * model.timescale,
            frame_rate=frame_rate,
        )
        dt = t_steps[step_index] - t_steps[step_index + 1]
        dw = torch.randn_like(target_t) * torch.sqrt(dt)
        target_t = target_t + neg_v * dt + eta * torch.sqrt(2 * dt) * dw

    last_frame = target_t.clone()
    images_out = model.decode_frames(last_frame)
    return target_t.squeeze(1), images_out


@torch.no_grad()
def roll_out_tensorrt(
    *,
    model: torch.nn.Module,
    runner: TensorRTRunner,
    x_0: torch.Tensor,
    num_gen_frames: int,
    eta: float,
    nfe: int,
    num_samples: int,
    frame_rate: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_c = model.encode_frames(x_0)
    x_all = x_c.clone()
    samples = []

    for _ in tqdm(range(num_gen_frames), desc="Rolling out frames", leave=False):
        x_last, sample = sample_tensorrt(
            model=model,
            runner=runner,
            images=x_c,
            eta=eta,
            nfe=nfe,
            num_samples=num_samples,
            frame_rate=frame_rate,
        )
        x_all = torch.cat([x_all, x_last.unsqueeze(1)], dim=1)
        x_c = torch.cat([x_c[:, 1:], x_last.unsqueeze(1)], dim=1)
        samples.append(sample)

    return x_all, torch.cat(samples, dim=1)


def _compare_rollout_dirs(frames_dir: Path, reference_dir: Path) -> dict[str, float | int | str | None]:
    current_fake_dir = frames_dir / "fake_images"
    reference_fake_dir = reference_dir / "fake_images"
    if not current_fake_dir.exists():
        raise FileNotFoundError(f"Generated fake_images directory not found: {current_fake_dir}")
    if not reference_fake_dir.exists():
        raise FileNotFoundError(f"Reference fake_images directory not found: {reference_fake_dir}")

    current_frames = sorted(path for path in current_fake_dir.glob("sequence_*/*.jpg") if path.is_file())
    if not current_frames:
        raise FileNotFoundError(f"No generated frame files found under {current_fake_dir}")

    frame_count = 0
    total_mae = 0.0
    total_mse = 0.0
    psnr_values: list[float] = []
    missing_references = 0

    for current_frame in current_frames:
        relative_path = current_frame.relative_to(current_fake_dir)
        reference_frame = reference_fake_dir / relative_path
        if not reference_frame.exists():
            missing_references += 1
            continue
        current_image = imageio.imread(current_frame).astype(np.float32)
        reference_image = imageio.imread(reference_frame).astype(np.float32)
        if current_image.shape != reference_image.shape:
            raise ValueError(f"Mismatched frame shape for {relative_path}: {current_image.shape} vs {reference_image.shape}")
        diff = current_image - reference_image
        mae = float(np.mean(np.abs(diff)))
        mse = float(np.mean(np.square(diff)))
        psnr = float("inf") if mse == 0.0 else float(20.0 * math.log10(255.0) - 10.0 * math.log10(mse))
        total_mae += mae
        total_mse += mse
        psnr_values.append(psnr)
        frame_count += 1

    return {
        "reference_dir": str(reference_dir),
        "frames_compared": frame_count,
        "missing_reference_frames": missing_references,
        "mean_absolute_error": (total_mae / frame_count) if frame_count else None,
        "mean_squared_error": (total_mse / frame_count) if frame_count else None,
        "mean_psnr_db": (sum(psnr_values) / frame_count) if frame_count else None,
    }


@torch.inference_mode()
def generate_images(args: argparse.Namespace, unknown_args: List[str]) -> None:
    _require_tensorrt()

    frames_dir = Path(args.frames_dir)
    if frames_dir.exists():
        logger.warning(
            "Output folder exists. New images will be added to the same folder. "
            "Delete it if you want to start from scratch."
        )
    else:
        ensure_dir(frames_dir)

    torch.backends.cudnn.deterministic = True
    seed_everything(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT rollout requires CUDA, but torch.cuda.is_available() is False")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("TensorRT rollout requires a CUDA device")
    torch.cuda.set_device(device)
    logger.info(f"Using device: {device}")

    ensure_tokenizer_env(Path(args.config), Path(args.exp_dir))

    base_cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, OmegaConf.from_dotlist(unknown_args))

    model = instantiate_from_config(cfg.model)
    state = torch.load(str(args.ckpt), map_location="cpu")["state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    runner = TensorRTRunner(Path(args.engine))
    if args.evaluate_ema:
        logger.info("Using weights baked into the TensorRT engine; evaluate_ema does not switch weights at runtime.")

    if args.val_config is not None:
        data_cfg = OmegaConf.merge(
            OmegaConf.load(args.val_config), OmegaConf.from_dotlist(unknown_args)
        )
    else:
        data_cfg = cfg

    num_condition_frames = None
    if args.save_real:
        validation_params = data_cfg.data.params.validation.params
        if hasattr(validation_params, "num_frames"):
            num_condition_frames = int(validation_params.num_frames) - 1
            validation_params.num_frames = num_condition_frames + args.num_gen_frames

    if hasattr(data_cfg.data.params, "train"):
        del data_cfg.data.params.train

    data_module = instantiate_from_config(data_cfg.data)
    data_module.prepare_data()
    data_module.setup()
    val_loader = data_module.val_dataloader()

    logger.info(f"Saving outputs to: {args.frames_dir}")
    total_batches = length_of(val_loader)
    pbar = tqdm(total=total_batches, desc="Generating", dynamic_ncols=True)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    rollout_start_time = time.perf_counter()

    sample_idx = 0
    for batch in val_loader:
        if args.num_videos is not None and sample_idx >= args.num_videos:
            break

        if isinstance(batch, dict):
            x = batch["images"].to(device, non_blocking=True)
            frame_rate = batch.get("frame_rate")
            if frame_rate is not None:
                frame_rate = frame_rate.to(device, non_blocking=True)
        else:
            x = batch.to(device, non_blocking=True)
            frame_rate = None

        if num_condition_frames is None:
            num_condition_frames = max(1, x.shape[1] - args.num_gen_frames)
        if x.shape[1] < num_condition_frames:
            raise ValueError(
                f"Validation sample provides {x.shape[1]} frames, but the TensorRT rollout needs {num_condition_frames} conditioning frames."
            )

        cond_x = x[:, :num_condition_frames]

        _, gen_frames = roll_out_tensorrt(
            model=model,
            runner=runner,
            x_0=cond_x,
            num_gen_frames=args.num_gen_frames,
            eta=args.eta,
            nfe=args.num_steps,
            num_samples=cond_x.size(0),
            frame_rate=frame_rate,
        )

        for batch_index in range(x.size(0)):
            if args.num_videos is not None and sample_idx >= args.num_videos:
                break

            seq_name = f"sequence_{sample_idx:04d}"
            fake_dir = frames_dir / "fake_images" / seq_name
            gif_dir = frames_dir / "gen_gifs"
            ensure_dir(fake_dir)
            ensure_dir(gif_dir)

            seq_frames = gen_frames[batch_index]
            time_steps = seq_frames.shape[0]

            for frame_index in range(time_steps):
                frame = (seq_frames[frame_index] + 1.0) / 2.0
                save_image(frame, fake_dir / f"frame_{frame_index:04d}.jpg")

            gif_frames = gif_from_frames([seq_frames[frame_index] for frame_index in range(time_steps)], fps=7)
            imageio.mimsave(gif_dir / f"{seq_name}.gif", gif_frames, fps=7, loop=0)

            if args.save_real:
                real_dir = frames_dir / "real_images" / seq_name
                ensure_dir(real_dir)
                for frame_index in range(x.shape[1]):
                    real_frame = (x[batch_index, frame_index] + 1.0) / 2.0
                    save_image(real_frame, real_dir / f"frame_{frame_index:04d}.jpg")

            sample_idx += 1

        if total_batches is not None:
            pbar.update(1)
        else:
            pbar.set_postfix_str(f"samples={sample_idx}")

    pbar.close()
    torch.cuda.synchronize(device)
    rollout_latency_ms = (time.perf_counter() - rollout_start_time) * 1000.0
    peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1024**3

    report = {
        "backend": "tensorrt",
        "checkpoint_path": str(args.ckpt),
        "engine_path": str(args.engine),
        "frames_dir": str(frames_dir),
        "num_videos": sample_idx,
        "num_gen_frames": int(args.num_gen_frames),
        "num_steps": int(args.num_steps),
        "seed": int(args.seed),
        "device": str(device),
        "latency_ms": rollout_latency_ms,
        "latency_ms_per_video": (rollout_latency_ms / sample_idx) if sample_idx else None,
        "peak_memory_gb": peak_memory_gb,
    }

    if args.reference_rollout_dir is not None:
        comparison = _compare_rollout_dirs(frames_dir, Path(args.reference_rollout_dir))
        report["comparison"] = comparison
        logger.info(f"Saved rollout comparison against {args.reference_rollout_dir}")

    report_path = frames_dir / "rollout_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"Saved rollout report to: {report_path}")

    logger.info(f"Max CUDA memory: {peak_memory_gb:.02f} GB")
    logger.info(f"Rollout latency: {rollout_latency_ms:.02f} ms")


def parse_args(argv: Optional[List[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(description="Generate rollouts (frames + GIFs) from a TensorRT engine.")
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment directory (contains config and checkpoints).")
    parser.add_argument("--ckpt", type=str, default="checkpoints/last.ckpt", help="Checkpoint path, relative to exp_dir.")
    parser.add_argument("--engine", type=str, default=None, help="TensorRT engine path, relative to exp_dir if relative. Defaults to the newest exp_dir/tensorrt/*.engine.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config path, relative to exp_dir.")
    parser.add_argument("--val_config", type=str, default=None, help="Optional validation data config path (absolute or relative to exp_dir).")
    parser.add_argument("--num_gen_frames", type=int, default=1, help="Number of frames to generate (roll-out length).")
    parser.add_argument("--frames_dir", type=str, default=None, help="Output directory for frames/GIFs (relative to exp_dir if relative).")
    parser.add_argument("--save_real", type=str2bool, default=False, help="Also save ground-truth frames next to generated ones.")
    parser.add_argument("--num_videos", type=int, default=None, help="Generate at most this many sequences (None = all).")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed.")
    parser.add_argument("--device", type=str, default="cuda:0", help='CUDA device string, for example "cuda:0".')
    parser.add_argument("--num_steps", type=int, default=30, help="Sampler steps (passed to roll_out as NFE).")
    parser.add_argument("--eta", type=float, default=0.0, help="Stochasticity for sampling (passed to roll_out).")
    parser.add_argument("--evaluate_ema", type=str2bool, default=True, help="Kept for CLI compatibility; TensorRT uses the weights baked into the engine.")
    parser.add_argument("--reference_rollout_dir", type=str, default=None, help="Optional rollout directory to compare against, typically from rollout.py or rollout_onnx.py.")
    args, unknown = parser.parse_known_args(argv)

    exp_dir = Path(args.exp_dir).resolve()
    ckpt = (exp_dir / args.ckpt).resolve() if not Path(args.ckpt).is_absolute() else Path(args.ckpt).resolve()
    engine_path = default_engine_path(exp_dir) if args.engine is None else ((exp_dir / args.engine).resolve() if not Path(args.engine).is_absolute() else Path(args.engine).resolve())
    config = (exp_dir / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
    val_config = (
        (exp_dir / args.val_config).resolve()
        if args.val_config and not Path(args.val_config).is_absolute()
        else (Path(args.val_config).resolve() if args.val_config else None)
    )
    reference_rollout_dir = (
        (exp_dir / args.reference_rollout_dir).resolve()
        if args.reference_rollout_dir and not Path(args.reference_rollout_dir).is_absolute()
        else (Path(args.reference_rollout_dir).resolve() if args.reference_rollout_dir else None)
    )

    if not exp_dir.exists():
        raise FileNotFoundError(f"exp_dir not found: {exp_dir}")
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not engine_path.exists():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
    if not config.exists():
        raise FileNotFoundError(f"Config not found: {config}")
    if val_config is not None and not val_config.exists():
        raise FileNotFoundError(f"val_config not found: {val_config}")
    if reference_rollout_dir is not None and not reference_rollout_dir.exists():
        raise FileNotFoundError(f"reference_rollout_dir not found: {reference_rollout_dir}")

    frames_dir = build_device_output_dir(
        exp_dir=exp_dir,
        frames_dir_arg=args.frames_dir,
        ckpt_path=ckpt,
        val_config=str(val_config) if val_config is not None else None,
        num_steps=args.num_steps,
        backend_tag="tensorrt",
    )

    args.exp_dir = str(exp_dir)
    args.ckpt = str(ckpt)
    args.engine = str(engine_path)
    args.config = str(config)
    args.val_config = str(val_config) if val_config is not None else None
    args.frames_dir = str(frames_dir)
    args.reference_rollout_dir = str(reference_rollout_dir) if reference_rollout_dir is not None else None
    return args, unknown


def main(argv: Optional[List[str]] = None) -> None:
    args, unknown = parse_args(argv)
    generate_images(args, unknown)


if __name__ == "__main__":
    main()