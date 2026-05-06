# SPDX-License-Identifier: MIT
# Author: Dr Shashank Pathak
# Email: shashank@computer.org
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "postprocessing" / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "dataset"


@dataclass(frozen=True)
class VideoRecord:
    source_path: str
    video_key: str
    h5_path: str
    stored_frames: int
    original_frames: int
    stored_fps: float
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build HDF5 train/validation datasets from videos under postprocessing/data."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing input videos.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated HDF5 files and manifests.")
    parser.add_argument("--train-h5-name", default="train.h5", help="Filename for the training HDF5 file.")
    parser.add_argument("--val-h5-name", default="val.h5", help="Filename for the validation HDF5 file.")
    parser.add_argument("--train-list-name", default="train_hdf5_paths.txt", help="Filename for the training HDF5 path list.")
    parser.add_argument("--val-list-name", default="val_hdf5_paths.txt", help="Filename for the validation HDF5 path list.")
    parser.add_argument("--val-samples-name", default="val_samples.json", help="Filename for validation sample manifest.")
    parser.add_argument("--metadata-name", default="dataset_metadata.json", help="Filename for dataset metadata summary.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of videos reserved for validation.")
    parser.add_argument("--seed", type=int, default=23, help="Random seed for split and validation sample selection.")
    parser.add_argument("--sample-every-n", type=int, default=1, help="Store every nth frame from each video.")
    parser.add_argument("--max-frames-per-video", type=int, default=None, help="Optional cap on stored frames per video.")
    parser.add_argument("--min-stored-frames", type=int, default=6, help="Skip videos with fewer stored frames than this.")
    parser.add_argument("--num-val-samples", type=int, default=200, help="Total number of validation windows to generate.")
    parser.add_argument("--val-num-frames", type=int, default=6, help="Frames per validation sample window.")
    parser.add_argument("--stored-data-frame-rate", type=float, default=5.0, help="Frame rate label to use for the stored dataset.")
    parser.add_argument("--validation-frame-rate", type=float, default=5.0, help="Frame rate label used when generating validation windows.")
    parser.add_argument("--compression", choices=["gzip", "lzf", "none"], default="lzf", help="HDF5 compression mode.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated files.")
    return parser.parse_args()


def discover_videos(input_dir: Path, output_dir: Path) -> list[Path]:
    output_dir = output_dir.resolve(strict=False)
    videos = []
    for path in input_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if output_dir in path.resolve(strict=False).parents:
            continue
        videos.append(path)
    return sorted(videos)


def split_videos(video_paths: list[Path], val_split: float, seed: int) -> tuple[list[Path], list[Path]]:
    if not video_paths:
        return [], []

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(len(video_paths))
    val_count = int(round(len(video_paths) * val_split))
    if len(video_paths) > 1:
        val_count = min(max(val_count, 1), len(video_paths) - 1)
    else:
        val_count = 1

    val_indices = set(int(index) for index in shuffled_indices[:val_count])
    train_videos = [path for index, path in enumerate(video_paths) if index not in val_indices]
    val_videos = [path for index, path in enumerate(video_paths) if index in val_indices]

    if not train_videos and val_videos:
        train_videos = val_videos.copy()

    return train_videos, val_videos


def unique_video_key(video_path: Path, used_keys: set[str]) -> str:
    candidate = video_path.stem
    sanitized = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in candidate)
    sanitized = sanitized.strip("_") or "video"
    key = sanitized
    suffix = 1
    while key in used_keys:
        key = f"{sanitized}_{suffix:03d}"
        suffix += 1
    used_keys.add(key)
    return key


def build_hdf5_split(
    *,
    video_paths: list[Path],
    output_h5_path: Path,
    sample_every_n: int,
    max_frames_per_video: int | None,
    min_stored_frames: int,
    stored_data_frame_rate: float,
    compression: str,
) -> list[VideoRecord]:
    records: list[VideoRecord] = []
    used_keys: set[str] = set()
    compression_arg = None if compression == "none" else compression

    with h5py.File(output_h5_path, "w") as handle:
        for video_path in tqdm(video_paths, desc=f"Writing {output_h5_path.name}"):
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                print(f"Skipping unreadable video: {video_path}")
                continue

            video_key = unique_video_key(video_path, used_keys)
            dataset = None
            stored_frames = 0
            original_frames = 0
            width = 0
            height = 0

            while True:
                success, frame = capture.read()
                if not success:
                    break
                if original_frames % sample_every_n != 0:
                    original_frames += 1
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if dataset is None:
                    height, width = frame_rgb.shape[:2]
                    dataset = handle.create_dataset(
                        video_key,
                        shape=(0, height, width, 3),
                        maxshape=(None, height, width, 3),
                        chunks=(1, height, width, 3),
                        dtype=np.uint8,
                        compression=compression_arg,
                    )
                dataset.resize(stored_frames + 1, axis=0)
                dataset[stored_frames] = frame_rgb
                stored_frames += 1
                original_frames += 1

                if max_frames_per_video is not None and stored_frames >= max_frames_per_video:
                    break

            capture.release()

            if dataset is None or stored_frames < min_stored_frames:
                if dataset is not None:
                    del handle[video_key]
                print(f"Skipping short video: {video_path} (stored_frames={stored_frames})")
                continue

            dataset.attrs["source_path"] = str(video_path.resolve(strict=False))
            dataset.attrs["stored_fps"] = float(stored_data_frame_rate)
            dataset.attrs["sample_every_n"] = int(sample_every_n)
            dataset.attrs["stored_frames"] = int(stored_frames)
            dataset.attrs["original_frames_processed"] = int(original_frames)

            records.append(
                VideoRecord(
                    source_path=str(video_path.resolve(strict=False)),
                    video_key=video_key,
                    h5_path=str(output_h5_path.resolve(strict=False)),
                    stored_frames=stored_frames,
                    original_frames=original_frames,
                    stored_fps=float(stored_data_frame_rate),
                    width=width,
                    height=height,
                )
            )

    return records


def write_hdf5_path_list(path: Path, h5_paths: list[Path]) -> None:
    path.write_text("\n".join(str(h5_path.resolve(strict=False)) for h5_path in h5_paths) + "\n", encoding="utf-8")


def build_validation_samples(
    records: list[VideoRecord],
    *,
    num_frames: int,
    stored_data_frame_rate: float,
    validation_frame_rate: float,
    num_samples: int,
    seed: int,
) -> list[dict[str, str | int]]:
    if validation_frame_rate > stored_data_frame_rate:
        raise ValueError("validation_frame_rate must be <= stored_data_frame_rate")

    frame_interval = max(1, int(round(stored_data_frame_rate / validation_frame_rate)))
    windows: list[dict[str, str | int]] = []
    for record in records:
        max_start = record.stored_frames - (num_frames - 1) * frame_interval
        if max_start <= 0:
            continue
        starts = list(range(0, max_start))
        for start_frame in starts:
            windows.append(
                {
                    "h5_path": record.h5_path,
                    "video_key": record.video_key,
                    "start_frame": start_frame,
                }
            )

    if not windows:
        return []

    rng = np.random.default_rng(seed)
    if len(windows) <= num_samples:
        return windows

    sampled_indices = rng.choice(len(windows), size=num_samples, replace=False)
    return [windows[int(index)] for index in sorted(sampled_indices)]


def build_metadata(
    *,
    input_dir: Path,
    output_dir: Path,
    train_records: list[VideoRecord],
    val_records: list[VideoRecord],
    val_samples: list[dict[str, str | int]],
    args: argparse.Namespace,
) -> dict:
    all_records = train_records + val_records
    total_frames = sum(record.stored_frames for record in all_records)
    return {
        "input_dir": str(input_dir.resolve(strict=False)),
        "output_dir": str(output_dir.resolve(strict=False)),
        "num_input_videos": len(train_records) + len(val_records),
        "num_train_videos": len(train_records),
        "num_val_videos": len(val_records),
        "num_val_samples": len(val_samples),
        "total_stored_frames": total_frames,
        "sample_every_n": args.sample_every_n,
        "stored_data_frame_rate": args.stored_data_frame_rate,
        "validation_frame_rate": args.validation_frame_rate,
        "min_stored_frames": args.min_stored_frames,
        "train_records": [asdict(record) for record in train_records],
        "val_records": [asdict(record) for record in val_records],
    }


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.resolve(strict=False)
    output_dir = args.output_dir.resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_h5_path = output_dir / args.train_h5_name
    val_h5_path = output_dir / args.val_h5_name
    train_list_path = output_dir / args.train_list_name
    val_list_path = output_dir / args.val_list_name
    val_samples_path = output_dir / args.val_samples_name
    metadata_path = output_dir / args.metadata_name

    outputs = [train_h5_path, val_h5_path, train_list_path, val_list_path, val_samples_path, metadata_path]
    if any(path.exists() for path in outputs) and not args.overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        raise FileExistsError(f"Refusing to overwrite existing outputs without --overwrite: {existing}")

    videos = discover_videos(input_dir, output_dir)
    if not videos:
        raise FileNotFoundError(f"No videos found under {input_dir}")

    train_videos, val_videos = split_videos(videos, args.val_split, args.seed)
    print(f"Found {len(videos)} videos: {len(train_videos)} train / {len(val_videos)} val")

    train_records = build_hdf5_split(
        video_paths=train_videos,
        output_h5_path=train_h5_path,
        sample_every_n=args.sample_every_n,
        max_frames_per_video=args.max_frames_per_video,
        min_stored_frames=args.min_stored_frames,
        stored_data_frame_rate=args.stored_data_frame_rate,
        compression=args.compression,
    )
    val_records = build_hdf5_split(
        video_paths=val_videos,
        output_h5_path=val_h5_path,
        sample_every_n=args.sample_every_n,
        max_frames_per_video=args.max_frames_per_video,
        min_stored_frames=args.min_stored_frames,
        stored_data_frame_rate=args.stored_data_frame_rate,
        compression=args.compression,
    )

    write_hdf5_path_list(train_list_path, [train_h5_path] if train_records else [])
    write_hdf5_path_list(val_list_path, [val_h5_path] if val_records else [])

    val_samples = build_validation_samples(
        val_records,
        num_frames=args.val_num_frames,
        stored_data_frame_rate=args.stored_data_frame_rate,
        validation_frame_rate=args.validation_frame_rate,
        num_samples=args.num_val_samples,
        seed=args.seed,
    )
    val_samples_path.write_text(json.dumps(val_samples, indent=2), encoding="utf-8")

    metadata = build_metadata(
        input_dir=input_dir,
        output_dir=output_dir,
        train_records=train_records,
        val_records=val_records,
        val_samples=val_samples,
        args=args,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote training HDF5 list: {train_list_path}")
    print(f"Wrote validation HDF5 list: {val_list_path}")
    print(f"Wrote validation samples: {val_samples_path}")
    print(f"Wrote metadata summary: {metadata_path}")
    print(f"Stored train videos: {len(train_records)}, validation videos: {len(val_records)}")
    print(f"Stored validation windows: {len(val_samples)}")


if __name__ == "__main__":
    main()