# SPDX-License-Identifier: MIT
# Author: Dr Shashank Pathak
# Email: shashank@computer.org
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from contextlib import contextmanager
from typing import Iterable

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from models.second_stage.fm_model import ModelIF, requires_grad, update_ema
from util import instantiate_from_config


BYTES_PER_GIB = float(1024 ** 3)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDENT_RUN_DIR = REPO_ROOT / "logs_wm" / "orbis_288x512_pruned"
DEFAULT_TEACHER_RUN_DIR = REPO_ROOT / "logs_wm" / "orbis_288x512"
DEFAULT_TOKENIZER_DIR = REPO_ROOT / "logs_tk" / "tokenizer_288x512"
DEFAULT_CHECKPOINT_NAME = "checkpoints/last.ckpt"


@dataclass(frozen=True)
class MemoryEstimate:
    batch_size: int
    static_model_gib: float
    optimizer_and_grad_gib: float
    student_activation_gib: float
    teacher_runtime_gib: float
    feature_buffer_gib: float
    estimated_total_gib: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _resolve_path(path: str | os.PathLike[str] | None) -> Path | None:
    if path is None:
        return None
    return Path(os.path.expandvars(os.fspath(path))).expanduser().resolve(strict=False)


def _resolve_run_dir(run_dir: str | os.PathLike[str] | None, default_run_dir: Path) -> Path:
    return _resolve_path(run_dir) or default_run_dir


def _resolve_checkpoint_path(
    checkpoint_path: str | os.PathLike[str] | None,
    run_dir: Path,
    default_name: str = DEFAULT_CHECKPOINT_NAME,
) -> Path:
    resolved = _resolve_path(checkpoint_path)
    return resolved or (run_dir / default_name)


def _load_run_config(run_dir: Path):
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Run config not found: {config_path}")
    return OmegaConf.load(config_path)


def _clone_config(config):
    return OmegaConf.create(OmegaConf.to_container(config, resolve=False))


def _resolve_model_configs(
    *,
    student_run_dir: Path,
    teacher_run_dir: Path,
    tokenizer_config,
    generator_config,
    teacher_generator_config,
):
    student_run_config = _load_run_config(student_run_dir)
    teacher_run_config = _load_run_config(teacher_run_dir)

    resolved_tokenizer_config = _clone_config(
        tokenizer_config or student_run_config.model.params.tokenizer_config
    )
    resolved_generator_config = _clone_config(
        generator_config or student_run_config.model.params.generator_config
    )
    resolved_teacher_generator_config = _clone_config(
        teacher_generator_config or teacher_run_config.model.params.generator_config
    )

    tokenizer_folder = _resolve_path(resolved_tokenizer_config.folder)
    if tokenizer_folder is None:
        tokenizer_folder = DEFAULT_TOKENIZER_DIR
    resolved_tokenizer_config.folder = str(tokenizer_folder)
    if not getattr(resolved_tokenizer_config, "ckpt_path", None):
        resolved_tokenizer_config.ckpt_path = "checkpoints/tokenizer_288x512.ckpt"

    return resolved_tokenizer_config, resolved_generator_config, resolved_teacher_generator_config


def _resolve_feature_indices(num_blocks: int, feature_distill_blocks, feature_distill_stride: int) -> list[int]:
    if feature_distill_blocks is not None:
        return sorted({int(index) for index in feature_distill_blocks if 0 <= int(index) < num_blocks})
    if feature_distill_stride <= 0:
        return []
    indices = list(range(feature_distill_stride - 1, num_blocks, feature_distill_stride))
    if (num_blocks - 1) not in indices:
        indices.append(num_blocks - 1)
    return sorted(set(indices))


@contextmanager
def _temporarily_disable_context_augmentation(*models: torch.nn.Module):
    saved_values = []
    for model in models:
        if model is None:
            continue
        saved_values.append(
            (
                model,
                getattr(model, "drop_ctx_rate", None),
                getattr(model, "ctx_noise_aug_prob", None),
            )
        )
        if hasattr(model, "drop_ctx_rate"):
            model.drop_ctx_rate = 0.0
        if hasattr(model, "ctx_noise_aug_prob"):
            model.ctx_noise_aug_prob = 0.0
    try:
        yield
    finally:
        for model, drop_ctx_rate, ctx_noise_aug_prob in saved_values:
            if drop_ctx_rate is not None:
                model.drop_ctx_rate = drop_ctx_rate
            if ctx_noise_aug_prob is not None:
                model.ctx_noise_aug_prob = ctx_noise_aug_prob


def _load_checkpoint_state(checkpoint_path: str) -> dict[str, torch.Tensor]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint.get("state_dict", checkpoint)


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


class TeacherStudentRecoveryModelIF(ModelIF):
    def __init__(
        self,
        *,
        student_run_dir: str | None = None,
        teacher_run_dir: str | None = None,
        student_checkpoint: str | None = None,
        teacher_checkpoint: str | None = None,
        tokenizer_config=None,
        generator_config=None,
        teacher_generator_config=None,
        teacher_use_ema: bool = True,
        output_distill_weight: float = 0.0,
        feature_distill_weight: float = 0.0,
        feature_distill_blocks: list[int] | None = None,
        feature_distill_stride: int = 4,
        weight_decay: float = 0.01,
        strict_checkpoint_load: bool = True,
        train_space_mlp: bool = True,
        train_time_mlp: bool = True,
        train_adaln: bool = True,
        train_time_attn_modulation: bool = True,
        train_final_layer: bool = True,
        train_t_embedder: bool = True,
        train_frame_emb: bool = True,
        activation_checkpoint_multiplier: float = 1.0,
        **kwargs,
    ):
        self.student_run_dir = _resolve_run_dir(student_run_dir, DEFAULT_STUDENT_RUN_DIR)
        self.teacher_run_dir = _resolve_run_dir(teacher_run_dir, DEFAULT_TEACHER_RUN_DIR)
        self.student_checkpoint = str(_resolve_checkpoint_path(student_checkpoint, self.student_run_dir))
        self.teacher_checkpoint = (
            str(_resolve_checkpoint_path(teacher_checkpoint, self.teacher_run_dir)) if teacher_checkpoint is not False else None
        )
        self.teacher_use_ema = teacher_use_ema
        self.output_distill_weight = output_distill_weight
        self.feature_distill_weight = feature_distill_weight
        self.feature_distill_stride = feature_distill_stride
        self.weight_decay = weight_decay
        self.strict_checkpoint_load = strict_checkpoint_load
        self.train_space_mlp = train_space_mlp
        self.train_time_mlp = train_time_mlp
        self.train_adaln = train_adaln
        self.train_time_attn_modulation = train_time_attn_modulation
        self.train_final_layer = train_final_layer
        self.train_t_embedder = train_t_embedder
        self.train_frame_emb = train_frame_emb
        self.activation_checkpoint_multiplier = activation_checkpoint_multiplier
        self.teacher_vit = None
        self._trainable_parameter_names: list[str] = []
        self._feature_distill_blocks: list[int] = []

        (
            resolved_tokenizer_config,
            resolved_generator_config,
            resolved_teacher_generator_config,
        ) = _resolve_model_configs(
            student_run_dir=self.student_run_dir,
            teacher_run_dir=self.teacher_run_dir,
            tokenizer_config=tokenizer_config,
            generator_config=generator_config,
            teacher_generator_config=teacher_generator_config,
        )
        self.teacher_generator_config = resolved_teacher_generator_config
        kwargs["tokenizer_config"] = resolved_tokenizer_config
        kwargs["generator_config"] = resolved_generator_config

        super().__init__(**kwargs)

        requires_grad(self.ae, False)
        self._load_student_weights()
        self._configure_stage1_trainability()
        self._load_teacher_weights()
        self._feature_distill_blocks = _resolve_feature_indices(
            num_blocks=len(self.vit.blocks),
            feature_distill_blocks=feature_distill_blocks,
            feature_distill_stride=self.feature_distill_stride,
        )

    def _load_student_weights(self) -> None:
        state_dict = _load_checkpoint_state(self.student_checkpoint)

        vit_state = _strip_prefix(state_dict, "vit.")
        if not vit_state:
            raise KeyError(f"Checkpoint does not contain vit weights: {self.student_checkpoint}")
        self.vit.load_state_dict(vit_state, strict=self.strict_checkpoint_load)

        ema_state = _strip_prefix(state_dict, "ema_vit.")
        if ema_state:
            self.ema_vit.load_state_dict(ema_state, strict=self.strict_checkpoint_load)
        else:
            update_ema(self.ema_vit, self.vit, decay=0.0)
        self.ema_vit.eval()

    def _load_teacher_weights(self) -> None:
        if not self.teacher_checkpoint:
            return

        self.teacher_vit = instantiate_from_config(self.teacher_generator_config)
        state_dict = _load_checkpoint_state(self.teacher_checkpoint)
        prefix = "ema_vit." if self.teacher_use_ema and any(
            key.startswith("ema_vit.") for key in state_dict
        ) else "vit."
        teacher_state = _strip_prefix(state_dict, prefix)
        if not teacher_state:
            raise KeyError(f"Checkpoint does not contain {prefix} weights: {self.teacher_checkpoint}")

        self.teacher_vit.load_state_dict(teacher_state, strict=self.strict_checkpoint_load)
        requires_grad(self.teacher_vit, False)
        self.teacher_vit.eval()

    def _set_trainable(self, module: torch.nn.Module | None) -> None:
        if module is not None:
            requires_grad(module, True)

    def _configure_stage1_trainability(self) -> None:
        requires_grad(self.vit, False)

        for block in self.vit.blocks:
            if self.train_space_mlp:
                self._set_trainable(getattr(block, "space_mlp", None))
            if self.train_time_mlp:
                self._set_trainable(getattr(block, "time_mlp", None))
            if self.train_adaln:
                self._set_trainable(getattr(block, "adaLN_modulation", None))
            if self.train_time_attn_modulation:
                self._set_trainable(getattr(block, "adaLN_time_attn_modulation", None))

        if self.train_final_layer:
            self._set_trainable(getattr(self.vit, "final_layer", None))
        if self.train_t_embedder:
            self._set_trainable(getattr(self.vit, "t_embedder", None))
        if self.train_frame_emb and hasattr(self.vit, "frame_emb"):
            self.vit.frame_emb.requires_grad = True

        self._trainable_parameter_names = [
            name for name, parameter in self.vit.named_parameters() if parameter.requires_grad
        ]
        if not self._trainable_parameter_names:
            raise ValueError("Stage 1 recovery did not enable any trainable parameters")

    def _split_latent_loss(self, loss: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if loss.ndim < 3 or loss.size(2) < 2:
            mean_loss = loss.mean()
            return mean_loss, mean_loss
        loss_recon, loss_sem = torch.split(loss, loss.size(2) // 2, dim=2)
        return loss_recon.mean(), loss_sem.mean()

    def _compute_feature_distill_loss(
        self,
        student_features: list[torch.Tensor],
        teacher_features: list[torch.Tensor],
    ) -> torch.Tensor:
        losses = [
            F.mse_loss(student_features[index].float(), teacher_features[index].float())
            for index in self._feature_distill_blocks
        ]
        return torch.stack(losses).mean() if losses else torch.zeros((), device=student_features[-1].device)

    def estimate_memory_requirements(self, batch_size: int) -> MemoryEstimate:
        frames = int(getattr(self.vit, "max_num_frames", 6))
        height, width = self.vit.input_size
        tokens_per_sample = frames * int(height) * int(width)
        hidden_size = int(self.vit.frame_emb.shape[-1])
        depth = len(self.vit.blocks)
        mlp_ratio = float(getattr(self.vit.blocks[0].space_mlp.fc1, "out_features", hidden_size) / hidden_size)
        trainable_params = sum(parameter.numel() for parameter in self.vit.parameters() if parameter.requires_grad)
        student_params = sum(parameter.numel() for parameter in self.vit.parameters())
        ema_params = sum(parameter.numel() for parameter in self.ema_vit.parameters())
        teacher_params = 0 if self.teacher_vit is None else sum(parameter.numel() for parameter in self.teacher_vit.parameters())
        tokenizer_params = sum(parameter.numel() for parameter in self.ae.parameters())

        static_model_gib = 4.0 * (student_params + ema_params + teacher_params + tokenizer_params) / BYTES_PER_GIB
        optimizer_and_grad_gib = 12.0 * trainable_params / BYTES_PER_GIB

        activation_bytes_per_token = 2.0
        per_block_multiplier = 10.0 + (2.0 * mlp_ratio)
        student_activation_gib = (
            batch_size
            * tokens_per_sample
            * hidden_size
            * depth
            * per_block_multiplier
            * activation_bytes_per_token
            * self.activation_checkpoint_multiplier
        ) / BYTES_PER_GIB

        teacher_runtime_gib = 0.0
        if self.teacher_vit is not None:
            teacher_runtime_gib = (
                batch_size
                * tokens_per_sample
                * hidden_size
                * (4.0 + mlp_ratio)
                * activation_bytes_per_token
            ) / BYTES_PER_GIB

        feature_buffer_gib = 0.0
        if self.feature_distill_weight > 0.0 and self.teacher_vit is not None and self._feature_distill_blocks:
            feature_buffer_gib = (
                batch_size
                * tokens_per_sample
                * hidden_size
                * len(self._feature_distill_blocks)
                * 2.0
                * activation_bytes_per_token
            ) / BYTES_PER_GIB

        estimated_total_gib = (
            static_model_gib
            + optimizer_and_grad_gib
            + student_activation_gib
            + teacher_runtime_gib
            + feature_buffer_gib
        )
        return MemoryEstimate(
            batch_size=batch_size,
            static_model_gib=static_model_gib,
            optimizer_and_grad_gib=optimizer_and_grad_gib,
            student_activation_gib=student_activation_gib,
            teacher_runtime_gib=teacher_runtime_gib,
            feature_buffer_gib=feature_buffer_gib,
            estimated_total_gib=estimated_total_gib,
        )

    def _compute_step(self, batch, stage: str) -> torch.Tensor:
        images, frame_rate = self.get_input(batch, "images")
        x = self.encode_frames(images)

        context = x[:, :-1].clone() if x.size(1) > 1 else None
        target = x[:, -1:]

        t = torch.rand((x.shape[0],), device=x.device)
        target_t, noise = self.add_noise(target, t)

        need_teacher_outputs = self.teacher_vit is not None and (
            self.output_distill_weight > 0.0 or self.feature_distill_weight > 0.0
        )
        need_features = self.feature_distill_weight > 0.0 and need_teacher_outputs and bool(self._feature_distill_blocks)

        with _temporarily_disable_context_augmentation(self.vit if need_teacher_outputs else None, self.teacher_vit if need_teacher_outputs else None):
            if need_features:
                pred, student_features = self.vit(target_t, context, t, frame_rate=frame_rate, return_features=True)
            else:
                pred = self.vit(target_t, context, t, frame_rate=frame_rate)
                student_features = None
        target_velocity = self.A(t) * target + self.B(t) * noise

        flow_loss_map = (pred.float() - target_velocity.float()) ** 2
        flow_loss = flow_loss_map.mean()
        loss_recon, loss_sem = self._split_latent_loss(flow_loss_map)

        total_loss = flow_loss
        self.log(f"{stage}/loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/loss_recon", loss_recon, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/loss_sem", loss_sem, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        if need_teacher_outputs:
            with _temporarily_disable_context_augmentation(self.vit, self.teacher_vit):
                with torch.no_grad():
                    if need_features:
                        teacher_pred, teacher_features = self.teacher_vit(
                            target_t,
                            context,
                            t,
                            frame_rate=frame_rate,
                            return_features=True,
                        )
                    else:
                        teacher_pred = self.teacher_vit(target_t, context, t, frame_rate=frame_rate)
                        teacher_features = None

            if self.output_distill_weight > 0.0:
                distill_loss = F.mse_loss(pred.float(), teacher_pred.float())
                total_loss = total_loss + self.output_distill_weight * distill_loss
                self.log(f"{stage}/distill_loss", distill_loss, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)

            if need_features and student_features is not None and teacher_features is not None:
                feature_distill_loss = self._compute_feature_distill_loss(student_features, teacher_features)
                total_loss = total_loss + self.feature_distill_weight * feature_distill_loss
                self.log(
                    f"{stage}/feature_distill_loss",
                    feature_distill_loss,
                    prog_bar=False,
                    logger=True,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                )
            self.log(f"{stage}/loss_total", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._compute_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._compute_step(batch, stage="val")

    def configure_optimizers(self):
        trainable_parameters = [parameter for parameter in self.vit.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable_parameters, lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = self.get_warmup_scheduler(optimizer, self.warmup_steps, self.min_lr_multiplier)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.teacher_vit is not None:
            self.teacher_vit.eval()
        self.ema_vit.eval()
        return self

    def on_train_start(self) -> None:
        total_params = sum(parameter.numel() for parameter in self.vit.parameters())
        trainable_params = sum(parameter.numel() for parameter in self.vit.parameters() if parameter.requires_grad)
        self.print(
            f"TeacherStudentRecoveryModelIF training {trainable_params}/{total_params} vit parameters "
            f"across {len(self._trainable_parameter_names)} tensors"
        )
        if self.teacher_vit is not None:
            self.print(
                f"Teacher distillation active: output_weight={self.output_distill_weight}, "
                f"feature_weight={self.feature_distill_weight}, feature_blocks={self._feature_distill_blocks}"
            )
        for batch_size in (1, 2, 4):
            estimate = self.estimate_memory_requirements(batch_size)
            self.print(f"Estimated GPU memory for batch_size={batch_size}: {estimate.to_dict()}")

    def get_trainable_parameter_names(self) -> Iterable[str]:
        return tuple(self._trainable_parameter_names)
