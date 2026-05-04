# SPDX-License-Identifier: MIT
# Author: Dr Shashank Pathak
# Email: shashank@computer.org
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

import os
from typing import Iterable

import torch
import torch.nn.functional as F

from models.second_stage.fm_model import ModelIF, requires_grad, update_ema
from util import instantiate_from_config


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


class Stage1RecoveryModelIF(ModelIF):
    def __init__(
        self,
        *,
        student_checkpoint: str,
        teacher_checkpoint: str | None = None,
        teacher_generator_config=None,
        teacher_use_ema: bool = True,
        output_distill_weight: float = 0.0,
        weight_decay: float = 0.01,
        strict_checkpoint_load: bool = True,
        train_space_mlp: bool = True,
        train_time_mlp: bool = True,
        train_adaln: bool = True,
        train_time_attn_modulation: bool = True,
        train_final_layer: bool = True,
        train_t_embedder: bool = True,
        train_frame_emb: bool = True,
        **kwargs,
    ):
        self.student_checkpoint = student_checkpoint
        self.teacher_checkpoint = teacher_checkpoint
        self.teacher_use_ema = teacher_use_ema
        self.output_distill_weight = output_distill_weight
        self.weight_decay = weight_decay
        self.strict_checkpoint_load = strict_checkpoint_load
        self.train_space_mlp = train_space_mlp
        self.train_time_mlp = train_time_mlp
        self.train_adaln = train_adaln
        self.train_time_attn_modulation = train_time_attn_modulation
        self.train_final_layer = train_final_layer
        self.train_t_embedder = train_t_embedder
        self.train_frame_emb = train_frame_emb
        self.teacher_generator_config = teacher_generator_config
        self.teacher_vit = None
        self._trainable_parameter_names: list[str] = []

        super().__init__(**kwargs)

        requires_grad(self.ae, False)
        self._load_student_weights()
        self._configure_stage1_trainability()
        self._load_teacher_weights()

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
        if self.teacher_generator_config is None:
            raise ValueError("teacher_generator_config is required when teacher_checkpoint is set")

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

    def _compute_step(self, batch, stage: str) -> torch.Tensor:
        images, frame_rate = self.get_input(batch, "images")
        x = self.encode_frames(images)

        context = x[:, :-1].clone() if x.size(1) > 1 else None
        target = x[:, -1:]

        t = torch.rand((x.shape[0],), device=x.device)
        target_t, noise = self.add_noise(target, t)

        pred = self.vit(target_t, context, t, frame_rate=frame_rate)
        target_velocity = self.A(t) * target + self.B(t) * noise

        flow_loss_map = (pred.float() - target_velocity.float()) ** 2
        flow_loss = flow_loss_map.mean()
        loss_recon, loss_sem = self._split_latent_loss(flow_loss_map)

        total_loss = flow_loss
        self.log(f"{stage}/loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/loss_recon", loss_recon, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/loss_sem", loss_sem, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)

        if self.teacher_vit is not None and self.output_distill_weight > 0.0:
            with torch.no_grad():
                teacher_pred = self.teacher_vit(target_t, context, t, frame_rate=frame_rate)
            distill_loss = F.mse_loss(pred.float(), teacher_pred.float())
            total_loss = total_loss + self.output_distill_weight * distill_loss
            self.log(f"{stage}/distill_loss", distill_loss, prog_bar=False, logger=True, on_step=True, on_epoch=True, sync_dist=True)
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
            f"Stage1RecoveryModelIF training {trainable_params}/{total_params} vit parameters "
            f"across {len(self._trainable_parameter_names)} tensors"
        )

    def get_trainable_parameter_names(self) -> Iterable[str]:
        return tuple(self._trainable_parameter_names)
