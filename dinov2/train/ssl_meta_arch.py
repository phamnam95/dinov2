# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging

import math
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from dinov2.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from dinov2.models import build_model_from_cfg
from dinov2.layers import DINOHead, apply_lora_to_vit
from dinov2.utils.utils import has_batchnorms
from dinov2.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from dinov2.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model
import dinov2.distributed as distributed

from dinov2.models.vision_transformer import BlockChunk


try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


logger = logging.getLogger("dinov2")


class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_model_dict = dict()
        teacher_model_dict = dict()

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        # Optionally apply LoRA adapters to ViT
        if getattr(cfg.student, "lora", None) is not None and cfg.student.lora.enabled:
            lora_cfg = cfg.student.lora
            student_backbone = apply_lora_to_vit(
                student_backbone,
                target_substrings=tuple(lora_cfg.get("targets", ["qkv", "proj", "fc1", "fc2"])),
                r=int(lora_cfg.get("r", 8)),
                alpha=float(lora_cfg.get("alpha", 16.0)),
                dropout=float(lora_cfg.get("dropout", 0.0)),
            )
            logger.info("Applied LoRA adapters to student backbone")

        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights, map_location="cpu")
            logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}")
            state = chkpt.get("model", chkpt)
            # If 3D backbone but checkpoint is 2D, adapt weights
            if getattr(student_backbone, "is_3d", False):
                state = self._adapt_2d_backbone_state_for_3d(student_backbone, state)
            missing, unexpected = student_backbone.load_state_dict(state, strict=False)
            if missing:
                logger.info(f"pretrained load - missing keys: {len(missing)}")
            if unexpected:
                logger.info(f"pretrained load - unexpected keys: {len(unexpected)}")

        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head

        logger.info("OPTIONS -- DINO")
        if self.do_dino:
            logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
            logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
            logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
            logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
            self.dino_loss_weight = cfg.dino.loss_weight
            dino_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
            self.dino_loss = DINOLoss(self.dino_out_dim)
            if self.do_koleo:
                logger.info("OPTIONS -- DINO -- applying KOLEO regularization")
                self.koleo_loss = KoLeoLoss()

        else:
            logger.info("OPTIONS -- DINO -- not using DINO")

        if self.do_dino or self.do_ibot:
            student_model_dict["dino_head"] = dino_head()
            teacher_model_dict["dino_head"] = dino_head()

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")
        if self.do_ibot:
            self.ibot_loss_weight = cfg.ibot.loss_weight
            assert max(cfg.ibot.mask_ratio_min_max) > 0, "please provide a positive mask ratio tuple for ibot"
            assert cfg.ibot.mask_sample_probability > 0, "please provide a positive mask probability for ibot"
            self.ibot_out_dim = cfg.ibot.head_n_prototypes if self.ibot_separate_head else cfg.dino.head_n_prototypes
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            if self.ibot_separate_head:
                logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
                logger.info(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
                logger.info(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
                logger.info(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
                ibot_head = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=cfg.ibot.head_n_prototypes,
                    hidden_dim=cfg.ibot.head_hidden_dim,
                    bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                    nlayers=cfg.ibot.head_nlayers,
                )
                student_model_dict["ibot_head"] = ibot_head()
                teacher_model_dict["ibot_head"] = ibot_head()
            else:
                logger.info("OPTIONS -- IBOT -- head shared with DINO")

        self.need_to_synchronize_fsdp_streams = True

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        logger.info(f"Student and Teacher are built: they are both {cfg.student.arch} network.")

        # If backbone uses model-parallel, place heads on last device and disable FSDP stream sync
        if getattr(self.student["backbone"], "model_parallel", False):
            # Require single process per node when using intra-node model-parallel
            if distributed.get_local_size() != 1:
                raise RuntimeError(
                    "Model-parallel backbone requires one process per node (LOCAL_WORLD_SIZE=1). "
                    "Launch with torchrun --nproc_per_node=1 and set student.mp_devices to GPUs per node."
                )
            last_device = self.student["backbone"].mp_devices[-1]
            if "dino_head" in self.student:
                self.student["dino_head"].to(last_device)
                self.teacher["dino_head"].to(last_device)
            if "ibot_head" in self.student:
                self.student["ibot_head"].to(last_device)
                self.teacher["ibot_head"].to(last_device)
            self.need_to_synchronize_fsdp_streams = False
            # Broadcast initial weights across nodes when distributed is enabled
            if distributed.get_global_size() > 1:
                self._broadcast_model_parameters(self.student)
                self._broadcast_model_parameters(self.teacher)

        # Configure optional frequency-domain masking for iBOT inputs
        fm_cfg = getattr(cfg.ibot, "freq_mask", None)
        self.freq_mask_enabled = bool(getattr(fm_cfg, "enabled", False)) if fm_cfg is not None else False
        self.freq_mask_prob = float(getattr(fm_cfg, "prob", 0.0)) if fm_cfg is not None else 0.0
        self.freq_mask_low = float(getattr(fm_cfg, "low", 0.0)) if fm_cfg is not None else 0.0
        self.freq_mask_high = float(getattr(fm_cfg, "high", 0.0)) if fm_cfg is not None else 0.0
        self.freq_mask_mode = str(getattr(fm_cfg, "mode", "bandpass")) if fm_cfg is not None else "bandpass"
        self.freq_mask_per_sample = bool(getattr(fm_cfg, "per_sample", False)) if fm_cfg is not None else False
        self.freq_mask_low_range = tuple(getattr(fm_cfg, "low_range", (self.freq_mask_low, self.freq_mask_low)))
        self.freq_mask_high_range = tuple(getattr(fm_cfg, "high_range", (self.freq_mask_high, self.freq_mask_high)))

    @staticmethod
    def _adapt_2d_backbone_state_for_3d(model: nn.Module, state: dict) -> dict:
        """Inflate 2D ViT weights to 3D for PatchEmbed and positional embeddings.

        - patch_embed.proj.weight: [C_out, C_in, Kh, Kw] -> [C_out, C_in, Kd, Kh, Kw] by repeat/avg
        - pos_embed: interpolate 2D grid to target 3D grid (D',H',W') using trilinear with depth=1 source
        Other parameters are copied as-is.
        """
        new_state = dict(state)
        # Adapt patch embedding conv
        pe_w_key = "patch_embed.proj.weight"
        if pe_w_key in state and state[pe_w_key].ndim == 4:
            w2d = state[pe_w_key]  # [E, Cin, Kh, Kw]
            kd = getattr(getattr(model, "patch_embed", None), "proj", None).weight.shape[2]
            # Inflate by repeating along depth and average
            w3d = w2d.unsqueeze(2).repeat(1, 1, kd, 1, 1) / kd
            new_state[pe_w_key] = w3d

        # Adapt positional embedding
        pos_key = "pos_embed"
        if pos_key in state:
            pos2d = state[pos_key]  # [1, N2+1, C]
            if pos2d.ndim == 3 and pos2d.shape[0] == 1:
                cls_pos = pos2d[:, :1, :]
                patch_pos = pos2d[:, 1:, :]
                n2 = patch_pos.shape[1]
                c = patch_pos.shape[2]
                m = int(math.sqrt(n2))
                if m * m == n2:
                    target_res = getattr(getattr(model, "patch_embed", None), "patches_resolution", None)
                    if isinstance(target_res, tuple) and len(target_res) == 3:
                        d0, h0, w0 = target_res
                        patch_pos_2d = patch_pos.view(1, m, m, c).permute(0, 3, 1, 2)  # [1,C,M,M]
                        patch_pos_3d = F.interpolate(
                            patch_pos_2d.unsqueeze(2),  # [1,C,1,M,M]
                            size=(d0, h0, w0),
                            mode="trilinear",
                            align_corners=False,
                        ).squeeze(2)
                        patch_pos_flat = patch_pos_3d.permute(0, 2, 3, 1).reshape(1, d0 * h0 * w0, c)
                        new_state[pos_key] = torch.cat([cls_pos, patch_pos_flat], dim=1)
        return new_state

    @staticmethod
    def _broadcast_model_parameters(module_dict: nn.ModuleDict):
        if not dist.is_available() or not dist.is_initialized():
            return
        # Use device0 per process to perform NCCL broadcasts to remain compatible with single-device NCCL groups
        device0 = torch.device("cuda:0")
        for m in module_dict.values():
            for p in m.state_dict().values():
                if isinstance(p, torch.Tensor):
                    tmp = p.to(device0, non_blocking=True)
                    dist.broadcast(tmp, src=0)
                    if tmp.data_ptr() != p.data_ptr():
                        p.copy_(tmp.to(p.device, non_blocking=True))

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward_backward(self, images, teacher_temp):
        n_global_crops = 2
        assert n_global_crops == 2
        n_local_crops = self.cfg.crops.local_crops_number

        # Choose device for inputs: first MP device if model-parallel, else current cuda device
        if getattr(self.student["backbone"], "model_parallel", False):
            device0 = self.student["backbone"].mp_devices[0]
        else:
            device0 = torch.device("cuda")

        global_crops = images["collated_global_crops"].to(device0, non_blocking=True)
        local_crops = images["collated_local_crops"].to(device0, non_blocking=True)

        # Optional frequency-domain masking on inputs before teacher/student forward
        if self.freq_mask_enabled and self.freq_mask_prob > 0.0:
            if not self.freq_mask_per_sample:
                if torch.rand((), device=global_crops.device).item() < self.freq_mask_prob:
                    global_crops = self._apply_frequency_mask(global_crops, self.freq_mask_low, self.freq_mask_high, self.freq_mask_mode)
                    if local_crops.numel() > 0:
                        local_crops = self._apply_frequency_mask(local_crops, self.freq_mask_low, self.freq_mask_high, self.freq_mask_mode)
            else:
                # Sample-wise random cutoffs within ranges; bandpass by default
                def _sample_cutoffs(n: int, device: torch.device):
                    lo_min, lo_max = self.freq_mask_low_range
                    hi_min, hi_max = self.freq_mask_high_range
                    lows = torch.empty(n, device=device).uniform_(lo_min, lo_max)
                    highs = torch.empty(n, device=device).uniform_(hi_min, hi_max)
                    highs = torch.maximum(highs, lows + 1e-4)
                    return lows, highs

                # Apply to each sample independently
                def _apply_per_sample(x: torch.Tensor):
                    B = x.shape[0]
                    lows, highs = _sample_cutoffs(B, x.device)
                    out_list = []
                    for i in range(B):
                        out_list.append(self._apply_frequency_mask(x[i:i+1], float(lows[i].item()), float(highs[i].item()), self.freq_mask_mode))
                    return torch.cat(out_list, dim=0)

                if torch.rand((), device=global_crops.device).item() < self.freq_mask_prob:
                    global_crops = _apply_per_sample(global_crops)
                    if local_crops.numel() > 0:
                        local_crops = _apply_per_sample(local_crops)

        masks = images["collated_masks"].to(device0, non_blocking=True)
        mask_indices_list = images["mask_indices_list"].to(device0, non_blocking=True)
        n_masked_patches_tensor = images["n_masked_patches"].to(device0, non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"].to(device0, non_blocking=True)

        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        do_dino = self.do_dino
        do_ibot = self.do_ibot

        # loss scales
        ibot_loss_scale = 1.0 / n_global_crops

        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            x, n_global_crops_teacher = global_crops, n_global_crops
            teacher_backbone_output_dict = self.teacher.backbone(x, is_training=True)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
            # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
            teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))
            ibot_teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
            _dim = ibot_teacher_patch_tokens.shape[-1]
            n_cls_tokens = teacher_cls_tokens.shape[0]

            if do_ibot and not self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, _dim)
                buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[n_cls_tokens : n_cls_tokens + n_masked_patches],
                )
                tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
                masked_teacher_patch_tokens_after_head = tokens_after_head[
                    n_cls_tokens : n_cls_tokens + n_masked_patches
                ]
            elif do_ibot and self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[:n_masked_patches],
                )
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)[
                    :n_masked_patches
                ]
            else:
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_ibot_softmaxed_centered = None

            if self.cfg.train.centering == "centering":
                teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
                if do_ibot:
                    masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(
                        masked_teacher_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=teacher_temp
                    )
                    masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)
                    self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head[:n_masked_patches])

            elif self.cfg.train.centering == "sinkhorn_knopp":
                teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])

                if do_ibot:
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                        masked_teacher_patch_tokens_after_head,
                        teacher_temp=teacher_temp,
                        n_masked_patches_tensor=n_masked_patches_tensor,
                    )

            else:
                raise NotImplementedError

            return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered

        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = get_teacher_output()
        reshard_fsdp_model(self.teacher)

        loss_dict = {}

        loss_accumulator = 0  # for backprop
        student_global_backbone_output_dict, student_local_backbone_output_dict = self.student.backbone(
            [global_crops, local_crops], masks=[masks, None], is_training=True
        )

        inputs_for_student_head_list = []

        # 1a: local crops cls tokens
        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

        # 1b: global crops cls tokens
        student_global_cls_tokens = student_global_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

        # 1c: global crops patch tokens
        if do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
            )
            if not self.ibot_separate_head:
                inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(buffer_tensor_patch_tokens)[
                    :n_masked_patches
                ]

        # 2: run
        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))

        # 3a: local crops cls tokens
        student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3b: global crops cls tokens
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3c: global crops patch tokens
        if do_ibot and not self.ibot_separate_head:
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(0)[:n_masked_patches]

        if n_local_crops > 0:
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            # store for display
            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss

        # process global crops
        loss_scales = 2  # this is here since we process global crops together

        if do_dino:
            # compute loss
            dino_global_crops_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list.flatten(0, 1)
                    ],  # these were chunked and stacked in reverse so A is matched to B
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_global_crops_loss

            student_cls_tokens = student_global_cls_tokens

            if self.do_koleo:
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
                )  # we don't apply koleo loss between cls tokens of a same image
                loss_accumulator += koleo_loss
                loss_dict["koleo_loss"] = (
                    koleo_loss / loss_scales
                )  # this is to display the same losses as before but we can remove eventually

        if do_ibot:
            # compute loss
            ibot_patch_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,
                    masked_teacher_ibot_softmaxed_centered,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )
                * loss_scales
                * ibot_loss_scale
            )

            # store for display
            loss_dict["ibot_loss"] = ibot_patch_loss / 2

            # accumulate loss
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        self.backprop_loss(loss_accumulator)

        # Average gradients across nodes when using model-parallel without FSDP/DDP
        if getattr(self.student["backbone"], "model_parallel", False) and distributed.get_global_size() > 1:
            world_size = distributed.get_global_size()
            device0 = torch.device("cuda:0")
            for sub in self.student.values():
                for p in sub.parameters():
                    if p.grad is None:
                        continue
                    tmp = p.grad.detach().to(device0, non_blocking=True)
                    dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
                    tmp.div_(world_size)
                    p.grad.copy_(tmp.to(p.grad.device, non_blocking=True))

        self.fsdp_synchronize_streams()

        return loss_dict

    @staticmethod
    def _build_frequency_mask(shape_spatial: torch.Size, device: torch.device, low: float, high: float, mode: str):
        # shape_spatial: (..., H, W) or (..., D, H, W) but we only need last 2/3
        if len(shape_spatial) == 2:
            H, W = shape_spatial
            fy = torch.fft.fftfreq(H, device=device)
            fx = torch.fft.fftfreq(W, device=device)
            grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
            r = torch.sqrt(grid_x**2 + grid_y**2)
        elif len(shape_spatial) == 3:
            D, H, W = shape_spatial
            fz = torch.fft.fftfreq(D, device=device)
            fy = torch.fft.fftfreq(H, device=device)
            fx = torch.fft.fftfreq(W, device=device)
            grid_z, grid_y, grid_x = torch.meshgrid(fz, fy, fx, indexing="ij")
            r = torch.sqrt(grid_x**2 + grid_y**2 + grid_z**2)
        else:
            raise ValueError("Unsupported spatial rank for frequency mask")

        if mode == "bandpass":
            mask = (r >= low) & (r <= high)
        elif mode == "lowpass":
            mask = (r <= high)
        elif mode == "highpass":
            mask = (r >= low)
        else:
            raise ValueError(f"Unknown freq mask mode: {mode}")
        return mask

    def _apply_frequency_mask(self, x: torch.Tensor, low: float, high: float, mode: str) -> torch.Tensor:
        # x: [N,C,H,W] or [N,C,D,H,W]
        original_dtype = x.dtype
        if x.dim() == 4:
            N, C, H, W = x.shape
            spatial = (H, W)
            dims = (-2, -1)
        elif x.dim() == 5:
            N, C, D, H, W = x.shape
            spatial = (D, H, W)
            dims = (-3, -2, -1)
        else:
            return x

        mask = self._build_frequency_mask(spatial, x.device, low, high, mode)
        # Broadcast mask to [1,1,...spatial]
        while mask.dim() < x.dim():
            mask = mask.unsqueeze(0)
        mask = mask.to(x.device)

        X = torch.fft.fftn(x.float(), dim=dims)
        X = X * mask
        y = torch.fft.ifftn(X, dim=dims).real.to(original_dtype)
        return y

    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            self.student.dino_head._streams = (
                self.teacher.dino_head._streams
            ) = self.student.backbone._streams = self.teacher.backbone._streams
            self.need_to_synchronize_fsdp_streams = False

    def update_teacher(self, m):
        with torch.no_grad():
            if any(get_fsdp_modules(self.student[k]) for k in self.student.keys()):
                student_param_list = []
                teacher_param_list = []
                for k in self.student.keys():
                    for ms, mt in zip(get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])):
                        student_param_list += ms.params
                        teacher_param_list += mt.params
                torch._foreach_mul_(teacher_param_list, m)
                torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)
            else:
                # Fallback when not using FSDP (e.g., model-parallel backbone)
                for k in self.student.keys():
                    for p_s, p_t in zip(self.student[k].parameters(), self.teacher[k].parameters()):
                        p_t.mul_(m).add_(p_s, alpha=1 - m)

    def train(self):
        super().train()
        self.teacher.eval()

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        # If model-parallel is enabled on backbone, skip FSDP wrapping because
        # FSDP cannot shard parameters spanning multiple devices in a single module.
        if getattr(self.student["backbone"], "model_parallel", False):
            logger.info("Model-parallel backbone detected; skipping FSDP wrapping.")
            # Still sync teacher weights from student
            for k in self.student.keys():
                self.teacher[k].load_state_dict(self.student[k].state_dict())
            return

        if has_batchnorms(self.student):
            raise NotImplementedError
        # below will synchronize all student subnetworks across gpus:
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])
            teacher_model_cfg = self.cfg.compute_precision.teacher[k]
            self.teacher[k] = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap={BlockChunk})(self.teacher[k])
