#!/usr/bin/env python
# coding: utf-8
"""
Qwen3-VL skill classifier — fully frozen backbone.

Design
──────
Qwen3VLForConditionalGeneration  (fully frozen)
  images + task text  →  ViT  →  LLM layers  →  hidden_states[-1]
                                                  (B, T, 3584)
                                                       ↓
                                               AttentionPooling     (trainable)
                                                  (B, 3584)
                                                       ↓
                                               classifier_head      (trainable)
                                     LayerNorm → 1024 → 512 → num_skill_classes

Batch keys expected
───────────────────
  "task"                              : list[str] or str, one per sample
  "observation.images.rgb.head"       : (B, C, H, W) float32 [0,1]
  "observation.images.rgb.left_wrist" : (B, C, H, W) float32 [0,1]
  "observation.images.rgb.right_wrist": (B, C, H, W) float32 [0,1]
  "skill_label"                       : (B,) int64   (optional at inference)

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _tensor_to_pil(img: torch.Tensor) -> Image.Image:
    """
    Convert a single image tensor to PIL.

    Expected formats:
      - [C, H, W]
      - [H, W, C]

    Values are assumed in [0, 1]. If your tensors are already uint8, they are
    handled too.
    """
    x = img.detach().cpu()
    if x.ndim != 3:
        raise ValueError(f"Expected 3D image tensor, got {tuple(x.shape)}")
    if x.shape[0] in (1, 3, 4):          # channel-first → channel-last
        x = x.permute(1, 2, 0)
    if x.dtype != torch.uint8:
        x = x.clamp(0.0, 1.0)
        x = (x * 255).to(torch.uint8)
    arr = x.numpy()
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    return Image.fromarray(arr)


# ──────────────────────────────────────────────────────────────────────────────
# Attention pooling  
# ──────────────────────────────────────────────────────────────────────────────

class AttentionPooling(nn.Module):
    """
    Learned query that attends over a token sequence → single vector.

    x        : (B, T, D)
    pad_mask : (B, T) bool, True = real token
    returns  : (B, D)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.scale  = dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q      = self.query.expand(x.shape[0], -1, -1)           # (B, 1, D)
        scores = torch.bmm(q, x.transpose(1, 2)) * self.scale    # (B, 1, T)
        if pad_mask is not None:
            scores = scores.masked_fill(
                ~pad_mask.unsqueeze(1), torch.finfo(scores.dtype).min
            )
        weights = torch.softmax(scores, dim=-1)                   # (B, 1, T)
        return torch.bmm(weights, x).squeeze(1)                   # (B, D)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QwenSkillClassifierConfig:
    """
    All hyper-parameters for QwenSkillClassifier.

    model_name        : HuggingFace ID of Qwen3-VL (or Qwen2.5-VL).
    num_skill_classes : Number of robot skill classes.
    llm_hidden_size   : Hidden dim of the LLM part. Qwen3-VL-8B = 3584.
                        Set to None to infer automatically from model config.
    dtype             : Backbone dtype. bfloat16 recommended on Ampere+.
    device            : Device string passed to device_map.
    task_key          : Batch key for the task description string.
    label_key         : Batch key for integer skill labels.
    image_keys        : Batch keys for the camera images, in order.
    classifier_hidden_1 / _2 : MLP hidden sizes (mirrors pi0 classifier_head).
    """
    model_name:           str   = "Qwen/Qwen3-VL-8B-Instruct"
    num_skill_classes:    int   = 8
    llm_hidden_size:      Optional[int] = None   # inferred if None
    dtype:                torch.dtype = torch.bfloat16
    device:               str   = "cuda"
    task_key:             str   = "task"
    label_key:            str   = "skill_label"
    image_keys:           Tuple[str, ...] = (
        "observation.images.rgb.head",
        "observation.images.rgb.left_wrist",
        "observation.images.rgb.right_wrist",
    )
    classifier_hidden_1:  int   = 1024
    classifier_hidden_2:  int   = 512


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class QwenSkillClassifier(nn.Module):
    """
    Frozen Qwen3-VL + trainable AttentionPooling + trainable classifier head.

    Quick start
    -----------
        cfg   = QwenSkillClassifierConfig(num_skill_classes=8)
        model = QwenSkillClassifier(cfg)
        model.load()                        # downloads weights, freezes backbone

        logits, loss, loss_dict = model(batch)   # training
        preds = model.predict(batch)             # inference
    """

    def __init__(self, cfg: QwenSkillClassifierConfig):
        super().__init__()
        self.cfg = cfg

        # backbone — loaded lazily in load()
        self.qwen      = None
        self.processor = None
        self._hidden_size: Optional[int] = cfg.llm_hidden_size

        # trainable modules — built after load() once hidden size is known
        self.attn_pool:       Optional[AttentionPooling] = None
        self.classifier_head: Optional[nn.Sequential]   = None

        self._loaded = False

    # ── loading ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load processor and backbone, freeze all backbone parameters,
        then build the trainable head.  Call once before training.
        """
        from transformers import AutoProcessor

        name = self.cfg.model_name
        if "Qwen2.5" in name or "qwen2.5" in name.lower():
            from transformers import Qwen2_5_VLForConditionalGeneration as QwenCls
        else:
            from transformers import Qwen3VLForConditionalGeneration as QwenCls

        log.info(f"Loading processor: {name}")
        self.processor = AutoProcessor.from_pretrained(name)

        log.info(f"Loading backbone: {name}")
        self.qwen = QwenCls.from_pretrained(
            name,
            torch_dtype=self.cfg.dtype,
            device_map=None,   # single-device; move with .to(device) after
        )
        self.qwen.eval()
        self._freeze_backbone()

        self.qwen.lm_head = torch.nn.Identity() # Sostituisce la testa di generazione testuale di Qwen con un modulo che non fa nulla.

        # Infer LLM hidden size if not set in config
        if self._hidden_size is None:
            self._hidden_size = self._infer_hidden_size()
        if self._hidden_size is None:
            raise RuntimeError(
                "Could not infer LLM hidden size from model config. "
                "Set llm_hidden_size explicitly in QwenSkillClassifierConfig."
            )

        # Build trainable modules now that hidden size is known
        self.attn_pool = AttentionPooling(dim=self._hidden_size)
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(self._hidden_size),
            nn.Linear(self._hidden_size, self.cfg.classifier_hidden_1),
            nn.ReLU(),
            nn.Linear(self.cfg.classifier_hidden_1, self.cfg.classifier_hidden_2),
            nn.ReLU(),
            nn.Linear(self.cfg.classifier_hidden_2, self.cfg.num_skill_classes),
        )

        self._loaded = True
        self._log_parameters()

    def _freeze_backbone(self) -> None:
        for p in self.qwen.parameters():
            p.requires_grad_(False)

    def _infer_hidden_size(self) -> Optional[int]:
        """Try common config attribute paths used by Qwen VL variants."""
        cfg = self.qwen.config
        candidates = [
            getattr(cfg, "hidden_size", None),
            getattr(getattr(cfg, "text_config", None), "hidden_size", None),
            getattr(getattr(cfg, "model_config", None), "hidden_size", None),
        ]
        for c in candidates:
            if isinstance(c, int) and c > 0:
                return c
        return None

    def _log_parameters(self) -> None:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        log.info(f"Trainable parameters : {trainable:,}")
        log.info(f"Frozen parameters    : {frozen:,}")

    # ── train() override ──────────────────────────────────────────────────────

    def train(self, mode: bool = True):
        """Keep the backbone in eval mode even when the wrapper is training."""
        super().train(mode)
        if self.qwen is not None:
            self.qwen.eval()
        return self

    # ── input preparation ─────────────────────────────────────────────────────

    def _build_messages(self, task: str, n_cameras: int) -> list:
        """
        Build Qwen chat-template messages with n_cameras image slots + task text.
        """
        content = [{"type": "image"} for _ in range(n_cameras)]
        content.append({
            "type": "text",
            "text": f"Task: {task}\nPredict the robot subskill from the images.",
        })
        return [{"role": "user", "content": content}]

    def _extract_tasks(self, batch: Dict[str, Any]) -> List[str]:
        tasks = batch[self.cfg.task_key]
        if isinstance(tasks, (list, tuple)):
            return [str(t) for t in tasks]
        return [str(tasks)]

    def _extract_images(self, batch: Dict[str, Any]) -> List[List[Image.Image]]:
        """
        Returns a list of length B, each element a list of PIL images
        (one per camera), in the order defined by cfg.image_keys.
        """
        # Infer batch size from the first image key
        first = batch[self.cfg.image_keys[0]]
        if not isinstance(first, torch.Tensor):
            raise TypeError(
                f"Expected tensor for key {self.cfg.image_keys[0]}, got {type(first)}"
            )
        batch_size = first.shape[0] if first.ndim == 4 else 1

        per_sample: List[List[Image.Image]] = []
        for b in range(batch_size):
            imgs = []
            for key in self.cfg.image_keys:
                t = batch[key]
                if t.ndim == 4:
                    imgs.append(_tensor_to_pil(t[b]))
                else:
                    imgs.append(_tensor_to_pil(t))
            per_sample.append(imgs)
        return per_sample

    def _make_inputs(
        self, batch: Dict[str, Any], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Build processor inputs (pixel_values, input_ids, attention_mask, …)
        from a batch dict and move them to device.
        """
        tasks             = self._extract_tasks(batch)
        images_per_sample = self._extract_images(batch)
        n_cameras         = len(self.cfg.image_keys)

        texts = [
            self.processor.apply_chat_template(
                self._build_messages(task, n_cameras),
                tokenize=False,
                add_generation_prompt=False,
            )
            for task in tasks
        ]

        encoded = self.processor(
            text=texts,
            images=images_per_sample,
            return_tensors="pt",
            padding=True,
        )
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in encoded.items()
        }

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """
        Training and inference forward pass.

        Returns
        -------
        logits    : (B, num_skill_classes)
        loss      : scalar CrossEntropy loss, or None if no labels in batch
        loss_dict : {"loss": float} or {}
        """
        if not self._loaded:
            raise RuntimeError("Call .load() before forward().")

        device = next(self.attn_pool.parameters()).device
        inputs = self._make_inputs(batch, device=device)

        # ── 1. frozen Qwen forward ────────────────────────────────────────────
        # We run the full model (ViT + LLM) but with no_grad on the backbone.
        # requires_grad=False on all backbone params already prevents gradient
        # computation there; torch.no_grad() additionally saves activation memory.
        with torch.no_grad():
            outputs = self.qwen(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
                #use_cache=False,
            )

        if outputs.hidden_states is None:
            raise RuntimeError(
                "hidden_states is None — il modello non ha restituito gli stati nascosti. "
                "Verifica che output_hidden_states=True sia supportato da questa versione del modello."
            )


        # hidden_states[-1]: (B, T, llm_hidden_size)
        hidden = outputs.hidden_states[-1].float()

        # ── 2. attention pooling (trainable) ──────────────────────────────────
        # Compress the full token sequence → one vector per sample.
        # Pass the attention mask so padding tokens are ignored.
        pad_mask = inputs.get("attention_mask", None)
        if pad_mask is not None:
            pad_mask = pad_mask.bool()

        pooled = self.attn_pool(hidden, pad_mask=pad_mask)    # (B, llm_hidden_size)

        # ── 3. classifier head (trainable) ────────────────────────────────────
        logits = self.classifier_head(pooled)                  # (B, num_skill_classes)

        # ── 4. loss ───────────────────────────────────────────────────────────
        labels = batch.get(self.cfg.label_key, None)
        if labels is None:
            return logits, None, {}

        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        else:
            labels = labels.to(device).long()

        loss = F.cross_entropy(logits, labels, ignore_index=-100)
        return logits, loss, {"loss": loss.item()}

    # ── inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Returns predicted class indices (B,).
        """
        self.eval()
        logits, _, _ = self.forward(batch)
        return logits.argmax(dim=-1)

    # ── trainable parameters ──────────────────────────────────────────────────

    def trainable_parameters(self) -> List[nn.Parameter]:
        """
        Parameters to pass to the optimiser.
        The backbone is never included.
        """
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    # ── checkpoint ────────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str | Path) -> None:
        """
        Save trainable weights + config.
        The backbone is NOT saved (always reloaded from HuggingFace).

        Checkpoint format
        -----------------
        {
            "config":     dict,          # QwenSkillClassifierConfig fields
            "state_dict": OrderedDict,   # attn_pool + classifier_head only
        }
        """
        trainable_keys = {
            n for n, p in self.named_parameters() if p.requires_grad
        }
        ckpt = {
            "config":     self.cfg.__dict__,
            "state_dict": {
                k: v for k, v in self.state_dict().items()
                if k in trainable_keys
            },
        }
        torch.save(ckpt, path)
        log.info(f"Checkpoint saved → {path}")

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        load_backbone: bool = True,
        device: str = "cpu",
    ) -> "QwenSkillClassifier":
        """
        Restore a model from a checkpoint produced by save_checkpoint().

        Parameters
        ----------
        load_backbone : If True, also download and load the Qwen backbone.
                        Set False to inspect head weights without a GPU.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)

        cfg_dict = {**ckpt["config"], "device": device}
        # torch.dtype is not JSON-serialisable — restore it if needed
        if isinstance(cfg_dict.get("dtype"), str):
            cfg_dict["dtype"] = getattr(torch, cfg_dict["dtype"])

        cfg   = QwenSkillClassifierConfig(**cfg_dict)
        model = cls(cfg)

        if load_backbone:
            model.load()   # builds attn_pool + classifier_head

        missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
        if missing:
            log.warning(f"Missing keys in checkpoint: {missing}")
        if unexpected:
            log.warning(f"Unexpected keys in checkpoint: {unexpected}")

        return model.to(device)