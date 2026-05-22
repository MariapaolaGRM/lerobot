#!/usr/bin/env python
"""
Training script for the PI0 skill classifier on BehaviorBot 1K.

This version keeps the code organized into clear stages:
- dataset split helper
- training step
- validation
- final test
- main training loop

Usage:
    python pipeline_train.py 
        --config_path=/home/mariapaolagerminario/venvs/lerobot/training_config/config.yaml
"""

import os
os.environ["LEROBOT_VIDEO_TIMESTAMP_TOLERANCE_S"] = "0.1"  # 10x la default

#import importlib
#from __future__ import annotations
#from pprint import pformat
#from lerobot.utils.import_utils import register_third_party_plugins

import argparse
import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

# LeRobot imports
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.logging_utils import AverageMeter
from lerobot.utils.random_utils import set_seed

from lerobot.common.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)

# PI0 imports
import lerobot
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Policy

# from lerobot.policies import make_pre_post_processors # PREPROCESSOR
from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors

import lerobot.datasets.video_utils as _vu
_original_decode = _vu.decode_video_frames_torchcodec
def _patched_decode(video_path, timestamps, tolerance_s, **kwargs):
    return _original_decode(video_path, timestamps, tolerance_s=max(tolerance_s, 0.2), **kwargs)
_vu.decode_video_frames_torchcodec = _patched_decode

try:
    import wandb
    USE_WANDB = True
except ImportError:
    USE_WANDB = False
    logging.warning("wandb not installed — logging to console only")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True
)
log = logging.getLogger(__name__)

#DA PROVARE
# from dataclasses import dataclass 

# @dataclass
# class PI0SkillTrainConfig(TrainPipelineConfig):
#     mode: str = "train_val_test"
#     eval_num_batches: int | None = None


# EVENTUALE ALTERNATIVA DI PREPROCESSING
# from transformers import AutoTokenizer
# _TOKENIZER = None
# _TOKENIZER_NAME = "google/paligemma-3b-pt-224"
# _TOKENIZER_MAX_LENGTH = 48

# def get_tokenizer():
#     global _TOKENIZER
#     if _TOKENIZER is None:
#         logging.info(f"Carico tokenizer: {_TOKENIZER_NAME}")
#         _TOKENIZER = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)
#     return _TOKENIZER

# ═════════════════════════════════════════════════════════════════════════════
# SKILL VOCABULARY
# ═════════════════════════════════════════════════════════════════════════════

SKILL_REGISTRY: dict[int, tuple[int, str]] = {
    # skill_id: (class_index, name)
    1: (0, "move to"),  # 1,770,229 frame
    2: (1, "pick up from"),  # 1,121,218 frame
    4: (2, "place in"),  # 582,636 frame
    10: (3, "open door"),  # 238,308 frame
    3: (4, "place on"),  # 154,266 frame
    12: (5, "close door"),  # 129,016 frame
    67: (6, "press"),  # 58,120 frame
    90: (7, "push to"),  # 615 frame
}

SKILL_ID_TO_CLASS = {sid: cls for sid, (cls, _) in SKILL_REGISTRY.items()}
CLASS_TO_SKILL_NAME = {cls: name for _, (cls, name) in SKILL_REGISTRY.items()}
NUM_SKILL_CLASSES = len(SKILL_REGISTRY)
IGNORE_LABEL = -100


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
 
    # PI0 legge sempre i token di linguaggio dal batch prima del check
    # classifier_mode — se non ci sono li aggiungo vuoti (sequenza di zeri).
    # tokenizer_max_length default = 48 (da PI0Config).
    # if "observation.language.tokens" not in batch:
    #     B = next(v.shape[0] for v in batch.values() if isinstance(v, torch.Tensor))
    #     T = 48  # tokenizer_max_length default
    #     batch["observation.language.tokens"] = torch.zeros(
    #         B, T, dtype=torch.long, device=device
    #     )
    #     batch["observation.language.attention_mask"] = torch.zeros(
    #         B, T, dtype=torch.bool, device=device
    #     )
    
    # EVENTUALE ALTERNATIVA DI PREPROCESSING
    # if "observation.language.tokens" not in batch:
    #     tokenizer = get_tokenizer()
    #     tasks = batch.get("task", None)

    #     if tasks is None or (isinstance(tasks, list) and all(t == "" for t in tasks)):
    #         B = next(v.shape[0] for v in batch.values() if isinstance(v, torch.Tensor))
    #         tokens = torch.zeros(B, _TOKENIZER_MAX_LENGTH, dtype=torch.long, device=device)
    #         mask   = torch.zeros(B, _TOKENIZER_MAX_LENGTH, dtype=torch.long, device=device)
    #     else:
    #         if isinstance(tasks, str):
    #             tasks = [tasks]
    #         tasks = [t if t.endswith("\n") else f"{t}\n" for t in tasks]
    #         encoded = tokenizer(
    #             tasks,
    #             max_length=_TOKENIZER_MAX_LENGTH,
    #             padding="max_length",
    #             truncation=True,
    #             return_tensors="pt",
    #         )
    #         tokens = encoded["input_ids"].to(device)
    #         mask   = encoded["attention_mask"].to(device)

    #     batch["observation.language.tokens"]        = tokens
    #     batch["observation.language.attention_mask"] = mask
 
    return batch


def unpack_policy_output(output: Any) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    """
    Make the training code robust to slightly different PI0 forward signatures.
    Expected common formats:
      - (logits, loss, loss_dict)
      - (logits, loss)
      - {'logits': ..., 'loss': ..., ...}
    """
    if isinstance(output, dict):
        logits = output["logits"]
        loss = output.get("loss")
        loss_dict = output.get("loss_dict", {})
        return logits, loss, loss_dict

    if isinstance(output, tuple):
        if len(output) == 3:
            logits, loss, loss_dict = output
            return logits, loss, (loss_dict if isinstance(loss_dict, dict) else {})
        if len(output) == 2:
            logits, loss = output
            return logits, loss, {}
        if len(output) == 1:
            return output[0], None, {}

    raise TypeError(
        f"Unsupported policy forward output type: {type(output)}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# ANNOTATION LOADER
# ═════════════════════════════════════════════════════════════════════════════

def load_skill_annotation(annotation_path: Path) -> np.ndarray | None:
    """Load an episode annotation JSON and return frame-wise skill labels."""
    if not annotation_path.exists():
        return None

    with open(annotation_path) as f:
        ann = json.load(f)

    total_frames = ann["meta_data"]["task_duration"]
    labels = np.full(total_frames, fill_value=IGNORE_LABEL, dtype=np.int64)

    for skill in ann["skill_annotation"]:
        skill_id = skill["skill_id"][0]
        f_start, f_end = skill["frame_duration"]

        if skill_id not in SKILL_ID_TO_CLASS:
            logging.warning(
                f"skill_id {skill_id} not in SKILL_REGISTRY — "
                f"frames {f_start}:{f_end} ignored."
            )
            continue

        labels[f_start:f_end] = SKILL_ID_TO_CLASS[skill_id]

    return labels


# ═════════════════════════════════════════════════════════════════════════════
# DATASET WRAPPER
# ═════════════════════════════════════════════════════════════════════════════

class SkillLabeledDataset(Dataset):
    """
    Wrap LeRobotDataset and add a 'skill_label' field to each sample.

    Current version keeps all samples (ignore_unlabeled is retained for API
    compatibility but is not used for filtering because filtering upfront was
    too expensive and brittle).
    """

    def __init__(
        self,
        lerobot_dataset: LeRobotDataset,
        annotations_root: Path,
        ignore_unlabeled: bool = False,
    ):
        self.dataset = lerobot_dataset
        self.annotations_root = Path(annotations_root)
        self._label_cache: dict[int, np.ndarray | None] = {}

        if ignore_unlabeled:
            logging.warning(
                "ignore_unlabeled=True is currently ignored; all samples are kept "
                "and unlabeled ones are marked with IGNORE_LABEL."
            )

        self._valid_indices = list(range(len(self.dataset)))

    def _get_episode_labels(self, episode_index: int) -> np.ndarray | None:
        """Load and cache an episode's labels from its annotation JSON."""
        if episode_index not in self._label_cache:
            ep_filename = f"episode_{episode_index:08d}.json"
            matches = list(self.annotations_root.rglob(ep_filename))

            if len(matches) == 0:
                logging.warning(
                    f"[ANNOTATIONS] annotation NOT found for episode {episode_index}"
                )
                self._label_cache[episode_index] = None
            else:
                if len(matches) > 1:
                    logging.warning(
                        f"[ANNOTATIONS] found {len(matches)} files for episode {episode_index}; "
                        f"using {matches[0]}"
                    )
                else:
                    logging.debug(
                        f"[ANNOTATIONS] episode {episode_index} -> {matches[0].name}"
                    )
                self._label_cache[episode_index] = load_skill_annotation(matches[0])

        return self._label_cache[episode_index]

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        real_idx = self._valid_indices[idx]
        sample = self.dataset[real_idx]

        ep_idx = int(sample["episode_index"])
        frame_idx = int(sample.get("frame_index", sample.get("index", 0)))
        labels = self._get_episode_labels(ep_idx)

        if labels is None or frame_idx >= len(labels):
            skill_label = IGNORE_LABEL
        else:
            skill_label = int(labels[frame_idx])

        sample["skill_label"] = torch.tensor(skill_label, dtype=torch.long) # aggiunge skill

        if idx == 0:
            print("Debug sample.keys: ", sample.keys()) 

        return sample


# ═════════════════════════════════════════════════════════════════════════════
# SPLIT HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def create_episode_splits(
    num_episodes: int,
    seed: int,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[list[int], list[int], list[int]]:
    """Split episodes into train/val/test at episode level."""
    total = train_fraction + val_fraction + test_fraction
    assert abs(total - 1.0) < 1e-6, f"Fractions must sum to 1.0, got {total}"

    rng = np.random.default_rng(seed)
    all_eps = np.arange(num_episodes)
    rng.shuffle(all_eps)

    if num_episodes <= 0:
        return [], [], []
    if num_episodes == 1:
        return all_eps[:1].tolist(), [], []
    if num_episodes == 2:
        return all_eps[:1].tolist(), all_eps[1:2].tolist(), []

    weights = np.array([train_fraction, val_fraction, test_fraction], dtype=float)
    weights /= weights.sum()

    raw = weights * num_episodes
    counts = np.floor(raw).astype(int)
    remainder = num_episodes - counts.sum()

    # Largest remainder allocation.
    frac_order = np.argsort(raw - counts)[::-1]
    for idx in frac_order[:remainder]:
        counts[idx] += 1

    # Ensure all three splits are non-empty when possible.
    while np.any(counts == 0):
        zero_idx = int(np.where(counts == 0)[0][0])
        donor_idx = int(np.argmax(counts))
        if counts[donor_idx] <= 1:
            break
        counts[donor_idx] -= 1
        counts[zero_idx] += 1

    num_train, num_val, num_test = map(int, counts)
    if num_train + num_val + num_test != num_episodes:
        num_test = num_episodes - num_train - num_val

    train_eps = all_eps[:num_train].tolist()
    val_eps = all_eps[num_train:num_train + num_val].tolist()
    test_eps = all_eps[num_train + num_val:].tolist()

    logging.info(
        f"Split: train={len(train_eps)} val={len(val_eps)} test={len(test_eps)}"
    )
    return train_eps, val_eps, test_eps


# ═════════════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    valid_mask = labels != IGNORE_LABEL
    if valid_mask.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    return (preds[valid_mask] == labels[valid_mask]).float().mean().item()


def compute_per_class_accuracy(
    logits: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    preds = logits.argmax(dim=-1)
    per_class: dict[str, float] = {}

    for c in range(NUM_SKILL_CLASSES):
        mask = labels == c
        if mask.sum() == 0:
            continue
        acc = (preds[mask] == c).float().mean().item()
        name = CLASS_TO_SKILL_NAME.get(c, str(c))
        per_class[name] = acc

    return per_class


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING STEP
# ═════════════════════════════════════════════════════════════════════════════

def training_step(
    policy: PI0Policy,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    grad_clip_norm: float,
    device: torch.device,
    preprocessor, # PREPROCESSOR
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, float]:
    """Single classifier training step."""
    policy.train()
    
    batch = move_batch_to_device(batch, device)

    # Salva skill_label prima che il preprocessor la rimuova
    #labels = batch.pop("skill_label")

    #print("Primo: ",batch)

    labels = batch["skill_label"]

    # PREPROCESSOR
    batch = preprocessor(batch)

    # tronca lo stato ai primi 28 valori
    batch["observation.state"] = batch["observation.state"][:, :28]
    #print("Secondo: ",batch)

    # Reinierisci skill_label nel batch preprocessato
    #batch["skill_label"] = labels
    #print("Terzo: ",batch)

    ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        #torch.autocast(device_type=device.type, dtype=torch.float16)
        if use_amp else nullcontext()
    )

    optimizer.zero_grad(set_to_none=True)

    with ctx:
        output = policy.forward(batch)
        logits, loss, _ = unpack_policy_output(output)
        if loss is None:
            loss = F.cross_entropy(logits, labels, ignore_index=IGNORE_LABEL)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad],
            grad_clip_norm,
        )
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad],
            grad_clip_norm,
        )
        optimizer.step()

    with torch.no_grad():
        acc = compute_accuracy(logits, labels)

    return {
        "loss": loss.item(),
        "accuracy": acc,
        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
    }


# ═════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(
    policy: PI0Policy,
    val_loader: DataLoader,
    device: torch.device,
    preprocessor, # PREPROCESSOR
    num_batches: int | None = None,
) -> dict[str, float]:
    """Validation loop on val_loader."""
    policy.eval()

    loss_meter = AverageMeter("val_loss")
    acc_meter = AverageMeter("val_accuracy")
    all_logits = []
    all_labels = []

    for i, batch in enumerate(val_loader):
        if num_batches is not None and i >= num_batches:
            break

        batch = move_batch_to_device(batch, device)

        labels = batch["skill_label"]
        batch = preprocessor(batch) # PREPROCESSOR

        # tronca lo stato ai primi 28 valori
        batch["observation.state"] = batch["observation.state"][:, :28]

        # Salva skill_label prima che il preprocessor la rimuova
        # labels = batch.pop("skill_label")
        # # PREPROCESSOR
        # batch = preprocessor(batch)
        # # Reinierisci skill_label nel batch preprocessato
        # batch["skill_label"] = labels

        output = policy.forward(batch)
        logits, loss, _ = unpack_policy_output(output)
        if loss is None:
            loss = F.cross_entropy(logits, labels, ignore_index=IGNORE_LABEL)

        acc = compute_accuracy(logits, labels)
        n_valid = max((labels != IGNORE_LABEL).sum().item(), 1)

        loss_meter.update(loss.item(), n=n_valid)
        acc_meter.update(acc, n=n_valid)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    if len(all_logits) == 0:
        return {"val/loss": 0.0, "val/accuracy": 0.0}

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    per_class = compute_per_class_accuracy(all_logits, all_labels)

    return {
        "val/loss": loss_meter.avg,
        "val/accuracy": acc_meter.avg,
        **{f"val/acc_{name}": acc for name, acc in per_class.items()},
    }


# ═════════════════════════════════════════════════════════════════════════════
# FINAL TEST
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def test(
    policy: PI0Policy,
    test_loader: DataLoader,
    device: torch.device,
    preprocessor, # PREPROCESSOR
) -> dict[str, float]:
    """Final evaluation on the test set (use only once, at the end)."""
    logging.info("\n==============================")
    logging.info("Running FINAL TEST evaluation")
    logging.info("==============================")

    metrics = validate(
        policy=policy,
        val_loader=test_loader,
        device=device,
        preprocessor=preprocessor, # PREPROCESSOR
        num_batches=None, # num_batches=cfg.eval_num_batches # DA PROVARE
    ) 

    for k, v in metrics.items():
        logging.info(f"{k}: {v:.4f}")

    return metrics


# ═════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

def train(cfg, mode: str = "train_val") -> None:
    """Main training loop.

    Args:
        cfg:  Training configuration.
        mode: One of 'train', 'train_val', 'train_val_test'.
              Controls whether periodic validation and final test are executed.
    """
    set_seed(cfg.seed)
    device = torch.device(getattr(cfg, "device", "cuda")) #cfg.device)
    logging.info(f"Device: {device}")

    # ── W&B initialisation ────────────────────────────────────────────────
    wandb_enabled = USE_WANDB and cfg.wandb.enable

    if wandb_enabled:
        wandb.init(
            project=cfg.wandb.project, # getattr(cfg, "wandb_project", "pi0-skill-classifier"),
            name=cfg.job_name or f"pi0-skill-classifier-{cfg.steps}-steps", # getattr(cfg, "wandb_run_name", None),
            config={
                "num_train_steps": cfg.steps, 
                "batch_size": cfg.batch_size,
                "lr": cfg.optimizer.lr if cfg.optimizer else 2.5e-5, #cfg.optimizer.lr,
                "grad_clip_norm": cfg.optimizer.grad_clip_norm if cfg.optimizer else 10.0, #cfg.training.grad_clip_norm,
                "seed": cfg.seed,
                "device": str(device),
                "dataset_repo_id": cfg.dataset.repo_id,
                "num_skill_classes": NUM_SKILL_CLASSES,
            },
        )
        logging.info(f"W&B run: {wandb.run.name} — {wandb.run.url}")

    # ── Dataset ───────────────────────────────────────────────────────────
    logging.info("Loading dataset...")
    dataset_root = Path(cfg.dataset.root).expanduser().resolve()
    annotations_root = dataset_root / "annotations"

    # import json
    info = json.load(open(dataset_root / "meta" / "info.json"))
    num_episodes = info["total_episodes"]

    logging.info(f"Annotations from: {annotations_root}")
    logging.info(f"Total episodes: {num_episodes}")
    # full_dataset = LeRobotDataset(
    #     repo_id=cfg.dataset.repo_id,
    #     root=cfg.dataset.root,
    #     episodes=cfg.dataset.episodes,
    #     image_transforms=cfg.dataset.image_transforms,
    #     delta_timestamps=cfg.dataset.delta_timestamps,
    #     video_backend=cfg.dataset.video_backend,
    # )

    # annotations_root = Path(full_dataset.root) / "annotations"
    # logging.info(f"Annotations from: {annotations_root}")
    # num_episodes = full_dataset.num_episodes

    train_fraction = getattr(cfg.dataset, "train_fraction", 0.7)
    val_fraction = getattr(cfg.dataset, "val_fraction", 0.15)
    test_fraction = getattr(cfg.dataset, "test_fraction", 0.15)

    # _test_eps is preserved for future evaluation (run a separate evaluate.py)
    train_eps, val_eps, _test_eps = create_episode_splits(
        num_episodes=num_episodes,
        seed=cfg.seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )

    train_raw = LeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=train_eps,
        #image_transforms=cfg.dataset.image_transforms,
        #delta_timestamps=None, #cfg.dataset.delta_timestamps,
        #video_backend=cfg.dataset.video_backend,

        revision="main",
        force_cache_sync=False,
    )
    val_raw = LeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=val_eps,
        #image_transforms=cfg.dataset.image_transforms,
        #delta_timestamps=None, #cfg.dataset.delta_timestamps,
        #video_backend=cfg.dataset.video_backend,

        revision="main",
        force_cache_sync=False,
    )

    train_dataset = SkillLabeledDataset(train_raw, annotations_root, ignore_unlabeled=False)
    val_dataset = SkillLabeledDataset(val_raw, annotations_root, ignore_unlabeled=False)
    logging.info(
        f"Annotated samples — train: {len(train_dataset)}, val: {len(val_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_dataset) >= cfg.batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    # ── Policy ────────────────────────────────────────────────────────────
    logging.info("Loading PI0 policy...")

    input_features: dict = {}
    features_meta = train_raw.meta.features  # dict chiave → {dtype, shape, ...}
    for key, feat in features_meta.items():
        if key.startswith("observation.images"):
            # Le immagini hanno shape (C, H, W)
            shape = tuple(feat["shape"])  # es. (3, 480, 640)
            input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=shape)
        elif key == "observation.state":
            #shape = tuple(feat["shape"])
            input_features[key] = PolicyFeature(type=FeatureType.STATE, shape=(28,))

    # input_features["task"] = PolicyFeature( # PREPROCESSOR
    #     type=FeatureType.TEXT,
    #     shape=(1,),
    # )

    output_features: dict = {}
    if "action" in features_meta:
        shape = tuple(features_meta["action"]["shape"])
        output_features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=shape)

    logging.info(f"input_features: {list(input_features.keys())}")
    #logging.info(f"output_features: {list(output_features.keys())}")

    state_dim = 32  # default PI0Config
    if "observation.state" in features_meta:
        state_dim = features_meta["observation.state"]["shape"][0]
        #logging.info(f"max_state_dim impostato a {state_dim} (da observation.state)")
 

    base_policy_cfg = PI0Config()
    policy_cfg = PI0Config(
        input_features=input_features,
        output_features=output_features,
        #max_state_dim=state_dim,
        **{
            k: v
            for k, v in vars(cfg.policy).items()
            if k not in ( #"classifier_mode", "train_expert_only", "num_subskill_classes",
                         "input_features", "output_features","max_action_dim") # parametri ignorati se passati nel config
            and hasattr(base_policy_cfg, k)
        },

    )

    policy = PI0Policy.from_pretrained(
        cfg.policy.pretrained_path, #cfg.policy.pretrained_model_name_or_path,
        config=policy_cfg,
        strict=False,
        ignore_mismatched_sizes=True,

        torch_dtype=torch.bfloat16,
        #torch_dtype=torch.float16, # pesi caricati in fp16 
    ).to(device)

    # PREPROCESSOR
    # processor_kwargs = {
    #     "dataset_stats": train_raw.meta.stats,
    # }
    # postprocessor_kwargs = {}
    # preprocessor, postprocessor = make_pre_post_processors(
    #     policy_cfg=policy_cfg,
    #     pretrained_path=cfg.policy.pretrained_path,
    #     **processor_kwargs,
    #     **postprocessor_kwargs,
    # )

    preprocessor, postprocessor = make_pi0_pre_post_processors(
        config=policy_cfg,
        dataset_stats=train_raw.meta.stats,
    )

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved  = torch.cuda.memory_reserved() / 1e9
    print(f"GPU dopo caricamento modello: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    #raise SystemExit("DEBUG STOP")

    trainable = [(n, p.numel()) for n, p in policy.named_parameters() if p.requires_grad]
    frozen = [(n, p.numel()) for n, p in policy.named_parameters() if not p.requires_grad]
    logging.info(f"Trainable parameters ({len(trainable)}):")
    for name, numel in trainable:
        logging.info(f"  {name:60s} {numel:>10,}")
    logging.info(
        f"Total trainable: {sum(n for _, n in trainable):,} | "
        f"Total frozen:    {sum(n for _, n in frozen):,}"
    )

    # ── Optimizer and scheduler ───────────────────────────────────────────
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=cfg.optimizer.lr if cfg.optimizer else 2.5e-5, #cfg.optimizer.lr,
        betas=getattr(cfg.optimizer, "betas", (0.9, 0.95)),
        eps=getattr(cfg.optimizer, "eps", 1e-8),
        weight_decay=getattr(cfg.optimizer, "weight_decay", 1e-10),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.steps, #cfg.training.num_train_steps,
        eta_min=1e-6, #getattr(cfg.optimizer, "lr_min", 1e-6),
    )

    use_amp = device.type == "cuda"

    #scaler = torch.amp.GradScaler("cuda") if use_amp else None # con float16
    scaler = None # con bfloat16
    
    if use_amp:
        logging.info("AMP enabled")

    # ── Resume from checkpoint ────────────────────────────────────────────
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    if cfg.resume:
        step, optimizer, scheduler = load_training_state(
            output_dir, policy, optimizer, scheduler
        )
        logging.info(f"Resumed from step {step}")

    # ── Training loop ─────────────────────────────────────────────────────
    logging.info(f"Starting training - {cfg.steps} total steps")

    loss_meter = AverageMeter("train_loss")
    acc_meter = AverageMeter("train_accuracy")
    grad_norm_meter = AverageMeter("grad_norm")
    step_time_meter = AverageMeter("step_time")

    best_val_loss = float("inf")

    train_iter = iter(train_loader)

    while step < cfg.steps: #cfg.training.num_train_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        t_start = time.perf_counter()

        metrics = training_step(
            policy=policy,
            batch=batch,
            optimizer=optimizer,
            grad_clip_norm=cfg.optimizer.grad_clip_norm if cfg.optimizer else 10.0, #cfg.training.grad_clip_norm,
            device=device,
            preprocessor=preprocessor, # PREPROCESSOR
            use_amp=use_amp,
            scaler=scaler,
        )

        scheduler.step()
        step += 1

        step_time = time.perf_counter() - t_start
        loss_meter.update(metrics["loss"])
        acc_meter.update(metrics["accuracy"])
        grad_norm_meter.update(metrics["grad_norm"])
        step_time_meter.update(step_time)

        # ── Periodic logging ──────────────────────────────────────────────
        if step % cfg.log_freq == 0:
            avg_loss = loss_meter.avg
            avg_acc = acc_meter.avg
            avg_grad = grad_norm_meter.avg
            current_lr = scheduler.get_last_lr()[0]
            logging.info(
                f"Step {step:6d}/{cfg.steps} | "
                f"loss={avg_loss:.4f} | "
                f"acc={avg_acc:.3f} | "
                f"grad={avg_grad:.3f} | "
                f"lr={current_lr:.2e} | "
                f"t={step_time_meter.avg:.3f}s"
            )
            # if USE_WANDB:
            if wandb_enabled:   
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/accuracy": avg_acc,
                        "train/grad_norm": avg_grad,
                        "train/lr": current_lr,
                        "train/step_time": step_time_meter.avg,
                    },
                    step=step,
                )
            loss_meter.reset()
            acc_meter.reset()
            grad_norm_meter.reset()
            step_time_meter.reset()

        # ── Periodic validation ───────────────────────────────────────────
        if mode in ("train_val", "train_val_test") and step % cfg.eval_freq == 0:
            val_metrics = validate(
                policy=policy,
                val_loader=val_loader,
                device=device,
                preprocessor=preprocessor,
                num_batches=None, #getattr(cfg.training, "eval_num_batches", None),
            )
            val_summary = " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            logging.info(f"Step {step:6d} [VAL] {val_summary}")
            
            # if USE_WANDB: 
            if wandb_enabled: 
                wandb.log(val_metrics, step=step)

            # Salva il miglior checkpoint se la val loss migliora
            if val_metrics["val/loss"] < best_val_loss:
                best_val_loss = val_metrics["val/loss"]
                best_path = output_dir / "best_classifier.pt"
                torch.save(policy.state_dict(), best_path)
                logging.info(f"Best model aggiornato (val/loss={best_val_loss:.4f}): {best_path}")
            
            policy.train()

        # ── Checkpoint saving ─────────────────────────────────────────────
        if step % cfg.save_freq == 0:
            ckpt_dir = get_step_checkpoint_dir(
                output_dir=output_dir,
                total_steps=cfg.steps,
                step=step,
            )

            save_checkpoint(
                checkpoint_dir=ckpt_dir,
                step=step,
                cfg=cfg,
                policy=policy,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            update_last_checkpoint(ckpt_dir)
            logging.info(f"Checkpoint saved: {ckpt_dir}")

    # ── Load best checkpoint (if available) for final evaluation ─────────
    best_ckpt_path = output_dir / "best_classifier.pt"
    if best_ckpt_path.exists():
        logging.info(f"Loading best checkpoint for final evaluation: {best_ckpt_path}")
        policy.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    # ── Final model saving ────────────────────────────────────────────────
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(final_dir)
    logging.info(f"Final model saved in: {final_dir}")

    # ── Final validation (skipped in 'train' mode) ───────────────────────
    if mode in ("train_val", "train_val_test"):
        logging.info("Running final full validation...")
        final_val = validate(policy, val_loader, device, preprocessor=preprocessor, num_batches=None)
        logging.info("Final validation results:")
        for k, v in final_val.items():
            logging.info(f"  {k}: {v:.4f}")
        #if USE_WANDB:
        if wandb_enabled: 
            wandb.log({f"final/{k}": v for k, v in final_val.items()})
    else:
        logging.info("Skipping final validation (mode='train').")

    # ── Final test (only in 'train_val_test' mode) ────────────────────────
    if mode == "train_val_test":
        best_ckpt_path = output_dir / "best_classifier.pt"
        if best_ckpt_path.exists():
            logging.info(f"Loading best checkpoint for test: {best_ckpt_path}")
            policy.load_state_dict(torch.load(best_ckpt_path, map_location=device))

        test_raw = LeRobotDataset(
            repo_id=cfg.dataset.repo_id,
            root=cfg.dataset.root,
            episodes=_test_eps,
            #image_transforms=cfg.dataset.image_transforms,
            #delta_timestamps=None, #cfg.dataset.delta_timestamps,
            #video_backend=cfg.dataset.video_backend,

            revision="main",
            force_cache_sync=False,
        )
        test_dataset = SkillLabeledDataset(test_raw, annotations_root, ignore_unlabeled=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
        )
        final_test = test(policy, test_loader, device, preprocessor=preprocessor)
        logging.info("Final TEST results:")
        for k, v in final_test.items():
            logging.info(f"  {k}: {v:.4f}")
        # if USE_WANDB:
        if wandb_enabled: 
            wandb.log({f"test/{k}": v for k, v in final_test.items()})
    else:
        logging.info(
            f"Skipping final test (mode='{mode}'). "
            "Re-run with --mode train_val_test to evaluate on the test set."
        )

    # ── W&B finish ────────────────────────────────────────────────────────
    # if USE_WANDB:
    if wandb_enabled: 
        wandb.finish()
    logging.info("Training complete.")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═════════════════════════════════════════════════════════════════════════════
# if __name__ == "__main__":
#     # ── Mode selector (must be stripped before lerobot parser sees argv) ──
#     mode_parser = argparse.ArgumentParser(add_help=False)
#     mode_parser.add_argument(
#         "--mode",
#         choices=["train", "train_val", "train_val_test"],
#         default="train_val",
#     )
#     mode_args, remaining_argv = mode_parser.parse_known_args()

#     import sys
#     sys.argv = [sys.argv[0]] + remaining_argv

#     cfg = parser.parse_args_into_dataclasses(TrainPipelineConfig)[0]
#     logging.info(f"Config:\n{pformat(vars(cfg))}")
#     logging.info(f"Mode: {mode_args.mode}")
#     train(cfg, mode=mode_args.mode)
#     main()

# if __name__ == "__main__":
#     import sys

#     mode_parser = argparse.ArgumentParser(add_help=False)
#     mode_parser.add_argument(
#         "--mode",
#         choices=["train", "train_val", "train_val_test"],
#         default="train_val",
#     )

#     mode_args, remaining_argv = mode_parser.parse_known_args()

#     sys.argv = [sys.argv[0]] + remaining_argv

#     @parser.wrap()
#     def main(cfg: TrainPipelineConfig):

#         logging.info(f"Config:\n{pformat(vars(cfg))}")
#         logging.info(f"Mode: {mode_args.mode}")

#         train(cfg, mode=mode_args.mode)

#     main()


if __name__ == "__main__":
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()   # carica i plugin PRIMA del parser

    MODE = "train_val_test"  # cambia qui: "train", "train_val", "train_val_test"
    @parser.wrap()
    def main(cfg: TrainPipelineConfig):
        logging.info(f"Mode: {MODE}")
        train(cfg, mode=MODE)

    # DA PROVARE
    # @parser.wrap()
    # def main(cfg: PI0SkillTrainConfig):
    #     logging.info(f"Mode: {cfg.mode}")
    #     train(cfg, mode=cfg.mode)

    main()
