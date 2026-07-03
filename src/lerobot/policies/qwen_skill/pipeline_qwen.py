#!/usr/bin/env python
"""
Training script for the Qwen skill classifier on BehaviorBot 1K.

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

#import argparse
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

# Inference
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# LeRobot imports
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.logging_utils import AverageMeter
from lerobot.utils.random_utils import set_seed
from lerobot.utils.import_utils import register_third_party_plugins
register_third_party_plugins()

from lerobot.common.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)

# PI0 imports
#import lerobot

from modeling_qwen_skill import (
    QwenSkillClassifier,
    QwenSkillClassifierConfig,
)

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

# DA PROVARE
from dataclasses import dataclass 

@dataclass
class SkillTrainConfig(TrainPipelineConfig):
    mode: str = "train_val"
    eval_num_batches: int | None = None
    test_num_batches: int | None = None
    inference_episodes: list[int] | None = None

    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    classifier_hidden_1: int = 1024
    classifier_hidden_2: int = 512
    task_key: str = "task"
    image_keys: tuple[str, ...] = (
        "observation.images.rgb.head",
        "observation.images.rgb.left_wrist",
        "observation.images.rgb.right_wrist",
    )

# ── SKILL VOCABULARY ───────────────────────────────────────────────────────────

SKILL_REGISTRY: dict[int, tuple[int, str]] = {
    # skill_id: (class_index, name)
    1: (0, "move to"),  
    2: (1, "pick up from"),  
    4: (2, "place in"), 
    10: (3, "open door"),  
    3: (4, "place on"),  
    12: (5, "close door"),  
    67: (6, "press"),  
    90: (7, "push to"),  
}

SKILL_ID_TO_CLASS = {sid: cls for sid, (cls, _) in SKILL_REGISTRY.items()}
CLASS_TO_SKILL_NAME = {cls: name for _, (cls, name) in SKILL_REGISTRY.items()}
NUM_SKILL_CLASSES = len(SKILL_REGISTRY)
IGNORE_LABEL = -100

CLASS_NAMES        = [CLASS_TO_SKILL_NAME[i] for i in range(NUM_SKILL_CLASSES)]

# ── HELPERS ───────────────────────────────────────────────────────────────────

def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
 
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


# ── ANNOTATION LOADER ──────────────────────────────────────────────────────────

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


# ── DATASET WRAPPER ────────────────────────────────────────────────────────────

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


# ── SPLIT HELPERS ──────────────────────────────────────────────────────────────

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


# ── METRICS ───────────────────────────────────────────────────────────────────

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    valid_mask = labels != IGNORE_LABEL
    if valid_mask.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    return (preds[valid_mask] == labels[valid_mask]).float().mean().item()


def compute_per_class_accuracy(
    logits_or_preds: torch.Tensor, #logits: torch.Tensor,
    labels: torch.Tensor, 
    already_argmax: bool = False,
) -> dict[str, float]:
    #preds = logits.argmax(dim=-1) # prende classe predetta

    preds = logits_or_preds if already_argmax else logits_or_preds.argmax(dim=-1)
    
    per_class: dict[str, float] = {}

    for c in range(NUM_SKILL_CLASSES):
        mask = labels == c
        if mask.sum() == 0:
            continue
        acc = (preds[mask] == c).float().mean().item()
        name = CLASS_TO_SKILL_NAME.get(c, str(c))
        per_class[name] = acc

    return per_class

# ── CONFUSION MATRIX PLOT ─────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, xticks_rotation=45, cmap="Blues")
    ax.set_title("Confusion Matrix — Skill Classifier", fontsize=14, pad=16)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(f"Confusion matrix saved: {output_path}")

# ── TRAINING STEP ──────────────────────────────────────────────────────────────

def training_step(
    cfg,
    policy: QwenSkillClassifier,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    grad_clip_norm: float,
    device: torch.device,
    #preprocessor, # PREPROCESSOR
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, float]:
    """Single classifier training step."""
    policy.train()
    
    batch = move_batch_to_device(batch, device)

    labels = batch["skill_label"]

    # PREPROCESSOR
    # batch = preprocessor(batch)

    # if cfg.policy.use_state:
    #     # tronca lo stato ai primi 28 valori
    #     batch["observation.state"] = batch["observation.state"][:, :28]

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


# ── VALIDATION ────────────────────────────────────────────────────────────────

@torch.no_grad() # disabilita calcolo dei gradienti
def validate(
    cfg,
    policy: QwenSkillClassifier,
    val_loader: DataLoader,
    device: torch.device,
    #preprocessor, # PREPROCESSOR
    num_batches: int | None = None,
) -> dict[str, float]:
    """Validation loop on val_loader."""
    policy.eval() # mette il modello in eval mode

    loss_meter = AverageMeter("val_loss")
    acc_meter = AverageMeter("val_accuracy")
    all_logits = []
    all_labels = []

    for i, batch in enumerate(val_loader):
        if num_batches is not None and i >= num_batches:
            break

        batch = move_batch_to_device(batch, device)

        labels = batch["skill_label"]
        # batch = preprocessor(batch) # PREPROCESSOR

        # if cfg.policy.use_state:
        #     # tronca lo stato ai primi 28 valori
        #     batch["observation.state"] = batch["observation.state"][:, :28]

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

# ── BUILD ─────────────────────────────────────────────────────────────────────
def _build_policy(cfg, device, features_meta):
    logging.info("Loading Qwen policy...")
    input_features = {
        key: PolicyFeature(type=FeatureType.VISUAL, shape=tuple(feat["shape"]))
        for key, feat in features_meta.items()
        if key.startswith("observation.images")
    }
    logging.info(f"input_features: {list(input_features.keys())}")

    policy_cfg = QwenSkillClassifierConfig(
        num_skill_classes=NUM_SKILL_CLASSES,
        model_name=cfg.model_name,
        image_keys=tuple(input_features.keys()),
        task_key=cfg.task_key,
        classifier_hidden_1=cfg.classifier_hidden_1,
        classifier_hidden_2=cfg.classifier_hidden_2,
        device=str(device),
    )
    policy = QwenSkillClassifier(policy_cfg)
    policy.load()
    return policy.to(device)

def build_everything(cfg, device):
     # ── Dataset ───────────────────────────────────────────────────────────
    logging.info("Loading dataset...")
    dataset_root = Path(cfg.dataset.root).expanduser().resolve()
    annotations_root = dataset_root / "annotations"

    # import json
    info = json.load(open(dataset_root / "meta" / "info.json"))
    num_episodes = info["total_episodes"]

    logging.info(f"Annotations from: {annotations_root}")
    logging.info(f"Total episodes: {num_episodes}")

    train_fraction = getattr(cfg.dataset, "train_fraction", 0.7)
    val_fraction = getattr(cfg.dataset, "val_fraction", 0.15)
    test_fraction = getattr(cfg.dataset, "test_fraction", 0.15)

    train_eps, val_eps, test_eps = create_episode_splits(
        num_episodes=num_episodes,
        seed=cfg.seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )   

    logging.info(f"Train episodes: {sorted(train_eps)}")
    logging.info(f"Val episodes:   {sorted(val_eps)}")
    logging.info(f"Test episodes:  {sorted(test_eps)}")
    
    train_raw = LeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=train_eps,
        revision="main",
        force_cache_sync=False,
    )
    val_raw = LeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=val_eps,
        revision="main",
        force_cache_sync=False,
    )
    test_raw = LeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=test_eps,
        revision="main",
        force_cache_sync=False,
    )

    train_dataset = SkillLabeledDataset(train_raw, annotations_root, ignore_unlabeled=False)
    val_dataset = SkillLabeledDataset(val_raw, annotations_root, ignore_unlabeled=False)
    test_dataset = SkillLabeledDataset(test_raw, annotations_root, ignore_unlabeled=False)
    logging.info(
        f"Samples — train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}"
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
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=True, # False per avere sempre gli stessi batch in ogni test 
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )


    # ── Policy ────────────────────────────────────────────────────────────
    # logging.info("Loading Qwen policy...")

    # input_features: dict = {}
    # features_meta = train_raw.meta.features  # dict chiave → {dtype, shape, ...}
    # for key, feat in features_meta.items():
    #     if key.startswith("observation.images"):
    #         # Le immagini hanno shape (C, H, W)
    #         shape = tuple(feat["shape"])  # es. (3, 480, 640)
    #         input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=shape)
       

    # output_features: dict = {}
    # if "action" in features_meta:
    #     shape = tuple(features_meta["action"]["shape"])
    #     output_features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=shape)

    # logging.info(f"input_features: {list(input_features.keys())}")

    # #base_policy_cfg = QwenSkillClassifierConfig()
    # policy_cfg = QwenSkillClassifierConfig(
    #     num_skill_classes=NUM_SKILL_CLASSES,
    #     model_name=cfg.model_name,
    #     image_keys=tuple(cfg.image_keys),
    #     task_key=cfg.task_key,
    #     classifier_hidden_1=cfg.classifier_hidden_1,
    #     classifier_hidden_2=cfg.classifier_hidden_2,
    #     device=str(device),
    # )

    # # policy = QwenSkillClassifier.from_pretrained(
    # #     cfg.policy.pretrained_path, #cfg.policy.pretrained_model_name_or_path,
    # #     config=policy_cfg,
    # #     strict=False,
    # #     ignore_mismatched_sizes=True,
    # #     torch_dtype=torch.bfloat16,
    # #     #torch_dtype=torch.float16, # pesi caricati in fp16 
    # # ).to(device)
    # policy = QwenSkillClassifier(policy_cfg)
    # policy.load()
    # policy = policy.to(device)

    # preprocessor, postprocessor = make_pi0_pre_post_processors(
    #     config=policy_cfg,
    #     dataset_stats=train_raw.meta.stats,
    # )
    policy = _build_policy(cfg, device, train_raw.meta.features)

    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved() / 1e9
        print(f"GPU dopo caricamento modello: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
        #raise SystemExit("DEBUG STOP")

    # trainable = [(n, p.numel()) for n, p in policy.named_parameters() if p.requires_grad]
    # frozen = [(n, p.numel()) for n, p in policy.named_parameters() if not p.requires_grad]
    # logging.info(f"Trainable parameters ({len(trainable)}):")
    # for name, numel in trainable:
    #     logging.info(f"  {name:60s} {numel:>10,}")
    # logging.info(
    #     f"Total trainable: {sum(n for _, n in trainable):,} | "
    #     f"Total frozen:    {sum(n for _, n in frozen):,}"
    # )

    return (
        policy,
        #preprocessor,
        train_loader,
        val_loader,
        test_loader,
    )

def build_for_inference(cfg, device, episodes=None):
    """
    Versione leggera di build_everything per la sola inference.
    episodes: lista di indici episodi su cui fare inference.
              Se None, usa tutti gli episodi del dataset.
    """
    dataset_root = Path(cfg.dataset.root).expanduser().resolve()
    annotations_root = dataset_root / "annotations"

    info = json.load(open(dataset_root / "meta" / "info.json"))
    num_episodes = info["total_episodes"]

    # Se non specificati, usa tutti gli episodi
    if episodes is None:
        episodes = list(range(num_episodes))

    raw_dataset = LeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=episodes,
        revision="main",
        force_cache_sync=False,
    )
    dataset = SkillLabeledDataset(raw_dataset, annotations_root)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,  # ordine temporale importante per la sequenza
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    # Carica modello
    # ... stesso codice di build_everything per policy e preprocessor ...

    # ── Policy ────────────────────────────────────────────────────────────
    # logging.info("Loading Qwen policy...")

    # input_features: dict = {}
    # features_meta = raw_dataset.meta.features  # dict chiave → {dtype, shape, ...}
    # for key, feat in features_meta.items():
    #     if key.startswith("observation.images"):
    #         # Le immagini hanno shape (C, H, W)
    #         shape = tuple(feat["shape"])  # es. (3, 480, 640)
    #         input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=shape)
    #     # elif cfg.policy.use_state and key == "observation.state":
    #     #     #shape = tuple(feat["shape"])
    #     #     input_features[key] = PolicyFeature(type=FeatureType.STATE, shape=(28,))

    # output_features: dict = {}
    # if "action" in features_meta:
    #     shape = tuple(features_meta["action"]["shape"])
    #     output_features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=shape)

    # logging.info(f"input_features: {list(input_features.keys())}")

    #base_policy_cfg = QwenSkillClassifierConfig()
    # policy_cfg = QwenSkillClassifierConfig(
    #     input_features=input_features,
    #     output_features=output_features,
    #     **{
    #         k: v
    #         for k, v in vars(cfg.policy).items()
    #         if k not in ( #"classifier_mode", "train_expert_only", "num_subskill_classes",
    #                      "input_features", "output_features","max_action_dim") # parametri ignorati se passati nel config
    #         and hasattr(base_policy_cfg, k)
    #     },
    # )
    # policy_cfg = QwenSkillClassifierConfig(
    #     num_skill_classes=NUM_SKILL_CLASSES,
    #     model_name=getattr(cfg.policy, "model_name", "Qwen/Qwen3-VL-8B-Instruct"),
    #     image_keys=tuple(input_features.keys()),  # se vuoi derivarle dal dataset
    #     device=str(device),
    # )

    # policy = QwenSkillClassifier.from_pretrained(
    #     cfg.policy.pretrained_path, #cfg.policy.pretrained_model_name_or_path,
    #     config=policy_cfg,
    #     strict=False,
    #     ignore_mismatched_sizes=True,
    #     torch_dtype=torch.bfloat16,
    #     #torch_dtype=torch.float16, # pesi caricati in fp16 
    # ).to(device)
    # policy = QwenSkillClassifier(policy_cfg)
    # policy.load()
    # policy = policy.to(device)

    # preprocessor, postprocessor = make_pi0_pre_post_processors(
    #     config=policy_cfg,
    #     dataset_stats=raw_dataset.meta.stats, # VERIFICARE - va bene solo se le statistiche di raw sono le stesse usate per il training (stesso dataset)
    # )
    policy = _build_policy(cfg, device, raw_dataset.meta.features)


    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved() / 1e9
        print(f"GPU dopo caricamento modello: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    return (
        policy, 
        #preprocessor, 
        loader
    )

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

# ── TRAIN AND VAL ──────────────────────────────────────────────────────────────
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
            config={ # valori visualizzati nella configurazione wandb
                "num_train_steps": cfg.steps, 
                "batch_size": cfg.batch_size,
                "lr": cfg.optimizer.lr if cfg.optimizer else 2.5e-5, #cfg.optimizer.lr,
                # "lr_gemma_expert": 2.5e-6, # differenziati
                # "lr_classifier": 2.5e-5,
                "weight_decay": getattr(cfg.optimizer, "weight_decay", 1e-4),
                "grad_clip_norm": cfg.optimizer.grad_clip_norm if cfg.optimizer else 10.0, #cfg.training.grad_clip_norm,
                "seed": cfg.seed,
                "device": str(device),
                "dataset_repo_id": cfg.dataset.repo_id,
                "num_skill_classes": NUM_SKILL_CLASSES,
            },
        )
        logging.info(f"W&B run: {wandb.run.name} — {wandb.run.url}")

    #policy,preprocessor,train_loader,val_loader,test_loader=build_everything(cfg,device)
    policy,train_loader,val_loader,test_loader=build_everything(cfg,device)

    # ── Optimizer and scheduler ───────────────────────────────────────────
    trainable_params = [p for p in policy.parameters() if p.requires_grad]

    # LR unico
    optimizer = AdamW(
        trainable_params,
        lr=cfg.optimizer.lr if cfg.optimizer else 2.5e-5, #cfg.optimizer.lr, getattr(cfg.optimizer, "lr", 2.5e-5),
        betas=getattr(cfg.optimizer, "betas", (0.9, 0.95)),
        eps=getattr(cfg.optimizer, "eps", 1e-8),
        weight_decay=getattr(cfg.optimizer, "weight_decay", 1e-4), # Weight decay (originale: 1e-10)
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

    # ── Early stopping ─────────────────────────────────────────────────────
    patience = 5  # ferma dopo 5 validazioni senza miglioramento nella loss
    no_improve_count = 0

    train_iter = iter(train_loader)

    while step < cfg.steps: #cfg.training.num_train_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        t_start = time.perf_counter()

        metrics = training_step(
            cfg,
            policy=policy,
            batch=batch,
            optimizer=optimizer,
            grad_clip_norm=cfg.optimizer.grad_clip_norm if cfg.optimizer else 10.0, #cfg.training.grad_clip_norm,
            device=device,
            #preprocessor=preprocessor, # PREPROCESSOR
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

            # lr_gemma = scheduler.get_last_lr()[0] # LR differenziati
            # lr_cls   = scheduler.get_last_lr()[1]
            logging.info(
                f"Step {step:6d}/{cfg.steps} | "
                f"loss={avg_loss:.4f} | "
                f"acc={avg_acc:.3f} | "
                f"grad={avg_grad:.3f} | "
                f"lr={current_lr:.2e} | "
                # f"lr_gemma={lr_gemma:.2e} | " # LR differenziati
                # f"lr_cls={lr_cls:.2e} | "
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

                        # "train/lr_gemma_expert": optimizer.param_groups[0]["lr"], # LR differenziati
                        # "train/lr_classifier":   optimizer.param_groups[1]["lr"],
                    },
                    step=step,
                )
            loss_meter.reset()
            acc_meter.reset()
            grad_norm_meter.reset()
            step_time_meter.reset()

        # ── Periodic validation ───────────────────────────────────────────
        if mode in ("train_val", "train_val_test") and step % cfg.eval_freq == 0:
            t_val_start = time.perf_counter()

            val_metrics = validate(
                cfg,
                policy=policy,
                val_loader=val_loader,
                device=device,
                #preprocessor=preprocessor,
                num_batches=cfg.eval_num_batches or 500 # getattr(cfg, "eval_num_batches", 50), # None cfg.eval_num_batches or None
            )

            val_time=time.perf_counter()-t_val_start
            val_metrics["val/time_s"] = val_time  # aggiunge il tempo al dizionario

            val_summary = " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
            logging.info(f"Step {step:6d} [VAL] {val_summary}")
            
            # if USE_WANDB: 
            if wandb_enabled: 
                wandb.log(val_metrics, step=step)

            # Salva il miglior checkpoint se la val loss migliora (tutti i pesi del modello)
            if val_metrics["val/loss"] < best_val_loss: # - min_delta:
                # Early stopping
                no_improve_count = 0

                best_val_loss = val_metrics["val/loss"]
                best_path = output_dir / "best_classifier.pt"
                #torch.save(policy.state_dict(), best_path)
                policy.save_checkpoint(best_path)
                logging.info(f"Best model aggiornato (val/loss={best_val_loss:.4f}): {best_path}")
            else: # Early stopping
                no_improve_count += 1
                logging.info(f"No improvement: {no_improve_count}/{patience}")
                if no_improve_count >= patience:
                    logging.info("Early stopping triggered")
                    break   
            
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

            # Cancella il checkpoint precedente
            prev_step = step - cfg.save_freq
            if prev_step > 0:
                prev_ckpt_dir = get_step_checkpoint_dir(
                    output_dir=output_dir,
                    total_steps=cfg.steps,
                    step=prev_step,
                )
                if prev_ckpt_dir.exists():
                    import shutil
                    shutil.rmtree(prev_ckpt_dir)
                    logging.info(f"Deleted old checkpoint: {prev_ckpt_dir}")

    # ── Load best checkpoint (if available) for final evaluation ─────────
    best_ckpt_path = output_dir / "best_classifier.pt"
    if best_ckpt_path.exists():
        logging.info(f"Loading best checkpoint for final evaluation: {best_ckpt_path}")
        # policy.load_state_dict(torch.load(best_ckpt_path, map_location=device))

        # ALTERNATIVA
        # state_dict = torch.load(best_ckpt_path, map_location="cpu") # carica tensori temporaneamente in RAM (alloca memoria)
        # policy.load_state_dict(state_dict) # copia pesi nel modello in GPU
        # del state_dict # elimina riferimento python dal dizionario temporaneo
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        policy.load_state_dict(ckpt["state_dict"], strict=False)
        del ckpt

    # ── Final model saving ────────────────────────────────────────────────
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    #policy.save_pretrained(final_dir)
    policy.save_checkpoint(final_dir/ "qwen_classifier.pt")
    logging.info(f"Final model saved in: {final_dir}")

    # ── Final validation (skipped in 'train' mode) ───────────────────────
    if mode in ("train_val", "train_val_test"):
        logging.info("Running final full validation...")
        final_val = validate(
            cfg, 
            policy, 
            val_loader, 
            device, 
            #preprocessor=preprocessor, 
            num_batches=5000
        )
        logging.info("Final validation results:")
        for k, v in final_val.items():
            logging.info(f"  {k}: {v:.4f}")
        #if USE_WANDB:
        if wandb_enabled: 
            wandb.log({f"final/{k}": v for k, v in final_val.items()})
    else:
        logging.info("Skipping final validation (mode='train').")

    # ── W&B finish ────────────────────────────────────────────────────────
    # if USE_WANDB:
    if wandb_enabled: 
        wandb.finish()
    logging.info("Training complete.")


# ── TEST ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_test(cfg, num_batches: int | None):
    set_seed(cfg.seed)
    device = torch.device(getattr(cfg, "device", "cuda"))
    logging.info(f"Device: {device}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    #policy,preprocessor,train_loader,val_loader,test_loader=build_everything(cfg, device)
    policy,train_loader,val_loader,test_loader=build_everything(cfg,device)

    best_ckpt_path = Path(cfg.output_dir) / "best_classifier.pt"

    logging.info(
        f"Loading checkpoint: {best_ckpt_path}"
    )

    # state_dict = torch.load(
    #     best_ckpt_path,
    #     map_location="cpu",
    # )
    # policy.load_state_dict(state_dict)
    # del state_dict
    ckpt = torch.load(best_ckpt_path, map_location="cpu")
    policy.load_state_dict(ckpt["state_dict"], strict=False)
    del ckpt

    policy.eval()

    # ── Test loop ────────────────────────────────────────────────────────
    all_preds  = []
    all_labels = []
    total_loss = 0.0
    n_valid_total = 0

    for i, batch in enumerate(test_loader):
        if num_batches is not None and i >= num_batches:
            break

        if i % 50 == 0:
            log.info(f"  Batch {i} / {len(test_loader) if num_batches is None else num_batches} ...")

        batch  = move_batch_to_device(batch, device)
        labels = batch["skill_label"]
        #batch  = preprocessor(batch)

        # if cfg.policy.use_state:
        #     batch["observation.state"] = batch["observation.state"][:, :28]

        output         = policy.forward(batch)
        logits, loss, _ = unpack_policy_output(output)
        if loss is None:
            loss = F.cross_entropy(logits, labels, ignore_index=IGNORE_LABEL)

        valid_mask = labels != IGNORE_LABEL
        n_valid    = valid_mask.sum().item()
        if n_valid > 0:
            total_loss    += loss.item() * n_valid
            n_valid_total += n_valid

        preds = logits.argmax(dim=-1)
        # Tieni solo sample con label valida
        all_preds.append(preds[valid_mask].cpu())
        all_labels.append(labels[valid_mask].cpu())

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    avg_loss   = total_loss / max(n_valid_total, 1)
    accuracy   = (all_preds == all_labels).mean()

    log.info(f"\n{'='*50}")
    log.info(f"  Samples:  {len(all_labels)} (labeled)")
    log.info(f"  Loss:     {avg_loss:.4f}")
    log.info(f"  Accuracy: {accuracy:.4f}")
    log.info(f"{'='*50}\n")

    # ── Per-class accuracy ────────────────────────────────────────────────────
    log.info("Per-class accuracy:")
    preds_tensor = torch.from_numpy(all_preds)
    labels_tensor = torch.from_numpy(all_labels)
    per_class = compute_per_class_accuracy(preds_tensor, labels_tensor, already_argmax=True)

    for name, acc in per_class.items():
        log.info(f"  {name:20s}  {acc:.3f}")

    # ── Classification report ─────────────────────────────────────────────────
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(NUM_SKILL_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    log.info(f"\nClassification report:\n{report}")

    report_path = output_dir / f"classification_report.txt"
    report_path.write_text(report)
    log.info(f"Classification report saved: {report_path}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_SKILL_CLASSES)))
    plot_confusion_matrix(cm, CLASS_NAMES, output_dir / f"confusion_matrix.png")

    # Normalizzata (per riga) — utile con classi sbilanciate
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    plot_confusion_matrix(cm_norm, CLASS_NAMES, output_dir / f"confusion_matrix_normalized.png")

    # ── Salva summary JSON ────────────────────────────────────────────────────
    summary = {
        "n_samples": int(len(all_labels)),
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "per_class_accuracy": { # con la funzione compute_per_class_accuracy
            CLASS_TO_SKILL_NAME[c]: per_class.get(CLASS_TO_SKILL_NAME[c])
            for c in range(NUM_SKILL_CLASSES)
        },
    }
    summary_path = output_dir / f"summary_test.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info(f"Summary saved: {summary_path}")

    return summary

# ── INFERENCE ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(cfg, num_batches: int | None = None) -> None:
    """
    Inference su episodi senza label (o con label usate solo per confronto).

    Per ogni episodio produce:
      - sequenza temporale di skill predette frame per frame
      - sequenza compatta (run-length encoding): [(skill, frame_start, frame_end), ...]
      - confronto con le annotazioni manuali se disponibili
      - grafici delle sequenze predette per episodio

    Risultati salvati in output_dir/inference_results/.
    """
    set_seed(cfg.seed)
    device = torch.device(getattr(cfg, "device", "cuda"))
    output_dir = Path(cfg.output_dir) / "inference_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Carica modello ────────────────────────────────────────────────────────
    # INTEGRAER LA FUNZIONE SOPRA build_for_inference
    # _, preprocessor, _, _, test_loader = build_everything(cfg, device) 

    best_ckpt_path = Path(cfg.output_dir) / "best_classifier.pt"
    # if not best_ckpt_path.exists():
    #     # fallback al final/ se best non esiste
    #     best_ckpt_path = Path(cfg.output_dir) / "final"
    #     log.info(f"best_classifier.pt not found, loading from: {best_ckpt_path}")

    log.info(f"Loading checkpoint: {best_ckpt_path}")
    # ???? policy, preprocessor, _, _, test_loader = build_everything(cfg, device)
    policy, loader = build_for_inference(
        cfg,
        device,
        #episodes=[0], # [106, 120, 135] numero dell'episodio passato per inferenza
        episodes=cfg.inference_episodes
    )

    # state_dict = torch.load(best_ckpt_path, map_location="cpu")
    # policy.load_state_dict(state_dict)
    # del state_dict
    ckpt = torch.load(best_ckpt_path, map_location="cpu")
    policy.load_state_dict(ckpt["state_dict"], strict=False)
    del ckpt

    torch.cuda.empty_cache()
    policy.eval()

    # ── Inference loop ────────────────────────────────────────────────────────
    # Accumula predizioni per episodio: {ep_idx: [(frame_idx, pred, conf, true_label)]}
    episode_data: dict[int, list] = {}

    all_preds = [] # cm
    all_labels = []

    for i, batch in enumerate(loader):
        if num_batches is not None and i >= num_batches:
            break

        if i % 100 == 0:
            log.info(f"  Batch {i} / {len(loader) if num_batches is None else num_batches} ...")

        batch         = move_batch_to_device(batch, device)
        true_labels   = batch["skill_label"]           # [B] — IGNORE_LABEL se non annotato
        ep_indices    = batch["episode_index"]          # [B]
        frame_indices = batch.get("frame_index",
                        batch.get("index",
                        torch.zeros(true_labels.shape[0], dtype=torch.long)))

        # batch   = preprocessor(batch)
        # if cfg.policy.use_state:
        #     batch["observation.state"] = batch["observation.state"][:, :28]

        output          = policy.forward(batch)
        logits, _, _    = unpack_policy_output(output)
        preds           = logits.argmax(dim=-1)                         # [B]
        confidences     = logits.softmax(dim=-1).max(dim=-1).values     # [B]

        valid_mask = true_labels != IGNORE_LABEL
        if valid_mask.any():
            all_preds.append(preds[valid_mask].cpu())
            all_labels.append(true_labels[valid_mask].cpu())

        for ep, fr, pred, conf, true in zip(
            ep_indices.cpu().tolist(),
            frame_indices.cpu().tolist(),
            preds.cpu().tolist(),
            confidences.cpu().float().tolist(),
            true_labels.cpu().tolist(),
        ):
            if ep not in episode_data:
                episode_data[ep] = []
            episode_data[ep].append((int(fr), int(pred), float(conf), int(true)))

    # ── Per ogni episodio: ordina per frame e produce output ──────────────────
    all_episode_summaries = []

    for ep_idx in sorted(episode_data.keys()):
        frames = sorted(episode_data[ep_idx], key=lambda x: x[0])  # ordina per frame_idx
        frame_ids   = [f[0] for f in frames]
        preds_seq   = [f[1] for f in frames]
        confs_seq   = [f[2] for f in frames]
        true_seq    = [f[3] for f in frames]

        # ── Run-length encoding (sequenza compatta) ───────────────────────────
        # Raggruppa frame consecutivi con la stessa skill predetta

        #preds_smooth = smooth_predictions(preds_seq, window=100) # POST-PROCESSING
        #rle = _run_length_encode(frame_ids, preds_smooth, confs_seq)
        ##rle = filter_by_confidence(rle, min_confidence=0.5) # POST-PROCESSING

        rle = _run_length_encode(frame_ids, preds_seq, confs_seq) # NO post-processing

        log.info(f"\nEpisodio {ep_idx:04d} — {len(frames)} frame")
        log.info(f"  Sequenza predetta (skill, frame_start, frame_end, conf_media):")
        for skill_cls, f_start, f_end, avg_conf in rle:
            skill_name = CLASS_TO_SKILL_NAME.get(skill_cls, f"class_{skill_cls}")
            log.info(f"    [{f_start:5d} → {f_end:5d}]  {skill_name:20s}  conf={avg_conf:.3f}")

        # ── Confronto con annotazioni manuali se disponibili ──────────────────
        has_labels = any(t != IGNORE_LABEL for t in true_seq)
        if has_labels:
            valid = [(p, t) for p, t in zip(preds_seq, true_seq) if t != IGNORE_LABEL]
            ep_acc = sum(p == t for p, t in valid) / len(valid)
            log.info(f"  Accuracy vs annotazioni manuali: {ep_acc:.3f} ({len(valid)} frame annotati)")
        else:
            ep_acc = None
            log.info(f"  Nessuna annotazione manuale disponibile")

        # ── Salva sequenza predetta per episodio in JSON ──────────────────────
        ep_output = {
            "episode_index": ep_idx,
            "n_frames": len(frames),
            "accuracy_vs_manual": ep_acc,
            "predicted_sequence": [
                {
                    "skill": CLASS_TO_SKILL_NAME.get(skill_cls, f"class_{skill_cls}"),
                    "skill_class": skill_cls,
                    "frame_start": f_start,
                    "frame_end": f_end,
                    "n_frames": f_end - f_start,
                    "avg_confidence": round(avg_conf, 4),
                }
                for skill_cls, f_start, f_end, avg_conf in rle
            ],
        }
        all_episode_summaries.append(ep_output)

        ep_json = output_dir / f"episode_{ep_idx:04d}_inference.json"
        ep_json.write_text(json.dumps(ep_output, indent=2))

        # ── Plot sequenza temporale ───────────────────────────────────────────
        _plot_episode_sequence(
            frame_ids=frame_ids,
            preds_seq=preds_seq,
            true_seq=true_seq if has_labels else None,
            ep_idx=ep_idx,
            output_path=output_dir / f"episode_{ep_idx:04d}_sequence.png",
        )

    # ── Salva summary globale ─────────────────────────────────────────────────
    if len(all_episode_summaries)>1:
        global_summary = {
            "n_episodes": len(all_episode_summaries),
            "episodes": all_episode_summaries,
        }
        summary_path = output_dir / "inference_summary.json"
        summary_path.write_text(json.dumps(global_summary, indent=2))

    
    # ── Confusion matrix globale su frame annotati ─────────────────────────────
    if all_labels:
        all_preds_np = torch.cat(all_preds).numpy()
        all_labels_np = torch.cat(all_labels).numpy()

        # binomial test per classe DA PROVARE
        from scipy.stats import binomtest, chi2_contingency

        per_class_tests = {} #

        for class_idx, class_name in CLASS_TO_SKILL_NAME.items():
            mask = all_labels_np == class_idx
            n_samples = int(mask.sum())

            if n_samples == 0:
                continue

            n_correct = int((all_preds_np[mask] == class_idx).sum())

            recall = n_correct / n_samples #

            p_random = n_samples / len(all_labels_np)

            result = binomtest(
                n_correct,
                n_samples,
                p=p_random,
                alternative="greater",
            )

            per_class_tests[class_name] = { #
                "class_idx": int(class_idx),
                "n_samples": n_samples,
                "n_correct": n_correct,
                "recall": float(recall),
                "p_random": float(p_random),
                "p_value": float(result.pvalue),
                "significant_0_05": bool(result.pvalue < 0.05),
            }
            
            log.info(
                f"{class_name:20s}: recall={n_correct/n_samples:.3f} "
                f"p-value={result.pvalue:.4g}"
            )
        #############
        
        # ── Classification report ─────────────────────────────────────────────────
        report = classification_report(
            all_labels_np, all_preds_np,
            labels=list(range(NUM_SKILL_CLASSES)),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
        log.info(f"\nClassification report:\n{report}")

        report_path = output_dir / f"classification_report_inference.txt"
        report_path.write_text(report)
        log.info(f"Classification report saved: {report_path}")
        #####

        cm = confusion_matrix(
            all_labels_np,
            all_preds_np,
            labels=list(range(NUM_SKILL_CLASSES)),
        )
        plot_confusion_matrix(
            cm,
            CLASS_NAMES,
            output_dir / "inference_confusion_matrix.png",
        )

        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        plot_confusion_matrix(
            cm_norm,
            CLASS_NAMES,
            output_dir / "inference_confusion_matrix_normalized.png",
        )
        
        ###
        #chi2, chi2_p_value, dof, expected = chi2_contingency(cm)

        row_sums = cm.sum(axis=1)
        col_sums = cm.sum(axis=0)

        valid_classes = (row_sums > 0) & (col_sums > 0)
        cm_for_chi2 = cm[valid_classes][:, valid_classes]

        if cm_for_chi2.shape[0] >= 2:
            chi2, chi2_p_value, dof, expected = chi2_contingency(cm_for_chi2)

            log.info(
                f"Chi-square test: chi2={chi2:.2f}, "
                f"p-value={chi2_p_value:.4g}, dof={dof}"
            )
        else:
            chi2 = None
            chi2_p_value = None
            dof = None
            log.warning("Chi-square test skipped: not enough non-empty classes.")

        # log.info(
        #     f"Chi-square test: chi2={chi2:.2f}, "
        #     f"p-value={chi2_p_value:.4g}, dof={dof}"
        # )

        stats_summary = { # print da provare
            # "chi_square_test": {
            #     "chi2": float(chi2),
            #     "p_value": float(chi2_p_value),
            #     "dof": int(dof),
            # },
            "chi_square_test": {
                "chi2": float(chi2) if chi2 is not None else None,
                "p_value": float(chi2_p_value) if chi2_p_value is not None else None,
                "dof": int(dof) if dof is not None else None,
                "used_classes": [
                    CLASS_TO_SKILL_NAME[i]
                    for i, keep in enumerate(valid_classes)
                    if keep
                ],
            },
            "per_class_binomial_tests": per_class_tests, #
        }
        
        ###

        stats_path = output_dir / "inference_statistical_tests.json"
        stats_path.write_text(json.dumps(stats_summary, indent=2))

        log.info("Inference confusion matrix saved.")
    else:
        log.info("No labeled frames found: skipping inference confusion matrix.")

        log.info(f"\nInference completata. Risultati in: {output_dir}")


# ── INFERENCE HELPERS ─────────────────────────────────────────────────────────
def smooth_predictions(preds, window):
    """
    Finestra centrata: guarda window//2 frame prima
    e window//2 frame dopo il frame corrente.
    """
    from collections import Counter
    half = window // 2
    smoothed = []
    for i in range(len(preds)):
        start = max(0, i - half)
        end   = min(len(preds), i + half + 1)
        window_preds = preds[start:end]
        most_common  = Counter(window_preds).most_common(1)[0][0]
        smoothed.append(most_common)
    return smoothed

def _run_length_encode(
    frame_ids: list[int],
    preds: list[int],
    confs: list[float],
) -> list[tuple[int, int, int, float]]:
    """
    Comprime la sequenza frame-by-frame in segmenti.
    Restituisce lista di (skill_class, frame_start, frame_end, avg_confidence).
    """
    if not preds:
        return []

    segments = []
    cur_skill = preds[0]
    cur_start = frame_ids[0]
    cur_confs = [confs[0]]

    for frame, pred, conf in zip(frame_ids[1:], preds[1:], confs[1:]):
        if pred == cur_skill:
            cur_confs.append(conf)
        else:
            segments.append((cur_skill, cur_start, frame - 1, float(np.mean(cur_confs))))
            cur_skill = pred
            cur_start = frame
            cur_confs = [conf]

    segments.append((cur_skill, cur_start, frame_ids[-1], float(np.mean(cur_confs))))
    return segments


def _plot_episode_sequence(
    frame_ids: list[int],
    preds_seq: list[int],
    true_seq: list[int] | None,
    ep_idx: int,
    output_path: Path,
) -> None:
    """
    Produce un grafico a barre colorate che mostra la sequenza di skill
    predette (e opzionalmente quelle reali) per un episodio.

    Esempio visivo:
      Frame: 0─────────────────────────────────────────────────────> N
      Pred:  [move to      ][pick up from][place in  ][move to      ]
      True:  [move to           ][pick up from  ][place in          ]
    """
    # Colore diverso per ogni classe
    colors = plt.get_cmap("tab10", NUM_SKILL_CLASSES)

    n_rows = 2 if true_seq is not None else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(18, 3 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for ax, seq, title in zip(
        axes,
        [preds_seq] + ([true_seq] if true_seq is not None else []),
        ["Predicted"] + (["Ground Truth"] if true_seq is not None else []),
    ):
        prev_skill = None
        seg_start  = frame_ids[0]

        for frame, skill in zip(frame_ids, seq):
            if skill != prev_skill and prev_skill is not None:
                color = colors(prev_skill % NUM_SKILL_CLASSES) if prev_skill != IGNORE_LABEL else "lightgrey"
                ax.barh(
                    0, frame - seg_start, left=seg_start,
                    color=color, edgecolor="white", height=0.6,
                )
                name = CLASS_TO_SKILL_NAME.get(prev_skill, "?") if prev_skill != IGNORE_LABEL else "unlabeled"
                ax.text(
                    seg_start + (frame - seg_start) / 2, 0,
                    name, ha="center", va="center", fontsize=7, color="white", weight="bold",
                )
                seg_start = frame
            prev_skill = skill

        # Ultimo segmento
        if prev_skill is not None:
            color = colors(prev_skill % NUM_SKILL_CLASSES) if prev_skill != IGNORE_LABEL else "lightgrey"
            ax.barh(
                0, frame_ids[-1] - seg_start + 1, left=seg_start,
                color=color, edgecolor="white", height=0.6,
            )
            name = CLASS_TO_SKILL_NAME.get(prev_skill, "?") if prev_skill != IGNORE_LABEL else "unlabeled"
            ax.text(
                seg_start + (frame_ids[-1] - seg_start + 1) / 2, 0,
                name, ha="center", va="center", fontsize=7, color="white", weight="bold",
            )

        ax.set_xlim(frame_ids[0], frame_ids[-1] + 1)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel("Frame")
        ax.set_title(f"Episode {ep_idx:04d} — {title}")

    plt.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Sequence plot saved: {output_path}")

# ── ENTRYPOINT ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    @parser.wrap()
    def main(cfg: SkillTrainConfig):
        mode = cfg.mode
        logging.info(f"Mode: {mode}")

        test_num_batches = cfg.test_num_batches
        if mode == "test":
            run_test(cfg, num_batches=test_num_batches)
        elif mode == "train_val_test":
            train(cfg, mode)
            run_test(cfg, num_batches=test_num_batches)
        elif mode == "inference":
            run_inference(cfg, num_batches=None)
        else:
            train(cfg, mode)
    main()
