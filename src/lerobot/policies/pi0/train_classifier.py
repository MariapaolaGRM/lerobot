#!/usr/bin/env python
"""
Training script for the PI0 skill classifier on BehaviorBot 1K.

Usage:
    python train_skill_classifier.py \
        --policy.pretrained_model_name_or_path physical-intelligence/pi0 \
        --dataset.repo_id behavior-1k/2025-challenge-demos \
        --output_dir outputs/skill_classifier \
        --training.num_train_steps 10000 \
        --batch_size 16
"""

import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
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
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.logging_utils import AverageMeter
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)

# PI0 imports
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.pi0.configuration_pi0 import PI0Config

 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ═════════════════════════════════════════════════════════════════════════════
# SKILL VOCABULARY
# ═════════════════════════════════════════════════════════════════════════════

SKILL_REGISTRY: dict[int, tuple[int, str]] = {
    # skill_id: (class_index, name)
    1:  (0, "navigate to"),
    2:  (1, "pick up from"),
    3:  (2, "place on"),
    4:  (3, "place in"),
    5:  (4, "open"),
    6:  (5, "close"),
    7:  (6, "toggle on"),
    8:  (7, "toggle off"),
    90: (8, "push to"),
    # add more skills after running analyze_skill.py
}

SKILL_ID_TO_CLASS   = {sid: cls  for sid, (cls, _)   in SKILL_REGISTRY.items()}
CLASS_TO_SKILL_NAME = {cls: name for _, (cls, name)  in SKILL_REGISTRY.items()}
NUM_SKILL_CLASSES   = len(SKILL_REGISTRY)
IGNORE_LABEL        = -100  # special value automatically ignored by F.cross_entropy


# ═════════════════════════════════════════════════════════════════════════════
# ANNOTATION LOADER
# ═════════════════════════════════════════════════════════════════════════════

def load_skill_annotation(annotation_path: Path) -> np.ndarray | None:
    """
    Loads an episode's annotation JSON and returns an array
    [num_frames] where each element is the class_index of the active skill
    in that frame, or IGNORE_LABEL if the frame is not annotated.

    Args:
        annotation_path: Path to the episode's JSON file.

    Returns:
        np.ndarray of shape [num_frames] with dtype int64,
        or None if the file does not exist.
    """
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
                f"frame {f_start}:{f_end} ignored. "
            )
            continue

        labels[f_start:f_end] = SKILL_ID_TO_CLASS[skill_id]

    return labels


# ═════════════════════════════════════════════════════════════════════════════
# DATASET WRAPPER
# ═════════════════════════════════════════════════════════════════════════════

class SkillLabeledDataset(Dataset):
    """
    Wrapper around LeRobotDataset that adds 'skill_label' to each
    sample, read from BehaviorBot 1K annotation JSON files.

    Args:
        lerobot_dataset: Pre-initialized LeRobotDataset instance.
        annotations_root: Annotations root folder, e.g.
            "~/.cache/huggingface/datasets/behavior-1k/2025-challenge-demos/annotations"
        ignore_unlabeled: If True, samples with the IGNORE_LABEL label are
            filtered from the index. If False, they are passed to the model with the
            IGNORE_LABEL label (ignored by loss thanks to ignore_index)
    """

    def __init__(
        self,
        lerobot_dataset: LeRobotDataset,
        annotations_root: Path,
        ignore_unlabeled: bool = True,
    ):
        self.dataset = lerobot_dataset
        self.annotations_root = Path(annotations_root)
        self._label_cache: dict[int, np.ndarray | None] = {}

        if ignore_unlabeled:
            logging.info(
                "Building annotated sample index "
                "(this may take a few minutes)..."
            )
            self._valid_indices = self._build_valid_indices()
            logging.info(
                f"Valid samples: {len(self._valid_indices)} / {len(self.dataset)}"
            )
        else:
            self._valid_indices = list(range(len(self.dataset)))

    def _build_valid_indices(self) -> list[int]:
        """Index only frames that have a valid skill label."""
        valid = []
        for i in range(len(self.dataset)):
            sample    = self.dataset[i]
            ep_idx    = int(sample["episode_index"])
            frame_idx = int(sample["frame_index"])
            labels    = self._get_episode_labels(ep_idx)
            if labels is not None and labels[frame_idx] != IGNORE_LABEL:
                valid.append(i)
        return valid

    def _get_episode_labels(self, episode_index: int) -> np.ndarray | None:
        """Load and cache an episode's labels from its annotation JSON."""
        if episode_index not in self._label_cache:
            # BehaviorBot path pattern:
            # annotations/task-XXXX/episode_XXXXXXXX.json
            # The first 4 characters of episode_index identify the task.
            ep_str   = f"{episode_index:08d}"
            task_str = f"task-{ep_str[:4]}"
            ann_path = (
                self.annotations_root / task_str / f"episode_{ep_str}.json"
            )
            self._label_cache[episode_index] = load_skill_annotation(ann_path)
        return self._label_cache[episode_index]

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        real_idx  = self._valid_indices[idx]
        sample    = self.dataset[real_idx]
        ep_idx    = int(sample["episode_index"])
        frame_idx = int(sample["frame_index"])
        labels    = self._get_episode_labels(ep_idx)

        if labels is None or frame_idx >= len(labels):
            skill_label = IGNORE_LABEL
        else:
            skill_label = int(labels[frame_idx])

        sample["skill_label"] = torch.tensor(skill_label, dtype=torch.long)
        return sample


# ═════════════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════════════

def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Global accuracy, ignoring samples with IGNORE_LABEL."""
    valid_mask = labels != IGNORE_LABEL
    if valid_mask.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    return (preds[valid_mask] == labels[valid_mask]).float().mean().item()


def compute_per_class_accuracy(
    logits: torch.Tensor, labels: torch.Tensor
) -> dict[str, float]:
    """
    Per-class accuracy, useful for detecting dataset imbalance.
    Returns a skill_name -> accuracy dictionary.
    """
    preds = logits.argmax(dim=-1)
    per_class = {}
    for c in range(NUM_SKILL_CLASSES):
        mask = labels == c
        if mask.sum() == 0:
            continue
        acc  = (preds[mask] == c).float().mean().item()
        name = CLASS_TO_SKILL_NAME.get(c, str(c))
        per_class[name] = acc
    return per_class


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING STEP
# Forward, cross-entropy loss computation, backward, weight update.
# ═════════════════════════════════════════════════════════════════════════════

def training_step(
    policy: PI0Policy,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    grad_clip_norm: float,
    device: torch.device,
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, float]:
    """
    Single classifier training step.

    policy.train() keeps PaliGemma in eval mode automatically
    thanks to train_expert_only=True in PI0Config.
    """
    policy.train()

    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    labels = batch["skill_label"]  # [B]

    ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if use_amp else nullcontext()
    )

    with ctx:
        #logits, _ = policy.forward(batch)
        #loss = F.cross_entropy(logits, labels, ignore_index=IGNORE_LABEL)
        logits, loss, loss_dict = policy.forward(batch)

    optimizer.zero_grad()

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
        "loss":      loss.item(),
        "accuracy":  acc,
        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
    }


# ═════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(
    policy: PI0Policy,
    val_loader: DataLoader,
    device: torch.device,
    num_batches: int | None = None,
) -> dict[str, float]:
    """
    Validation on val_loader.
    If num_batches is specified, validate only on the first N batches,
    useful for quick validation during training.
    Returns loss, global accuracy, and per-class accuracy.
    """
    policy.eval()

    loss_meter = AverageMeter("val_loss")
    acc_meter  = AverageMeter("val_accuracy")
    all_logits = []
    all_labels = []

    for i, batch in enumerate(val_loader):
        if num_batches is not None and i >= num_batches:
            break

        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        labels    = batch["skill_label"]
        # logits, _ = policy.forward(batch)
        # loss      = F.cross_entropy(logits, labels, ignore_index=IGNORE_LABEL)
        logits, loss, _ = policy.forward(batch)
        acc       = compute_accuracy(logits, labels)
        n_valid   = max((labels != IGNORE_LABEL).sum().item(), 1)

        loss_meter.update(loss.item(), n=labels.shape[0])
        acc_meter.update(acc, n=n_valid)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    per_class  = compute_per_class_accuracy(all_logits, all_labels)

    return {
        "val/loss":     loss_meter.avg,
        "val/accuracy": acc_meter.avg,
        **{f"val/acc_{name}": acc for name, acc in per_class.items()},
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

def train(cfg) -> None:
    """
    Main training loop. Follows the lerobot_train structure with
    added handling for BehaviorBot skill labels.

    Frozen parameters (train_expert_only=True):
      - full PaliGemma (SigLIP + Gemma 2B)

    Trained parameters:
      - gemma_expert     fine-tuning (adapts to the target robot)
      - state_proj       fine-tuning (adapts to the target robot joints)
      - attn_pool        training from scratch
      - classifier_head  training from scratch
    """
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    logging.info(f"Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────
    logging.info("Loading dataset...")

    # To build the train/val split, first read the full dataset only to get
    # the number of episodes, then create the two subsets.
    full_dataset = LeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=cfg.dataset.episodes,
        image_transforms=cfg.dataset.image_transforms,
        delta_timestamps=cfg.dataset.delta_timestamps,
        video_backend=cfg.dataset.video_backend,
    )

    annotations_root = Path(full_dataset.root) / "annotations"
    logging.info(f"Annotations from: {annotations_root}")

    # Train/val split by episode (90/10 by default)
    num_episodes  = full_dataset.num_episodes
    val_fraction  = getattr(cfg.dataset, "val_fraction", 0.1) # 0.1 = 10% validation, 90% training
    num_val_eps   = max(1, int(num_episodes * val_fraction))
    num_train_eps = num_episodes - num_val_eps

    # train_eps     = list(range(num_train_eps))
    # val_eps       = list(range(num_train_eps, num_episodes))


    rng = np.random.default_rng(cfg.seed)

    all_eps = np.arange(num_episodes)
    rng.shuffle(all_eps)

    val_eps = all_eps[:num_val_eps].tolist()
    train_eps = all_eps[num_val_eps:].tolist()


    logging.info(
        f"Split: {num_train_eps} train episodes, {num_val_eps} val episodes"
    )

    train_raw = LeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=train_eps,
        image_transforms=cfg.dataset.image_transforms,
        delta_timestamps=cfg.dataset.delta_timestamps,
        video_backend=cfg.dataset.video_backend,
    )
    val_raw = LeRobotDataset(
        repo_id=cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=val_eps,
        image_transforms=cfg.dataset.image_transforms,
        delta_timestamps=cfg.dataset.delta_timestamps,
        video_backend=cfg.dataset.video_backend,
    )

    train_dataset = SkillLabeledDataset(
        train_raw, annotations_root, ignore_unlabeled=True
    )
    val_dataset = SkillLabeledDataset(
        val_raw, annotations_root, ignore_unlabeled=True
    )
    logging.info(
        f"Annotated samples - train: {len(train_dataset)}, val: {len(val_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    # ── Policy ────────────────────────────────────────────────────────────
    # classifier_mode=True:   forward returns logits, does not generate actions
    # train_expert_only=True: freezes PaliGemma automatically,
    #                         including after each policy.train()
    logging.info("Loading PI0 policy...")

    policy_cfg = PI0Config(
        classifier_mode=True,
        train_expert_only=True,
        num_subskill_classes=NUM_SKILL_CLASSES,
        **{
            k: v for k, v in vars(cfg.policy).items()
            if k not in ("classifier_mode", "train_expert_only", "num_subskill_classes")
            and hasattr(PI0Config(), k)
        },
    )

    policy = PI0Policy.from_pretrained(
        cfg.policy.pretrained_model_name_or_path,
        config=policy_cfg,
        strict=False,  # in classifier_mode, pretrained action weights are missing
    )
    policy = policy.to(device)

    # Log trainable parameters for verification
    trainable = [(n, p.numel()) for n, p in policy.named_parameters() if p.requires_grad]
    frozen    = [(n, p.numel()) for n, p in policy.named_parameters() if not p.requires_grad]
    logging.info(f"Trainable parameters ({len(trainable)}):")
    for name, numel in trainable:
        logging.info(f"  {name:60s} {numel:>10,}")
    logging.info(
        f"Total trainable: {sum(n for _, n in trainable):,} | "
        f"Total frozen:    {sum(n for _, n in frozen):,}"
    )

    # ── Optimizer and scheduler ───────────────────────────────────────────
    trainable_params = [p for p in policy.parameters() if p.requires_grad]

    # optimizer = AdamW(
    #     trainable_params,
    #     lr=cfg.optimizer.lr,
    #     weight_decay=getattr(cfg.optimizer, "weight_decay", 1e-4),
    #     betas=getattr(cfg.optimizer, "betas", (0.9, 0.999)),
    # )
    optimizer = AdamW(
        trainable_params,
        lr=cfg.optimizer.lr,
        betas=getattr(cfg.optimizer, "betas", (0.9, 0.95)),
        eps=getattr(cfg.optimizer, "eps", 1e-8),
        weight_decay=getattr(cfg.optimizer, "weight_decay", 1e-10),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.num_train_steps,
        eta_min=getattr(cfg.optimizer, "lr_min", 1e-6),
    )

    use_amp = device.type == "cuda" and getattr(cfg.training, "use_amp", True)
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        logging.info("AMP enabled (bfloat16)")

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
    logging.info(f"Starting training - {cfg.training.num_train_steps} total steps")

    loss_meter      = AverageMeter("train_loss")
    acc_meter       = AverageMeter("train_accuracy")
    grad_norm_meter = AverageMeter("grad_norm")
    step_time_meter = AverageMeter("step_time")

    train_iter = iter(train_loader)

    while step < cfg.training.num_train_steps:

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
            grad_clip_norm=cfg.training.grad_clip_norm,
            device=device,
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
        if step % cfg.training.log_freq == 0:
            logging.info(
                f"Step {step:6d}/{cfg.training.num_train_steps} | "
                f"loss={loss_meter.avg:.4f} | "
                f"acc={acc_meter.avg:.3f} | "
                f"grad={grad_norm_meter.avg:.3f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | "
                f"t={step_time_meter.avg:.3f}s"
            )
            loss_meter.reset()
            acc_meter.reset()
            grad_norm_meter.reset()
            step_time_meter.reset()

        # ── Periodic validation ───────────────────────────────────────────
        if step % cfg.training.eval_freq == 0:
            val_metrics = validate(
                policy=policy,
                val_loader=val_loader,
                device=device,
                num_batches=getattr(cfg.training, "eval_num_batches", None),
            )
            val_summary = " | ".join(
                f"{k}={v:.4f}" for k, v in val_metrics.items()
            )
            logging.info(f"Step {step:6d} [VAL] {val_summary}")

            # Return to train mode after validation.
            # PaliGemma stays in eval thanks to train_expert_only.
            policy.train()

        # ── Checkpoint saving ─────────────────────────────────────────────
        if step % cfg.training.save_checkpoint_interval == 0:
            ckpt_dir = get_step_checkpoint_dir(output_dir, step)
            save_checkpoint(
                ckpt_dir,
                step=step,
                policy=policy,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            update_last_checkpoint(output_dir, ckpt_dir)
            logging.info(f"Checkpoint saved: {ckpt_dir}")

    # ── Final model saving ────────────────────────────────────────────────
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    policy.save_pretrained(final_dir)
    logging.info(f"Final model saved in: {final_dir}")

    # Final full validation on the entire val set
    logging.info("Running final full validation...")
    final_val = validate(policy, val_loader, device, num_batches=None)
    logging.info("Final results:")
    for k, v in final_val.items():
        logging.info(f"  {k}: {v:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# Follows the same pattern as lerobot_train: the config is parsed
# from CLI or YAML through the standard LeRobot parser.
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = parser.parse_args_into_dataclasses(TrainPipelineConfig)[0]
    logging.info(f"Config:\n{pformat(vars(cfg))}")
    train(cfg)
