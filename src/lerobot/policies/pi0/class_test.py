#!/usr/bin/env python

import argparse
import copy
import logging
import sys
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

NUM_SKILL_CLASSES   = 9          # must match len(SKILL_REGISTRY) in training
IGNORE_LABEL        = -100       # must match IGNORE_LABEL in training
STATE_DIM           = 28         # BehaviorBot: 28 joint state values
IMG_H, IMG_W        = 224, 224   # image size expected by SigLIP
NUM_CAMERAS         = 2          # number of cameras (head + wrist in BehaviorBot)
LANG_SEQ_LEN        = 48         # language token sequence length
BATCH_SIZE_DEFAULT  = 2          

# Batch keys
OBS_STATE                   = "observation.state"
OBS_LANGUAGE_TOKENS         = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"
SKILL_LABEL_KEY             = "skill_label"


# Image keys for the two cameras (Behavior1K format)
IMG_KEYS = [
    "observation.images.head",
    "observation.images.wrist",
]

def make_fake_batch(
    batch_size: int,
    device: torch.device,
    all_labeled: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Build a synthetic batch with the same keys and dimensions expected by
    the model from LeRobotDataset + SkillLabeledDataset.

    Args:
        batch_size:   number of samples in the batch
        device:       device on which tensors are created
        all_labeled:  if True, all samples have a valid label;
                      if False, half have IGNORE_LABEL to test ignore_index
    """
    batch = {}

    # ── Images ────────────────────────────────────────────────────────────
    # Shape: [B, C, H, W] with values in [-1, 1] (SigLIP normalization)
    # Black image = all -1
    for key in IMG_KEYS:
        batch[key] = torch.full(
            (batch_size, 3, IMG_H, IMG_W),
            fill_value=-1.0,
            dtype=torch.float32,
            device=device,
        )

    # ── Robot state ───────────────────────────────────────────────────────
    # Shape: [B, STATE_DIM]
    # Reasonable joint values: approximately in the [-pi, pi] range
    batch[OBS_STATE] = torch.zeros(
        (batch_size, STATE_DIM),
        dtype=torch.float32,
        device=device,
    )

    # ── Language tokens ───────────────────────────────────────────────────
    # Shape: [B, LANG_SEQ_LEN] — integers (token ids)
    batch[OBS_LANGUAGE_TOKENS] = torch.ones(
        (batch_size, LANG_SEQ_LEN),
        dtype=torch.long,
        device=device,
    )

    # ── Language attention mask ───────────────────────────────────────────
    # Shape: [B, LANG_SEQ_LEN] — 1 for real tokens, 0 for padding
    # Simulate 20 real tokens + 28 padding tokens
    # lang_mask = torch.zeros((batch_size, LANG_SEQ_LEN), dtype=torch.long, device=device)
    # lang_mask[:, :20] = 1 # 20 real tokens with value 1
    # batch[OBS_LANGUAGE_ATTENTION_MASK] = lang_mask

    lang_mask = torch.zeros((batch_size, LANG_SEQ_LEN), dtype=torch.bool, device=device)
    lang_mask[:, :20] = True
    batch[OBS_LANGUAGE_ATTENTION_MASK] = lang_mask


    # ── Skill label ───────────────────────────────────────────────────────
    # Shape: [B] — class index in [0, NUM_SKILL_CLASSES-1] or IGNORE_LABEL
    if all_labeled:
        # Random but valid labels for all samples
        labels = torch.randint(
            low=0, high=NUM_SKILL_CLASSES,
            size=(batch_size,),
            dtype=torch.long,
            device=device,
        )
    else:
        # Half the samples have a valid label, half have IGNORE_LABEL
        labels = torch.randint(
            low=0, high=NUM_SKILL_CLASSES,
            size=(batch_size,),
            dtype=torch.long,
            device=device,
        )
        labels[batch_size // 2:] = IGNORE_LABEL

    batch[SKILL_LABEL_KEY] = labels

    return batch

def make_random_policy(device: torch.device) -> "PI0Policy":
    """
    Initialize PI0Policy in classifier_mode with fully random weights —
    no pretrained checkpoint required.

    Use config-based initialization instead of from_pretrained to avoid
    downloading the original weights.
    """

    try:
        from lerobot.policies.pi0.configuration_pi0 import PI0Config
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        from lerobot.configs import FeatureType, PolicyFeature
    except ImportError as e:
        log.error(f"Unable to import PI0: {e}")
        log.error("Make sure to run this script from the lerobot environment.")
        sys.exit(1)


    cfg = PI0Config(
        classifier_mode=True,
        train_expert_only=True,
        num_subskill_classes=NUM_SKILL_CLASSES,

        device=str(device),
        input_features={
            "observation.images.head": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, IMG_H, IMG_W),
            ),
            "observation.images.wrist": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, IMG_H, IMG_W),
            ),
            "observation.state": PolicyFeature(
                type=FeatureType.STATE,
                shape=(STATE_DIM,),
            ),
        },
    )


    # Config-based initialization creates the model with random weights.
    policy = PI0Policy(cfg)
    policy = policy.to(device)

    log.info("Policy initialized with random weights (no pretrained weights loaded)")
    return policy

def assert_shape(tensor: torch.Tensor, expected: tuple, name: str) -> None:
    """Check that a tensor has the expected shape and log the result."""
    actual = tuple(tensor.shape)
    if actual != expected:
        log.error(f"  ✗ {name}: shape {actual} != expected {expected}")
        raise AssertionError(f"Shape mismatch for {name}: {actual} != {expected}")
    log.info(f"  ✓ {name}: shape {actual}")


def assert_finite(tensor: torch.Tensor, name: str) -> None:
    """Check that a tensor does not contain NaN or Inf."""
    if not torch.isfinite(tensor).all():
        log.error(f"  ✗ {name}: contains NaN or Inf")
        raise AssertionError(f"{name} contains NaN or Inf")
    log.info(f"  ✓ {name}: finite values")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: synthetic batch
# ─────────────────────────────────────────────────────────────────────────────

def test_batch_construction(batch_size: int, device: torch.device) -> None:
    log.info("\n" + "="*60)
    log.info("TEST 1: Synthetic batch construction")

    batch = make_fake_batch(batch_size, device, all_labeled=True)

    log.info("Batch keys:")
    for key, val in batch.items():
        log.info(f"  {key}: shape={tuple(val.shape)}, dtype={val.dtype}")

    # Check required keys
    required_keys = IMG_KEYS + [
        OBS_STATE,
        OBS_LANGUAGE_TOKENS,
        OBS_LANGUAGE_ATTENTION_MASK,
        SKILL_LABEL_KEY,
    ]
    for key in required_keys:
        assert key in batch, f"Missing key in batch: {key}"
        log.info(f"  ✓ key '{key}' present")

    # Check dimensions
    assert_shape(batch[OBS_STATE], (batch_size, STATE_DIM), "observation.state")
    assert_shape(batch[OBS_LANGUAGE_TOKENS], (batch_size, LANG_SEQ_LEN), "language_tokens")
    assert_shape(batch[SKILL_LABEL_KEY], (batch_size,), "skill_label")
    for key in IMG_KEYS:
        assert_shape(batch[key], (batch_size, 3, IMG_H, IMG_W), key)

    log.info("TEST 1 PASSED ✓")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: forward pass — dimensions and values
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_pass(
    policy: "PI0Policy",
    batch_size: int,
    device: torch.device,
) -> tuple:
    log.info("\n" + "="*60)
    log.info("TEST 2: Forward pass — dimensions and values")

    batch = make_fake_batch(batch_size, device, all_labeled=True)

    policy.eval()
    with torch.no_grad():
        output = policy.forward(batch)

    # In classifier_mode, forward must return (logits, loss, loss_dict)
    assert len(output) == 3, (
        f"policy.forward must return 3 values (logits, loss, loss_dict), "
        f"but returned {len(output)}"
    )
    logits, loss, loss_dict = output

    log.info("Output del forward pass:")
    assert_shape(logits, (batch_size, NUM_SKILL_CLASSES), "logits")
    assert_finite(logits, "logits")

    assert loss.ndim == 0, f"loss must be a scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), f"loss = {loss.item()} is not finite"
    log.info(f"  ✓ loss: {loss.item():.4f} (finite scalar)")

    assert "loss" in loss_dict, "loss_dict must contain the 'loss' key"
    log.info(f"  ✓ loss_dict['loss']: {loss_dict['loss']:.4f}")

    log.info("TEST 2 PASSED ✓")
    return logits, loss


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: forward with IGNORE_LABEL — loss must not be NaN
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_with_ignore_label(
    policy: "PI0Policy",
    batch_size: int,
    device: torch.device,
) -> None:
    log.info("\n" + "="*60)
    log.info("TEST 3: Forward with IGNORE_LABEL — loss must be finite")

    batch = make_fake_batch(batch_size, device, all_labeled=False)

    n_ignored = (batch[SKILL_LABEL_KEY] == IGNORE_LABEL).sum().item()
    n_valid   = batch_size - n_ignored
    log.info(f"  Samples in batch: {batch_size} total, {n_valid} labeled, {n_ignored} IGNORE_LABEL")

    policy.eval()
    with torch.no_grad():
        logits, loss, _ = policy.forward(batch)

    assert torch.isfinite(loss), f"loss = {loss.item()} with IGNORE_LABEL is not finite"
    log.info(f"  ✓ loss with IGNORE_LABEL: {loss.item():.4f} (finite)")

    log.info("TEST 3 PASSED ✓")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: backward pass — gradients are computed
# ─────────────────────────────────────────────────────────────────────────────

def test_backward_pass(
    policy: "PI0Policy",
    batch_size: int,
    device: torch.device,
) -> None:
    log.info("\n" + "="*60)
    log.info("TEST 4: Backward pass — gradients computed")

    batch = make_fake_batch(batch_size, device, all_labeled=True)

    policy.train()
    logits, loss, _ = policy.forward(batch)
    loss.backward()

    # Check that trainable parameters have gradients
    trainable_with_grad = []
    trainable_without_grad = []
    for name, param in policy.named_parameters():
        if param.requires_grad:
            if param.grad is not None and param.grad.abs().sum() > 0:
                trainable_with_grad.append(name)
            else:
                trainable_without_grad.append(name)

    log.info(f"  Trainable parameters with gradients: {len(trainable_with_grad)}")
    for n in trainable_with_grad:
        log.info(f"    ✓ {n}")

    if trainable_without_grad:
        log.warning(f"  Trainable parameters WITHOUT gradients: {len(trainable_without_grad)}")
        for n in trainable_without_grad:
            log.warning(f"    ⚠ {n}")

    assert len(trainable_with_grad) > 0, \
        "No trainable parameter received a gradient — check the forward pass"

    # Check that frozen parameters do NOT have gradients
    frozen_with_grad = [
        name for name, param in policy.named_parameters()
        if not param.requires_grad and param.grad is not None
    ]
    if frozen_with_grad:
        log.error(f"  FROZEN parameters with gradients (this should not happen):")
        for n in frozen_with_grad:
            log.error(f"    ✗ {n}")
        raise AssertionError("Frozen parameters received gradients")
    log.info(f"  ✓ No frozen parameter received gradients")

    log.info("TEST 4 PASSED ✓")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: mini training loop (3 step)
# ─────────────────────────────────────────────────────────────────────────────

def test_mini_training_loop(
    policy: "PI0Policy",
    batch_size: int,
    device: torch.device,
    num_steps: int = 2,
) -> None:
    log.info("\n" + "="*60)
    log.info(f"TEST 5: Mini training loop ({num_steps} step)")

    optimizer = AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=1e-6)

    losses = []

    for step in range(1, num_steps + 1):
        batch = make_fake_batch(batch_size, device, all_labeled=True)

        policy.train()
        optimizer.zero_grad()

        logits, loss, loss_dict = policy.forward(batch)

        assert torch.isfinite(loss), f"Step {step}: loss is not finite ({loss.item()})"

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad],
            max_norm=1.0,
        )

        optimizer.step()
        scheduler.step()

        # Compute accuracy
        with torch.no_grad():
            labels = batch[SKILL_LABEL_KEY]
            preds  = logits.argmax(dim=-1)
            acc    = (preds == labels).float().mean().item()

        losses.append(loss.item())
        lr = scheduler.get_last_lr()[0]
        log.info(
            f"  Step {step}/{num_steps} | "
            f"loss={loss.item():.4f} | "
            f"acc={acc:.3f} | "
            f"grad_norm={grad_norm.item():.3f} | "
            f"lr={lr:.2e}"
        )

    assert all(torch.isfinite(torch.tensor(l)) for l in losses), \
        "Some losses in the training loop are not finite"
    log.info("TEST 5 PASSED ✓")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: mini validation loop (2 batch)
# ─────────────────────────────────────────────────────────────────────────────

def test_mini_validation_loop(
    policy: "PI0Policy",
    batch_size: int,
    device: torch.device,
    num_batches: int = 2,
) -> None:
    log.info("\n" + "="*60)
    log.info(f"TEST 6: Mini validation loop ({num_batches} batch)")

    policy.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for i in range(num_batches):
            batch   = make_fake_batch(batch_size, device, all_labeled=True)
            labels  = batch[SKILL_LABEL_KEY]
            logits, loss, _ = policy.forward(batch)

            assert torch.isfinite(loss), f"Batch {i}: val loss is not finite"
            log.info(f"  Batch {i+1}/{num_batches} | val_loss={loss.item():.4f}")

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Check aggregated dimensions
    assert_shape(all_logits, (batch_size * num_batches, NUM_SKILL_CLASSES), "aggregated all_logits")
    assert_shape(all_labels, (batch_size * num_batches,), "aggregated all_labels")

    log.info("TEST 6 PASSED ✓")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7: train → eval → train — PaliGemma stays in eval
# ─────────────────────────────────────────────────────────────────────────────

def test_eval_mode_consistency(policy: "PI0Policy") -> None:
    log.info("\n" + "="*60)
    log.info("TEST 7: train → eval → train — PaliGemma stays in eval (train_expert_only)")

    # After policy.train(), PaliGemma must stay in eval thanks to train_expert_only
    policy.train()

    paligemma = policy.model.paligemma_with_expert.paligemma
    is_train = paligemma.training

    if is_train:
        log.error(
            "  ✗ PaliGemma is in TRAIN mode after policy.train() — "
            "train_expert_only is not working correctly"
        )
        raise AssertionError("PaliGemma is in train mode — train_expert_only is not working")
    else:
        log.info("  ✓ PaliGemma is in EVAL mode after policy.train() (train_expert_only OK)")

    # After policy.eval(), everything must be in eval
    policy.eval()
    assert not paligemma.training, "PaliGemma must be in eval after policy.eval()"
    log.info("  ✓ PaliGemma is in EVAL mode after policy.eval()")

    log.info("TEST 7 PASSED ✓")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test for PI0 classifier training"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device: 'cpu' or 'cuda' (default: cpu)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE_DEFAULT,
        help=f"Synthetic batch size (default: {BATCH_SIZE_DEFAULT})"
    )
    args = parser.parse_args()

    print("START")

    device = torch.device(args.device)
    batch_size = args.batch_size

    log.info("\n" + "=" * 60)
    log.info("TEST — PI0 Skill Classifier Pipeline")
    log.info(f"Device:     {device}")
    log.info(f"Batch size: {batch_size}")
    log.info(f"Num skill classes: {NUM_SKILL_CLASSES}")
    log.info(f"State dim:         {STATE_DIM}")
    log.info(f"Image size:        {IMG_H}x{IMG_W}")
    log.info(f"Language seq len:  {LANG_SEQ_LEN}")

    passed = []
    failed = []

    def run_test(name: str, fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
            passed.append(name)
        except Exception as e:
            log.error(f"\n{'='*60}")
            log.error(f"TEST FAILED: {name}")
            log.error(f"Error: {e}")
            log.error(f"{'='*60}\n")
            failed.append((name, str(e)))

    # Initialize the policy once for all tests
    try:
        policy = make_random_policy(device)
    except SystemExit:
        log.error("Unable to initialize the policy. Check the lerobot installation.")
        sys.exit(1)

    # # Log trainable vs frozen parameters
    # trainable_params = [(n, p.numel()) for n, p in policy.named_parameters() if p.requires_grad]
    # frozen_params    = [(n, p.numel()) for n, p in policy.named_parameters() if not p.requires_grad]
    # log.info(f"Total trainable: {sum(n for _, n in trainable_params):,}")
    # log.info(f"Total frozen:    {sum(n for _, n in frozen_params):,}")

    # Run tests in sequence
    run_test("1 - Batch construction",      test_batch_construction,       batch_size, device)
    run_test("2 - Forward pass",            test_forward_pass,             policy, batch_size, device)
    run_test("3 - Forward with IGNORE_LABEL",test_forward_with_ignore_label,policy, batch_size, device)
    run_test("4 - Backward pass",           test_backward_pass,            policy, batch_size, device)
    run_test("5 - Mini training loop",      test_mini_training_loop,       policy, batch_size, device)
    run_test("6 - Mini validation loop",    test_mini_validation_loop,     policy, batch_size, device)
    run_test("7 - Eval mode consistency",   test_eval_mode_consistency,    policy)

    # Final summary
    log.info("\n" + "=" * 60)
    log.info("SUMMARY")
    log.info(f"  Passed: {len(passed)}/{len(passed) + len(failed)}")
    for name in passed:
        log.info(f"  ✓ {name}")
    if failed:
        log.info(f"  Failed: {len(failed)}")
        for name, err in failed:
            log.error(f"  ✗ {name}: {err}")
        sys.exit(1)
    else:
        log.info("All tests passed ✓")

if __name__ == "__main__":
    main()
