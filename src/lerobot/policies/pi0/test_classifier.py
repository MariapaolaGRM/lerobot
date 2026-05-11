#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from pathlib import Path
import json
from lerobot.policies.pi0.train_classifier import load_skill_annotation

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi0.train_classifier import SkillLabeledDataset

BATCH_SIZE = 4 # quattro esempi finti
NUM_CAMERAS = 2
NUM_PREFIX_TOKENS = NUM_CAMERAS * 256 + 48  # 560
PREFIX_DIM = 2048
SUFFIX_DIM = 1024
NUM_CLASSES = 9 # numero di classi del classificatore
STATE_DIM = 28 
SEQ_LEN = 16 # lunghezza dei token di testo
IGNORE_LABEL = -100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ═════════════════════════════════════════════════════════════════════════════
SKILL_REGISTRY: dict[int, tuple[int, str]] = {
    # skill_id: (class_index, name)
    1: (0, "move to"),  # 73,882 frame
    2: (1, "pick up from"),  # 66,453 frame
    4: (2, "place in"),  # 26,634 frame
    #10: (3, "open door"),  # 10,582 frame
    3: (4, "place on"),  # 8,291 frame
    12: (5, "close door"),  # 7,541 frame
    67: (6, "press"),  # 2,919 frame
    8:  (7, "toggle off"),
    90: (8, "push to"),

    9: (3, "push"),
    
}

SKILL_ID_TO_CLASS   = {sid: cls  for sid, (cls, _)   in SKILL_REGISTRY.items()}

def test_load_skill_annotation_real(annotations_root: Path):
    """
    Testa load_skill_annotation su file reali di BehaviorBot.
    Passa il path della cartella annotations del subset scaricato.
    """
    ann_files = sorted(annotations_root.rglob("episode_*.json"))
    assert len(ann_files) > 0, f"Nessun file trovato in {annotations_root}"
    print(f"File trovati: {len(ann_files)}")

    unknown_skill_ids = set()

    for ann_path in ann_files:
        labels = load_skill_annotation(ann_path)

        assert labels is not None, f"load_skill_annotation ha restituito None per {ann_path}"

        # Lunghezza coerente con task_duration nel JSON
        with open(ann_path) as f:
            ann = json.load(f)
        expected_len = ann["meta_data"]["task_duration"]
        assert len(labels) == expected_len, \
            f"{ann_path.name}: lunghezza array {len(labels)} != task_duration {expected_len}"

        # Tutti i valori sono IGNORE_LABEL o class_index validi
        valid_classes = set(SKILL_ID_TO_CLASS.values())
        for i, label in enumerate(labels):
            assert label == IGNORE_LABEL or label in valid_classes, \
                f"{ann_path.name} frame {i}: label {label} non valida"

        # Controlla skill_id non nel registry
        for skill in ann["skill_annotation"]:
            sid = skill["skill_id"][0]
            if sid not in SKILL_ID_TO_CLASS:
                unknown_skill_ids.add(sid)

        n_annotated = (labels != IGNORE_LABEL).sum()
        pct = 100 * n_annotated / len(labels)
        print(f"  {ann_path.name}: {len(labels)} frame, "
              f"{n_annotated} annotati ({pct:.2f}%)")

        # n_ignored = (labels == IGNORE_LABEL).sum()

        # print(f"  {ann_path.name}: {len(labels)} frame, "
        #     f"{n_annotated} annotati ({pct:.1f}%), "
        #     f"{n_ignored} ignorati")

    #print("  labels[:100]:", labels[:100])
    #print("  labels[-100:]:", labels[-900:-500])

    if unknown_skill_ids:
        print(f"Skill_id non nel SKILL_REGISTRY: {unknown_skill_ids}")
        print("  Aggiorna SKILL_REGISTRY con questi id")
    else:
        print("✓ tutti gli skill_id presenti nel SKILL_REGISTRY")


test_load_skill_annotation_real(
    Path("/home/mariapaolagerminario/Documents/behavior_subset/annotations")
)

# ═════════════════════════════════════════════════════════════════════════════
# PROVA
# per il funzionamento della funzione SkillLabeledDataset (non fattibile 
# perchè dataset in formato v2.1 ma attuale formato lerobot dataset v3.0)

# raw_dataset = LeRobotDataset(
#     repo_id="behavior-1k/2025-challenge-demos",
#     root=Path("/home/mariapaolagerminario/Documents/behavior_subset"),
# )

# annotations_root = Path("/home/mariapaolagerminario/Documents/behavior_subset/annotations")

# dataset_keep_all = SkillLabeledDataset(
#     raw_dataset,
#     annotations_root,
#     ignore_unlabeled=False,
# )

# dataset_filtered = SkillLabeledDataset(
#     raw_dataset,
#     annotations_root,
#     ignore_unlabeled=True,
# )

# print("raw dataset:", len(raw_dataset))
# print("keep all:", len(dataset_keep_all))
# print("filtered:", len(dataset_filtered))
# print("rimossi:", len(dataset_keep_all) - len(dataset_filtered))

# print("\n=== primi sample ignorati nel dataset keep_all ===")
# found = 0

# for i in range(len(dataset_keep_all)):
#     sample = dataset_keep_all[i]
#     label = sample["skill_label"].item()

#     if label == IGNORE_LABEL:
#         print(
#             "idx:", i,
#             "episode:", int(sample["episode_index"]),
#             "frame:", int(sample["frame_index"]),
#             "label:", label,
#         )
#         found += 1

#         if found >= 20:
#             break

# print("ignorati mostrati:", found)
# ═════════════════════════════════════════════════════════════════════════════
class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.scale = dim ** -0.5

    def forward(self, x, pad_mask=None):
        q = self.query.expand(x.shape[0], -1, -1)

        scores = torch.bmm(q, x.transpose(1, 2)) * self.scale

        if pad_mask is not None:
            scores = scores.masked_fill(
                ~pad_mask.unsqueeze(1),
                float("-inf")
            )

        weights = torch.softmax(scores, dim=-1)

        out = torch.bmm(weights, x)

        return out.squeeze(1)


classifier_head = nn.Sequential(
    nn.LayerNorm(3072),
    nn.Linear(3072, 1024),
    nn.ReLU(),
    nn.Linear(1024, NUM_CLASSES),
).to(DEVICE)

attn_pool = AttentionPooling(dim=PREFIX_DIM).to(DEVICE)

# ═════════════════════════════════════════════════════════════════════════════
batch = {
    "observation.images.top": torch.zeros(
        BATCH_SIZE,
        3,
        224,
        224,
        device=DEVICE,
    ),

    "observation.language_tokens": torch.randint(
        0,
        1000,
        (BATCH_SIZE, SEQ_LEN),
        device=DEVICE,
    ),

    "observation.language_attention_mask": torch.ones(
        BATCH_SIZE,
        SEQ_LEN,
        dtype=torch.bool,
        device=DEVICE,
    ),

    "observation.state": torch.randn(
        BATCH_SIZE,
        STATE_DIM,
        device=DEVICE,
    ),

    "skill_label": torch.randint(
        0,
        NUM_CLASSES,
        (BATCH_SIZE,),
        device=DEVICE,
    ),
}

print("\n=== BATCH ===")
for k, v in batch.items():
    print(f"{k:40s} {tuple(v.shape)}")

# ═════════════════════════════════════════════════════════════════════════════
prefix_out = torch.randn(
    BATCH_SIZE,
    NUM_PREFIX_TOKENS,
    PREFIX_DIM,
    device=DEVICE,
)

suffix_out = torch.randn(
    BATCH_SIZE,
    1,
    SUFFIX_DIM,
    device=DEVICE,
)

print("\n=== SIMULATED MODEL OUTPUTS ===")
print("prefix_out shape:", tuple(prefix_out.shape))
print("suffix_out shape:", tuple(suffix_out.shape))

# ═════════════════════════════════════════════════════════════════════════════
prefix_pad_masks = torch.ones(
    BATCH_SIZE,
    NUM_PREFIX_TOKENS,
    dtype=torch.bool,
    device=DEVICE,
)


state_feat = suffix_out[:, 0, :]

vlm_feat = attn_pool(
    prefix_out,
    pad_mask=prefix_pad_masks,
)

print("\n=== FEATURES ===")
print("state_feat shape:", tuple(state_feat.shape))
print("vlm_feat shape:", tuple(vlm_feat.shape))


x = torch.cat([vlm_feat, state_feat], dim=-1)

print("\n=== CONCAT ===")
print("concat shape:", tuple(x.shape))


logits = classifier_head(x)

print("\n=== LOGITS ===")
print("logits shape:", tuple(logits.shape))


labels = batch["skill_label"]

loss = F.cross_entropy(logits, labels)

print("\n=== LOSS ===")
print("cross entropy:", loss.item())

# ═════════════════════════════════════════════════════════════════════════════
labels_ignore = labels.clone()
labels_ignore[0] = IGNORE_LABEL

loss_ignore = F.cross_entropy(
    logits,
    labels_ignore,
    ignore_index=IGNORE_LABEL,
)

print("loss with ignore label:", loss_ignore.item())


preds = logits.argmax(dim=-1)
print("\n=== PREDICTIONS ===")
print("pred:", preds)

valid_mask = labels_ignore != IGNORE_LABEL

acc = (
    preds[valid_mask] == labels_ignore[valid_mask]
).float().mean()

print("\n=== ACCURACY ===")
print("accuracy:", acc.item())

# ═════════════════════════════════════════════════════════════════════════════
def fake_forward(x, labels):
    logits = classifier_head(x)
    loss = F.cross_entropy(logits, labels, ignore_index=IGNORE_LABEL)
    loss_dict = {"loss": loss.item()}
    return loss, loss_dict, logits   # come PI0Policy.forward

loss, loss_dict, logits = fake_forward(x, labels)
print("\n=== SIMULAZIONE PI0Policy.forward ===")
print("loss:", loss.item())
print("loss_dict:", loss_dict)
print("logits shape:", tuple(logits.shape))

# ═════════════════════════════════════════════════════════════════════════════
print("\n=== MINI TRAINING LOOP ===")

optimizer = AdamW(
    list(classifier_head.parameters()) +
    list(attn_pool.parameters()),
    lr=1e-4,
) 

for step in range(11):

    optimizer.zero_grad()

    vlm_feat = attn_pool(prefix_out, pad_mask=prefix_pad_masks)
    state_feat = suffix_out[:, 0, :]

    x = torch.cat([vlm_feat, state_feat], dim=-1)

    logits = classifier_head(x)

    loss = F.cross_entropy(logits, labels)

    loss.backward()

    attn_grad = attn_pool.query.grad.norm().item()
    clf_grad = classifier_head[-1].weight.grad.norm().item()

    optimizer.step()

    print(
        f"step={step:02d} | "
        f"loss={loss.item():.4f} | "
        f"attn_grad={attn_grad:.4f} | "
        f"clf_grad={clf_grad:.4f}"
    )


print("\nTest completed successfully.")
