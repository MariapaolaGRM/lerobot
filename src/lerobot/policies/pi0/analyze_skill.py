#!/usr/bin/env python
"""
Analisi della distribuzione delle skill nel dataset BehaviorBot 1K.

Itera tutti i file JSON di annotazione, raccoglie le statistiche per skill
e genera automaticamente il blocco SKILL_REGISTRY da copiare in
train_skill_classifier.py.

Usage:
    python analyze_skill.py \
        --annotations_root ~/Documents/behavior1k_training_mixed \
        --output_registry skill_registry_2.py

"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ═════════════════════════════════════════════════════════════════════════════
# RACCOLTA STATISTICHE
# ═════════════════════════════════════════════════════════════════════════════

def collect_statistics(annotations_root: Path) -> dict:
    """
    Itera tutti i JSON di annotazione e raccoglie per ogni skill_id:
      - nome (skill_description)
      - numero totale di frame annotati
      - numero di episodi in cui appare
      - numero di task in cui appare
      - lista dei task in cui appare

    Returns:
        dict skill_id -> {
            "name": str,
            "total_frames": int,
            "num_episodes": int,
            "num_tasks": int,
            "tasks": set of str,
        }
    """
    stats: dict[int, dict] = defaultdict(lambda: {
        "name":         "unknown",
        "total_frames": 0,
        "episodes":     set(),
        "tasks":        set(),
    })

    episode_skills = defaultdict(set)

    annotation_files = sorted(annotations_root.rglob("episode_*.json"))

    if not annotation_files:
        logging.error(
            f"Nessun file JSON trovato in {annotations_root}. "
            "Verifica il path delle annotazioni."
        )
        return {}

    logging.info(f"File di annotazione trovati: {len(annotation_files)}")

    for ann_path in annotation_files:
        task_str    = ann_path.parent.name          # es. "task-0021"
        episode_str = ann_path.stem                  # es. "episode_00210020"

        try:
            with open(ann_path) as f:
                ann = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logging.warning(f"Errore lettura {ann_path}: {e} — saltato.")
            continue

        for skill in ann.get("skill_annotation", []):
            skill_id   = skill["skill_id"][0]
            skill_desc = skill.get("skill_description", ["unknown"])
            skill_name = skill_desc[0] if isinstance(skill_desc, list) else skill_desc

            episode_skills[episode_str].add(
                f"{skill_id}:{skill_name}"
            )
            
            f_start, f_end = skill["frame_duration"]
            #num_frames = max(0, f_end - f_start)

            try:
                num_frames = max(0, f_end - f_start)
            except Exception:
                print("\nFILE:", ann_path)
                print("skill:", skill)
                print("f_start:", f_start, type(f_start))
                print("f_end:", f_end, type(f_end))
                raise

            stats[skill_id]["name"]          = skill_name
            stats[skill_id]["total_frames"] += num_frames
            stats[skill_id]["episodes"].add(episode_str)
            stats[skill_id]["tasks"].add(task_str)

    # ----------------------------------------------------
    # Salvataggio report
    # ----------------------------------------------------

    report_path = Path("episode_skill_summary.txt")

    with open(report_path, "w") as f:

        for episode in sorted(episode_skills):

            f.write(f"{episode}\n")

            for skill in sorted(episode_skills[episode]):
                f.write(f"    {skill}\n")

            f.write("\n")

    print(f"Report salvato in: {report_path}")
    return dict(stats)


# ═════════════════════════════════════════════════════════════════════════════
# STAMPA RISULTATI
# ═════════════════════════════════════════════════════════════════════════════

def print_statistics(stats: dict) -> None:
    """Stampa le statistiche in formato tabellare leggibile."""

    if not stats:
        logging.error("Nessuna statistica da mostrare.")
        return

    total_frames_all = sum(s["total_frames"] for s in stats.values())

    # Ordina per numero di frame decrescente
    sorted_skills = sorted(
        stats.items(), key=lambda x: x[1]["total_frames"], reverse=True
    )

    print("\n" + "═" * 90)
    print(f"{'DISTRIBUZIONE SKILL — BehaviorBot 1K':^90}")
    print("═" * 90)
    print(
        f"{'skill_id':>10}  "
        f"{'nome':<30}  "
        f"{'frame':>10}  "
        f"{'%':>6}  "
        f"{'episodi':>8}  "
        f"{'task':>6}"
    )
    print("─" * 90)

    for skill_id, s in sorted_skills:
        pct = 100.0 * s["total_frames"] / total_frames_all if total_frames_all > 0 else 0
        print(
            f"{skill_id:>10}  "
            f"{s['name']:<30}  "
            f"{s['total_frames']:>10,}  "
            f"{pct:>5.1f}%  "
            f"{len(s['episodes']):>8,}  "
            f"{len(s['tasks']):>6,}"
        )

    print("─" * 90)
    print(
        f"{'TOTALE':>10}  "
        f"{'':30}  "
        f"{total_frames_all:>10,}  "
        f"{'100.0%':>6}  "
        f"{len(set().union(*[s['episodes'] for s in stats.values()])):>8,}  "
        f"{len(set().union(*[s['tasks'] for s in stats.values()])):>6,}"
    )
    print("═" * 90 + "\n")

    # Avvisa se il dataset è molto sbilanciato
    max_frames = max(s["total_frames"] for s in stats.values())
    min_frames = min(s["total_frames"] for s in stats.values())
    if max_frames > 0 and min_frames > 0:
        ratio = max_frames / min_frames
        if ratio > 10:
            print(
                f"Dataset sbilanciato: la skill più frequente ha {ratio:.0f}x "
                f"più frame della meno frequente.\n"
                f"   Considera di usare class weights nella cross-entropy:\n"
                f"   weight = total_frames / (num_classes * frames_per_class)\n"
            )
    
    


# ═════════════════════════════════════════════════════════════════════════════
# GENERAZIONE SKILL_REGISTRY
# ═════════════════════════════════════════════════════════════════════════════

def generate_skill_registry(stats: dict) -> str:
    """
    Genera il blocco SKILL_REGISTRY da copiare in train_skill_classifier.py.
    Gli skill_id vengono ordinati per numero di frame decrescente e
    assegnati a class_index contigui 0, 1, 2, ...
    """
    sorted_skills = sorted(
        stats.items(), key=lambda x: x[1]["total_frames"], reverse=True
    )

    lines = [
        "# ── SKILL_REGISTRY generato da analyze_skill_distribution.py ──",
    ]

    for class_idx, (skill_id, s) in enumerate(sorted_skills):
        frames = s["total_frames"]
        name   = s["name"]
        lines.append(
            f"    {skill_id}: ({class_idx}, \"{name}\"),  "
            f"# {frames:,} frame"
        )

    lines += [
        "}",
        "",
        f"# Numero totale di classi: {len(sorted_skills)}",
    ]

    return "\n".join(lines)


def generate_class_weights(stats: dict) -> str:
    """
    Genera i class weights da usare in F.cross_entropy se il dataset
    è sbilanciato. Pesi inversamente proporzionali alla frequenza.
    """
    sorted_skills = sorted(
        stats.items(), key=lambda x: x[1]["total_frames"], reverse=True
    )

    total_frames = sum(s["total_frames"] for s in stats.values())
    num_classes  = len(sorted_skills)

    lines = [
        "# ── CLASS WEIGHTS per dataset sbilanciato ──",
        # "# Usa questi pesi in F.cross_entropy:",
        # "#   loss = F.cross_entropy(logits, labels,",
        # "#                          weight=CLASS_WEIGHTS.to(device),",
        # "#                          ignore_index=IGNORE_LABEL)",
        # "",
        # "CLASS_WEIGHTS = torch.tensor([",
    ]

    for class_idx, (skill_id, s) in enumerate(sorted_skills):
        frames = s["total_frames"]
        weight = total_frames / (num_classes * frames) if frames > 0 else 1.0
        lines.append(
            f"    {weight:.4f},  # class {class_idx}: {s['name']} ({frames:,} frame)"
        )

    lines += [
        "], dtype=torch.float32)",
    ]

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analisi distribuzione skill BehaviorBot 1K"
    )
    parser.add_argument(
        "--annotations_root",
        type=Path,
        required=True,
        help=(
            "Cartella radice delle annotazioni, es. "
            "~/.cache/huggingface/datasets/behavior-1k/2025-challenge-demos/annotations"
        ),
    )
    parser.add_argument(
        "--output_registry",
        type=Path,
        default=None,
        help=(
            "Se specificato, salva SKILL_REGISTRY e CLASS_WEIGHTS "
            "in questo file .py invece di stamparli a schermo."
        ),
    )
    args = parser.parse_args()

    annotations_root = args.annotations_root.expanduser().resolve()

    if not annotations_root.exists():
        logging.error(f"Path non trovato: {annotations_root}")
        return

    # ── Raccolta statistiche ───────────────────────────────────────────────
    logging.info(f"Analisi annotazioni in: {annotations_root}")
    stats = collect_statistics(annotations_root)

    if not stats:
        return

    # ── Stampa tabella ────────────────────────────────────────────────────
    print_statistics(stats)

    # ── Genera SKILL_REGISTRY ─────────────────────────────────────────────
    registry_code   = generate_skill_registry(stats)
    weights_code    = generate_class_weights(stats)
    full_output     = registry_code + "\n\n" + weights_code

    if args.output_registry is not None:
        args.output_registry.write_text(full_output)
        logging.info(f"SKILL_REGISTRY salvato in: {args.output_registry}")
    else:
        print("\n" + "═" * 90)
        print("SKILL_REGISTRY — copia in train_skill_classifier.py")
        print("═" * 90)
        print(registry_code)
        print("\n" + "═" * 90)
        print("CLASS_WEIGHTS — da usare se il dataset è sbilanciato")
        print("═" * 90)
        print(weights_code)
        print()


if __name__ == "__main__":
    main()