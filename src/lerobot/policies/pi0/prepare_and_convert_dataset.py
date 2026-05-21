#!/usr/bin/env python3
"""
Script completo per preparare un subset del dataset BehaviorBot 1K
alla conversione nel formato LeRobot v3.0.

Usage:
    python /home/mariapaolagerminario/venvs/lerobot/lerobot/src/lerobot/policies/pi0/prepare_and_convert_dataset.py \
        --root ~/Documents/behavior_subset \
        --convert_script /home/mariapaolagerminario/venvs/lerobot/lerobot/src/lerobot/scripts/convert_dataset_v21_to_v30.py \
        --repo_id behavior-1k/2025-challenge-demos

    # Dry run — mostra cosa farebbe senza modificare nulla
    python prepare_and_convert_dataset.py \
        --root ~/Documents/behavior_subset \
        --convert_script /path/to/convert_dataset_v21_to_v30.py \
        --repo_id behavior-1k/2025-challenge-demos \
        --dry_run
"""

import argparse
import json
import logging
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# STEP 0 — Spostamenti annotations
# ═════════════════════════════════════════════════════════════════════════════

# def copy_dataset(root: Path, dry_run: bool) -> Path:
#     """
#     Copia root → root_v30 in modo da lavorare sulla copia
#     e lasciare intatto il dataset originale.
#     Le annotations vengono incluse nella copia.
#     """
#     output_root = root.parent / (root.name + "_v30")

#     if output_root.exists():
#         log.info(f"Cartella {output_root} già esistente — eliminata e ricreata")
#         if not dry_run:
#             shutil.rmtree(output_root)

#     log.info(f"Copia dataset: {root} → {output_root}")

#     if dry_run:
#         n_files = sum(1 for _ in root.rglob("*") if _.is_file())
#         log.info(f"[DRY RUN] Verrebbero copiati ~{n_files} file")
#         return output_root

#     shutil.copytree(root, output_root)
#     log.info(f"Copia completata")

#     # Verifica che le annotations siano state copiate
#     ann = output_root / "annotations"
#     if ann.exists():
#         n_ann = len(list(ann.rglob("*.json")))
#         log.info(f"Annotations incluse: {n_ann} file JSON")
#     else:
#         log.warning("Cartella annotations non trovata nel dataset originale")

#     return output_root
# ═════════════════════════════════════════════════════════════════════════════
def move_annotations_out(root: Path, dry_run: bool) -> Path:
    src = root / "annotations"
    tmp = root.parent / f"{root.name}_annotations_tmp"

    if tmp.exists():
        shutil.rmtree(tmp)

    log.info(f"Sposto annotations: {src} → {tmp}")

    if not dry_run and src.exists():
        shutil.move(str(src), str(tmp))

    return tmp

def move_annotations_back(root: Path, tmp_annotations: Path, dry_run: bool) -> None:
    dst = root / "annotations"

    if dst.exists():
        shutil.rmtree(dst)

    log.info(f"Ripristino annotations: {tmp_annotations} → {dst}")

    if not dry_run and tmp_annotations.exists():
        shutil.move(str(tmp_annotations), str(dst))

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Trova gli episode_index presenti nei dati
# ═════════════════════════════════════════════════════════════════════════════

def find_existing_episodes(root: Path) -> list[int]:
    """
    Legge la cartella data/ e restituisce la lista ordinata
    degli episode_index effettivamente presenti.
    """
    data_path = root / "data"
    indices = sorted([
        int(re.search(r"episode_(\d+)", f.stem).group(1))
        for f in data_path.rglob("*.parquet")
        if re.search(r"episode_(\d+)", f.stem)
    ])
    log.info(f"Episodi presenti: {len(indices)} → {indices[:5]}{'...' if len(indices) > 5 else ''}")
    return indices

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Rimuovi camere video non scaricate da info.json
# ═════════════════════════════════════════════════════════════════════════════

def remove_missing_cameras(root: Path, dry_run: bool) -> None:
    """
    Confronta le camere dichiarate in info.json con le cartelle
    video effettivamente presenti e rimuove le camere mancanti.
    """
    info_path = root / "meta" / "info.json"
    info = json.load(open(info_path))

    declared_cams = [
        k for k, v in info["features"].items()
        if v.get("dtype") == "video"
    ]

    videos_path = root / "videos"
    present_cams = set()
    if videos_path.exists():
        for task_dir in videos_path.iterdir():
            if task_dir.is_dir():
                for cam_dir in task_dir.iterdir():
                    if cam_dir.is_dir() and list(cam_dir.glob("*.mp4")):
                        present_cams.add(cam_dir.name)

    missing_cams = [c for c in declared_cams if c not in present_cams]

    if not missing_cams:
        log.info("Tutte le camere dichiarate sono presenti — nessuna rimozione necessaria")
        return

    log.info(f"Camere da rimuovere da info.json: {missing_cams}")

    if dry_run:
        log.info("[DRY RUN] info.json non modificato")
        return

    for cam in missing_cams:
        del info["features"][cam]

    n_present = len(present_cams)
    # total_videos calcolato sul numero di episodi reali (non 10000)
    # verrà aggiornato correttamente nello step 5
    info["total_videos"] = n_present * info["total_episodes"]
    log.info(f"Camere rimaste: {sorted(present_cams)}")

    json.dump(info, open(info_path, "w"), indent=2)
    log.info("info.json aggiornato — camere mancanti rimosse")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — Aggiungi total_chunks a info.json
# ═════════════════════════════════════════════════════════════════════════════

def add_total_chunks(root: Path, dry_run: bool) -> None:
    """
    Aggiunge il campo total_chunks a info.json se mancante.
    """
    info_path = root / "meta" / "info.json"
    info = json.load(open(info_path))

    if "total_chunks" in info:
        log.info(f"total_chunks già presente: {info['total_chunks']}")
        return

    total_chunks = max(1, math.ceil(info["total_episodes"] / info.get("chunks_size", 1000)))
    log.info(f"total_chunks calcolato: {total_chunks}")

    if dry_run:
        log.info("[DRY RUN] info.json non modificato")
        return

    info["total_chunks"] = total_chunks
    json.dump(info, open(info_path, "w"), indent=2)
    log.info("info.json aggiornato — total_chunks aggiunto")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — Rinumera gli episodi in modo contiguo
# ═════════════════════════════════════════════════════════════════════════════

def renumber_episodes(root: Path, existing_indices: list[int], dry_run: bool) -> dict[int, int]:
    """
    Rinumera tutti i file del dataset in modo che gli episode_index
    siano contigui (0, 1, 2, ...).
    Rinomina: parquet, video, meta/episodes/*.json, annotations/*.json
    Restituisce la mappa {vecchio_index → nuovo_index}.
    """
    remap = {old: new for new, old in enumerate(existing_indices)}
    log.info(f"Rimappatura episodi: {dict(list(remap.items())[:3])} ...")

    if dry_run:
        log.info("[DRY RUN] Nessun file rinominato")
        return remap

    # def rename_file(f: Path) -> None:
    #     m = re.search(r"episode_(\d+)", f.stem)
    #     if not m:
    #         return
    #     old_idx = int(m.group(1))
    #     if old_idx not in remap:
    #         return
    #     new_idx = remap[old_idx]
    #     new_name = f.name.replace(
    #         f"episode_{old_idx:08d}",
    #         f"episode_{new_idx:08d}"
    #     )
    #     if new_name != f.name:
    #         f.rename(f.parent / new_name)
    
    def safe_rename_files(files: list[Path]) -> int:
        """
        Rinomina una lista di file in due passaggi (old→tmp→new)
        per evitare sovrascritture quando old e new si sovrappongono.
        Restituisce il numero di file rinominati con successo.
        """
        # Passaggio 1: old → tmp
        tmp_map: dict[Path, tuple[Path, int]] = {}  # tmp_path → (parent, new_idx)
        for f in files:
            m = re.search(r"episode_(\d+)", f.stem)
            if not m:
                continue
            old_idx = int(m.group(1))
            if old_idx not in remap:
                continue
            new_idx = remap[old_idx]
            tmp_name = f.name.replace(
                f"episode_{old_idx:08d}",
                f"episode_tmp_{new_idx:08d}"
            )
            tmp_path = f.parent / tmp_name
            try:
                f.rename(tmp_path)
                tmp_map[tmp_path] = (f.parent, new_idx)
            except Exception as e:
                log.error(f"  ERRORE rinominando {f} → {tmp_path}: {e}")
 
        # Passaggio 2: tmp → new
        renamed = 0
        for tmp_path, (parent, new_idx) in tmp_map.items():
            final_name = tmp_path.name.replace(
                f"episode_tmp_{new_idx:08d}",
                f"episode_{new_idx:08d}"
            )
            final_path = parent / final_name
            try:
                tmp_path.rename(final_path)
                renamed += 1
            except Exception as e:
                log.error(f"  ERRORE rinominando {tmp_path} → {final_path}: {e}")
 
        return renamed


    # Parquet - rinomina file
    parquet_files = list((root / "data").rglob("*.parquet"))
    # for f in parquet_files:
    #     rename_file(f)
    # log.info(f"Rinominati {len(parquet_files)} file parquet")
    renamed_pq = safe_rename_files(parquet_files)
    log.info(f"Rinominati {renamed_pq}/{len(parquet_files)} file parquet")
    

    # Parquet - aggiorna episode_index e aggiungi frame_index nel contenuto
    updated = 0
    for f in sorted((root / "data").rglob("*.parquet")):
        m = re.search(r"episode_(\d+)", f.stem)
        if not m:
            continue
        new_idx = int(m.group(1))   # il nuovo indice è già nel nome del file rinominato
        df = pd.read_parquet(f)

        # Aggiorna episode_index nel contenuto
        df["episode_index"] = new_idx

        # Aggiungi frame_index (indice locale all'episodio) se mancante
        if "frame_index" not in df.columns:
            df["frame_index"] = range(len(df))

        df.to_parquet(f, index=False)
        updated += 1

    log.info(f"Aggiornati {updated} file parquet (episode_index + frame_index)")

    # Video
    video_files = list((root / "videos").rglob("*.mp4"))
    # for f in video_files:
    #     rename_file(f)
    # log.info(f"Rinominati {len(video_files)} file video")
    renamed_vid = safe_rename_files(video_files)
    log.info(f"Rinominati {renamed_vid}/{len(video_files)} file video")


    # meta/episodes/*.json — aggiorna anche il contenuto
    episodes_dir = root / "meta" / "episodes"
    ep_json_files = list(episodes_dir.rglob("episode_*.json")) if episodes_dir.exists() else []
    
    tmp_ep_map: dict[Path, Path] = {}  # tmp → final
    for f in ep_json_files:
        m = re.search(r"episode_(\d+)", f.stem)
        if not m:
            continue
        old_idx = int(m.group(1))
        if old_idx not in remap:
            continue
        new_idx = remap[old_idx]
        content = json.load(open(f))
        content["episode_index"] = new_idx
        # new_name = f.name.replace(f"episode_{old_idx:08d}", f"episode_{new_idx:08d}")
        # new_path = f.parent / new_name
        # json.dump(content, open(new_path, "w"), indent=2)
        # if new_path != f:
        #     f.unlink()
        tmp_name = f.name.replace(f"episode_{old_idx:08d}", f"episode_tmp_{new_idx:08d}")
        final_name = f.name.replace(f"episode_{old_idx:08d}", f"episode_{new_idx:08d}")
        tmp_path = f.parent / tmp_name
        final_path = f.parent / final_name
        json.dump(content, open(tmp_path, "w"), indent=2)
        f.unlink()
        tmp_ep_map[tmp_path] = final_path

    log.info(f"Rinominati {len(ep_json_files)} file meta/episodes/")

    # Annotazioni — aggiorna anche il contenuto
    annotations_dir = root / "annotations"
    ann_files = list(annotations_dir.rglob("episode_*.json")) if annotations_dir.exists() else []
    
    tmp_ann_map: dict[Path, Path] = {}

    for f in ann_files:
        m = re.search(r"episode_(\d+)", f.stem)
        if not m:
            continue
        old_idx = int(m.group(1))
        if old_idx not in remap:
            continue
        new_idx = remap[old_idx]
        content = json.load(open(f))
        if "episode_index" in content:
            content["episode_index"] = new_idx

        tmp_name = f.name.replace(f"episode_{old_idx:08d}", f"episode_tmp_{new_idx:08d}")
        final_name = f.name.replace(f"episode_{old_idx:08d}", f"episode_{new_idx:08d}")
        tmp_path = f.parent / tmp_name
        final_path = f.parent / final_name
        json.dump(content, open(tmp_path, "w"), indent=2)
        f.unlink()
        tmp_ann_map[tmp_path] = final_path
 
    for tmp_path, final_path in tmp_ann_map.items():
        tmp_path.rename(final_path)
    log.info(f"Rinominati {len(tmp_ann_map)} file annotazioni")

    #     new_name = f.name.replace(f"episode_{old_idx:08d}", f"episode_{new_idx:08d}")
    #     new_path = f.parent / new_name
    #     json.dump(content, open(new_path, "w"), indent=2)
    #     if new_path != f:
    #         f.unlink()
    # log.info(f"Rinominati {len(ann_files)} file annotazioni")

    return remap


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — Allinea i metadati al subset
# ═════════════════════════════════════════════════════════════════════════════

def align_metadata(root: Path, existing_indices: list[int], remap: dict[int, int], dry_run: bool) -> None:
    """
    Aggiorna tutti i file di metadati per riflettere
    solo gli episodi presenti nel subset, con i nuovi indici.
    """
    meta = root / "meta"
    old_indices = set(existing_indices)

    # episodes.jsonl
    ep_path = meta / "episodes.jsonl"
    all_eps = [json.loads(l) for l in open(ep_path)]
    filtered_eps = [
        {**ep, "episode_index": remap[ep["episode_index"]]}
        for ep in all_eps
        if ep["episode_index"] in old_indices
    ]
    filtered_eps.sort(key=lambda e: e["episode_index"])
    log.info(f"episodes.jsonl: {len(all_eps)} → {len(filtered_eps)}")
    if not dry_run:
        with open(ep_path, "w") as f:
            for ep in filtered_eps:
                f.write(json.dumps(ep) + "\n")

    # episodes_stats.jsonl
    stats_path = meta / "episodes_stats.jsonl"
    if stats_path.exists():
        all_stats = [json.loads(l) for l in open(stats_path)]
        filtered_stats = [
            {**s, "episode_index": remap[s["episode_index"]]}
            for s in all_stats
            if s["episode_index"] in old_indices
        ]
        filtered_stats.sort(key=lambda s: s["episode_index"])
        log.info(f"episodes_stats.jsonl: {len(all_stats)} → {len(filtered_stats)}")
        if not dry_run:
            with open(stats_path, "w") as f:
                for s in filtered_stats:
                    f.write(json.dumps(s) + "\n")

    # info.json
    info_path = meta / "info.json"
    info = json.load(open(info_path))
    n_eps = len(existing_indices)
    total_frames = sum(ep["length"] for ep in filtered_eps)
    n_cams = len([k for k, v in info["features"].items() if v.get("dtype") == "video"])

    info["total_episodes"] = n_eps
    info["total_frames"]   = total_frames
    info["total_videos"]   = n_cams * n_eps   # corretto con episodi reali
    info["splits"]         = {"train": f"0:{n_eps}"}
    info["total_chunks"]   = max(1, math.ceil(n_eps / info.get("chunks_size", 1000)))

    log.info(f"info.json: {n_eps} episodi, {total_frames} frame, {n_cams} camere")

    # Aggiungi frame_index alle features se mancante
    if "frame_index" not in info["features"]:
        info["features"]["frame_index"] = {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        }
        log.info("frame_index aggiunto alle features in info.json")
    
    # salva
    if not dry_run:
        json.dump(info, open(info_path, "w"), indent=2)

    log.info("Metadati allineati al subset")


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — Esegui la conversione v21→v30 in-place
# ═════════════════════════════════════════════════════════════════════════════

def run_conversion(root: Path, convert_script: Path, repo_id: str, dry_run: bool) -> None:
    """
    Esegue lo script di conversione v21→v30 sulla cartella root.
    La conversione avviene in-place — scrive tutto in root stessa.
    """
    log.info(f"Avvio conversione in-place su: {root}")

    cmd = [
        sys.executable,
        str(convert_script),
        "--repo-id", repo_id,
        "--root", str(root),
        "--push-to-hub", "False",
    ]

    if dry_run:
        log.info(f"[DRY RUN] Comando che verrebbe eseguito:\n  {' '.join(cmd)}")
        return

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        log.error("Conversione fallita — controlla l'output sopra")
        sys.exit(1)

    log.info("Conversione completata")

# ═════════════════════════════════════════════════════════════════════════════
# preprocessing sul dataset convertito
# ═════════════════════════════════════════════════════════════════════════════
def fix_info_json(
    root: Path,
    #remove_task_info: bool = False,
    task_info_dim: int | None = None,
    state_dtype: str | None = None,
    dry_run: bool = False,
) -> None:

    info_path = root / "meta" / "info.json"

    if not info_path.exists():
        log.error(f"info.json non trovato: {info_path}")
        return

    log.info(f"Correzione info.json: {info_path}")

    with open(info_path, "r") as f:
        info = json.load(f)

    features = info.get("features", {})

    # ---------------------------------------------------------
    # observation.task_info 
    # ---------------------------------------------------------

    # if task_info_dim is not None:
    #     if "observation.task_info" in features:
    #         old_shape = features["observation.task_info"].get("shape")

    #         features["observation.task_info"]["shape"] = [task_info_dim]

    #         log.info(
    #             f"  ✓ observation.task_info shape: "
    #             f"{old_shape} → {[task_info_dim]}"
    #         )
    #     else:
    #         log.warning("  observation.task_info non presente")

    # ---------------------------------------------------------
    # observation.state dtype
    # ---------------------------------------------------------

    if state_dtype is not None:
        if "observation.state" in features:
            old_dtype = features["observation.state"].get("dtype")

            features["observation.state"]["dtype"] = state_dtype

            log.info(
                f"  ✓ observation.state dtype: "
                f"{old_dtype} → {state_dtype}"
            )
        else:
            log.warning("  observation.state non presente")

    # ---------------------------------------------------------
    # task_index
    # ---------------------------------------------------------

    if "task_index" not in features:
        features["task_index"] = {
            "dtype": "int64",
            "shape": [1],
            "names": None,
        }

        log.info("  ✓ task_index aggiunto")

    # ---------------------------------------------------------
    # rimozione observation.task_info
    # ---------------------------------------------------------
    data_path = root/ "data" 

    #data_path = Path('~/Documents/Test/behavior_subset/data').expanduser()
    #info_path = Path('~/Documents/Test/behavior_subset/meta/info.json').expanduser()
    # Rimuovi dai parquet
    files = list(data_path.rglob('*.parquet'))
    removed = 0
    for f in files:
        df = pd.read_parquet(f)
        if 'observation.task_info' in df.columns:
            df = df.drop(columns=['observation.task_info'])
            df.to_parquet(f, index=False)
            removed += 1

    print(f'Rimossa da {removed}/{len(files)} file parquet')

    # Rimuovi da info.json
    #info = json.load(open(info_path))

    # if 'observation.task_info' in info['features']:
    #     del info['features']['observation.task_info']
    #     json.dump(info, open(info_path, 'w'), indent=2)
    #     print('Rimossa da info.json')
    # else:
    #     print('observation.task_info non in info.json')
    
    if 'observation.task_info' in features:
        del features['observation.task_info']
        print('Rimossa da info.json')
    else:
        print('observation.task_info non in info.json')

    # ---------------------------------------------------------
    # Salvataggio
    # ---------------------------------------------------------

    if dry_run:
        log.info("dry_run=True → nessuna modifica salvata")
        return

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    log.info("info.json aggiornato con successo")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 7 — Verifica finale
# ═════════════════════════════════════════════════════════════════════════════

def verify_result(root: Path) -> None:
#def verify_result(root: Path, repo_id: str) -> None:
    """
    Verifica che il dataset convertito sia caricabile con LeRobotDataset
    e che le annotations siano presenti.
    """
    log.info("Verifica del dataset convertito...")

    # Verifica versione nel meta
    info_path = root / "meta" / "info.json"
    if info_path.exists():
        info = json.load(open(info_path))
        log.info(f"  codebase_version: {info.get('codebase_version', 'non trovata')}")

    # Verifica annotations
    ann = root / "annotations"
    if ann.exists():
        n_ann = len(list(ann.rglob("*.json")))
        log.info(f"  Annotations presenti: {n_ann} file JSON")
    else:
        log.warning("  Cartella annotations NON presente — da verificare")

    # Prova a caricare con LeRobotDataset
    # try:
    #     from lerobot.datasets.lerobot_dataset import LeRobotDataset
    #     ds = LeRobotDataset(repo_id=repo_id, root=root)
    #     log.info(f"  Dataset caricato: {ds.num_episodes} episodi, {len(ds)} frame")
    #     log.info(f"  Feature: {list(ds.features.keys())}")
    #     sample = ds[0]
    #     log.info(f"  Chiavi primo sample: {list(sample.keys())}")
    #     log.info("Verifica PASSATA ✓")
    # except Exception as e:
    #     log.error(f"Verifica fallita: {e}")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepara e converte un subset di BehaviorBot 1K in formato LeRobot v3.0"
    )
    parser.add_argument(
        "--root", type=Path, required=True,
        help="Path locale del dataset originale (es. ~/Documents/behavior_subset)"
    )
    parser.add_argument(
        "--convert_script", type=Path, required=True,
        help="Path dello script convert_dataset_v21_to_v30.py"
    )
    parser.add_argument(
        "--repo_id", type=str, default="behavior-1k/2025-challenge-demos",
        help="Repository ID del dataset su HuggingFace"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Mostra cosa verrebbe fatto senza modificare nulla"
    )
    args = parser.parse_args()

    root           = args.root.expanduser().resolve() # cartella originale
    convert_script = args.convert_script.expanduser().resolve()
    dry_run        = args.dry_run

    if not root.exists():
        log.error(f"Path non trovato: {root}")
        sys.exit(1)
    if not convert_script.exists():
        log.error(f"Script di conversione non trovato: {convert_script}")
        sys.exit(1)

    log.info("=" * 60)
    log.info("PREPARAZIONE E CONVERSIONE DATASET BEHAVIOR BOT 1K")
    log.info("=" * 60)
    log.info(f"Root originale: {root}")
    log.info(f"Script:         {convert_script}")
    log.info(f"Dry run:        {dry_run}")


    # Step 1 — Trova episodi presenti
    log.info("\n── Step 1: Ricerca episodi presenti ──")
    existing = find_existing_episodes(root if not dry_run else root)

    # Step 2 — Rimuovi camere mancanti
    log.info("\n── Step 2: Rimozione camere non scaricate ──")
    remove_missing_cameras(root if not dry_run else root, dry_run)

    # Step 3 — Aggiungi total_chunks
    log.info("\n── Step 3: Aggiunta total_chunks ──")
    add_total_chunks(root if not dry_run else root, dry_run)

    # Step 4 — Rinumera episodi
    log.info("\n── Step 4: Rinumerazione episodi ──")
    remap = renumber_episodes(root if not dry_run else root, existing, dry_run)

    # Step 5 — Allinea metadati
    log.info("\n── Step 5: Allineamento metadati ──")
    align_metadata(root if not dry_run else root, existing, remap, dry_run)

    # Step 6 — Sposta annotations fuori dal dataset
    log.info("\n── Step 5: Spostamento temporaneo annotations ──")
    tmp_annotations = move_annotations_out(root, dry_run)

    # Step 7 — Conversione v21→v30 in-place su work_root
    log.info("\n── Step 7: Conversione v21→v30 ──")
    run_conversion(root if not dry_run else root, convert_script, args.repo_id, dry_run)

    # Step 8 — Riporta annotations nel dataset convertito
    log.info("\n── Step 8: Riposizionamento annotations ──")
    move_annotations_back(root, tmp_annotations, dry_run)

    # Step 9 — Elimina cartella backup
    log.info("\n── Step 9: Eliminazione repo temporanea ──")
    backup_root = Path(str(root) + "_old")
    if backup_root.exists():
        shutil.rmtree(backup_root)
        log.info(f"Backup eliminato: {backup_root}")
    
    
    # Step 9 — Correzione info.json
    log.info("\n── Step 9: Correzione info.json ──")
    fix_info_json(
        root,
        task_info_dim=130,
        state_dtype="float64",
        dry_run=dry_run,
    )

    # Step 10 — Verifica finale
    if not dry_run:
        log.info("\n── Step 10: Verifica finale ──")
        #verify_result(work_root, args.repo_id)
        verify_result(root)

    log.info("\n" + "=" * 60)
    if dry_run:
        log.info("DRY RUN completato — nessun file modificato")
        log.info(f"Dataset originale intatto in: {root}")
    else:
        log.info(f"Completato.")
        log.info(f"Dataset v3.0 pronto in:       {root}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()