"""
src/make_splits.py
==================
Bagi manifest menjadi train / validation / test berdasarkan speaker_id
menggunakan GroupShuffleSplit (tanpa speaker leakage).

Alur:
1. Baca manifests/source_manifest.csv
2. GroupShuffleSplit → (trainval, test) berdasarkan speaker
3. GroupShuffleSplit lagi pada trainval → (train, val)
4. Assign kolom 'split' dan simpan ke manifests/split_manifest.csv

Cara pakai:
    python src/make_splits.py [--config configs/baseline.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from reproducibility import load_config, set_seed


def make_splits(
    source_csv: str = "manifests/source_manifest.csv",
    out_csv:    str = "manifests/split_manifest.csv",
    config_path: str = "configs/baseline.yaml",
) -> pd.DataFrame:
    """
    Bagi dataset menjadi train/validation/test tanpa speaker leakage.

    Aturan (dari panduan):
    - Pembagian dilakukan pada tingkat utterance / speaker sebelum segmentasi
    - Scaler, feature selection, dan threshold HANYA dipelajari dari train
    - ASVspoof 2021 DF TIDAK dimasukkan ke split ini (hanya DEEP_VOICE)
    """
    cfg = load_config(config_path)
    seed = cfg["seed"]
    set_seed(seed)

    df = pd.read_csv(source_csv)
    print(f"[make_splits] Loaded {len(df)} utterances from {source_csv}")

    # Hanya gunakan dataset utama untuk split development
    df_dev = df[df["dataset"] == "DEEP_VOICE"].copy()
    df_ext = df[df["dataset"] != "DEEP_VOICE"].copy()
    print(f"  Dev set: {len(df_dev)} | External (no split): {len(df_ext)}")

    # Validasi
    missing_label = (df_dev["label"] == "FILL_REQUIRED").sum()
    if missing_label > 0:
        raise ValueError(
            f"{missing_label} baris memiliki label='FILL_REQUIRED'. "
            "Lengkapi manifest terlebih dahulu."
        )

    # Fallback jika jumlah speaker terlalu sedikit
    n_speakers = df_dev["speaker_id"].nunique()
    
    if n_speakers < 5:
        print(f"  [WARNING] Hanya {n_speakers} speaker unik. Fallback ke random split (tanpa grouping).")
        trainval, test = train_test_split(
            df_dev, test_size=cfg.get("test_size", 0.2), random_state=seed, stratify=df_dev["label"]
        )
        train, val = train_test_split(
            trainval, test_size=cfg.get("val_size", 0.2), random_state=seed + 1, stratify=trainval["label"]
        )
    else:
        # Langkah 1: Split trainval vs test (20% test)
        gss1 = GroupShuffleSplit(
            n_splits=1, test_size=cfg.get("test_size", 0.2), random_state=seed
        )
        train_idx, test_idx = next(
            gss1.split(df_dev, y=df_dev["label"], groups=df_dev["speaker_id"])
        )
        trainval = df_dev.iloc[train_idx].copy()
        test     = df_dev.iloc[test_idx].copy()

        # Langkah 2: Split train vs val (20% dari trainval)
        gss2 = GroupShuffleSplit(
            n_splits=1, test_size=cfg.get("val_size", 0.2), random_state=seed + 1
        )
        tr_idx, va_idx = next(
            gss2.split(trainval, y=trainval["label"], groups=trainval["speaker_id"])
        )
        train = trainval.iloc[tr_idx].copy()
        val   = trainval.iloc[va_idx].copy()

    # Assign split label
    train["split"] = "train"
    val["split"]   = "validation"
    test["split"]  = "test"
    if len(df_ext) > 0:
        df_ext["split"] = "external"

    out = pd.concat([train, val, test, df_ext], ignore_index=True)

    # Ringkasan
    for sp in ["train", "validation", "test"]:
        sub = out[out["split"] == sp]
        n_spk = sub["speaker_id"].nunique()
        n_real = (sub["label"] == 0).sum()
        n_fake = (sub["label"] == 1).sum()
        print(f"  {sp:12s}: {len(sub):5d} utterances | {n_spk} speakers | real={n_real} fake={n_fake}")

    # Simpan
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[make_splits] Saved → {out_csv}")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make group-based train/val/test splits")
    parser.add_argument("--source",  default="manifests/source_manifest.csv")
    parser.add_argument("--out",     default="manifests/split_manifest.csv")
    parser.add_argument("--config",  default="configs/baseline.yaml")
    args = parser.parse_args()

    make_splits(
        source_csv  = args.source,
        out_csv     = args.out,
        config_path = args.config,
    )
