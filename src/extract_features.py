"""
src/extract_features.py
========================
Script utama ekstraksi semua fitur dari manifest split.

Output:
    results/features_train.csv
    results/features_validation.csv
    results/features_test.csv
    results/features_external.csv  (bila ada)

Kolom per file:
    utterance_id, speaker_id, label, split, dataset,
    [mfcc_*], [lfcc_*],
    [res_energy_mean_p16, ..., res_kurtosis_median_p16],
    [mod_band_0.5_2, ..., mod_depth],
    residual_failure_rate

Cara pakai:
    python src/extract_features.py [--config configs/baseline.yaml]

Catatan:
    Untuk ablation LPC order, jalankan dengan --lpc_order 12 atau 20.
    Output akan disimpan ke results/features_p{order}_*.csv
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Tambah src ke path agar import relatif bisa berjalan
sys.path.insert(0, str(Path(__file__).parent))

from reproducibility import load_config, set_seed
from audio_io import load_and_segment
from spectral_features import mfcc_features, mfcc_feature_names
from lfcc import lfcc_features, lfcc_feature_names
from residual_features import (
    residual_feature_vector, residual_feature_names,
    frame_signal, residual_from_frame,
)
from modulation_features import modulation_features, modulation_feature_names


# ──────────────────────────────────────────────
# Per-utterance extraction
# ──────────────────────────────────────────────

def extract_utterance_features(
    file_path: str,
    cfg: dict,
    lpc_order: int = 16,
) -> dict | None:
    """
    Ekstraksi semua fitur untuk satu utterance.

    Strategi:
    - Load utterance lengkap
    - Segmentasi → ambil fitur per segmen
    - Agregasi segmen: mean (untuk baseline MFCC/LFCC dan modulasi)
    - Residual: dihitung per frame pada seluruh utterance

    Returns
    -------
    dict fitur, atau None bila file gagal dimuat
    """
    sr      = cfg["sample_rate"]
    seconds = cfg["segment_seconds"]
    hop_s   = cfg["hop_seconds"]
    n_mfcc  = cfg["mfcc_n"]
    n_lfcc  = cfg["lfcc_n"]

    try:
        segments, _ = load_and_segment(
            file_path,
            target_sr=sr,
            seconds=seconds,
            hop_seconds=hop_s,
            min_duration_s=1.0,
        )
    except Exception as e:
        print(f"  [SKIP] {file_path}: {e}")
        return None

    if len(segments) == 0:
        return None

    # ── MFCC: rata-rata atas semua segmen ──
    mfcc_vecs = []
    for seg in segments:
        try:
            v = mfcc_features(
                seg, sr=sr, n_mfcc=n_mfcc,
                n_fft=cfg["mfcc_n_fft"],
                hop_length=cfg["mfcc_hop_length"],
                win_length=cfg["mfcc_win_length"],
            )
            mfcc_vecs.append(v)
        except Exception:
            pass
    mfcc_feat = np.nanmean(mfcc_vecs, axis=0) if mfcc_vecs else np.full(4 * n_mfcc * 3, np.nan)

    # ── LFCC: rata-rata atas semua segmen ──
    lfcc_vecs = []
    for seg in segments:
        try:
            v = lfcc_features(
                seg, sr=sr, n_lfcc=n_lfcc,
                n_filters=cfg["lfcc_n_filters"],
                fmax=cfg["lfcc_fmax"],
            )
            lfcc_vecs.append(v)
        except Exception:
            pass
    lfcc_feat = np.nanmean(lfcc_vecs, axis=0) if lfcc_vecs else np.full(2 * n_lfcc, np.nan)

    # ── Residual: hitung pada segmen pertama (perwakilan) ──
    # Gunakan semua segmen dan rata-rata
    res_vecs = []
    for seg in segments:
        try:
            v = residual_feature_vector(
                seg,
                order=lpc_order,
                frame_ms=cfg["lpc_frame_ms"],
                hop_ms=cfg["lpc_hop_ms"],
                sr=sr,
                min_energy=cfg["lpc_min_energy"],
            )
            res_vecs.append(v)
        except Exception:
            pass

    if res_vecs:
        res_feat = np.nanmean(res_vecs, axis=0)
        failure_rate = float(np.isnan(np.array(res_vecs)).all(axis=1).mean())
    else:
        res_feat = np.full(15, np.nan)
        failure_rate = 1.0

    # ── Modulasi: rata-rata atas semua segmen ──
    mod_vecs = []
    for seg in segments:
        try:
            v = modulation_features(
                seg, sr=sr,
                envelope_sr=cfg["envelope_sr"],
                lowpass_hz=cfg["modulation_lowpass_hz"],
                max_hz=cfg["modulation_max_hz"],
            )
            mod_vecs.append(v)
        except Exception:
            pass
    mod_feat = np.nanmean(mod_vecs, axis=0) if mod_vecs else np.full(7, np.nan)

    # ── Gabung semua fitur ──
    feat_dict = {}

    mfcc_names = mfcc_feature_names(n_mfcc)
    for name, val in zip(mfcc_names, mfcc_feat):
        feat_dict[name] = float(val)

    lfcc_names = lfcc_feature_names(n_lfcc)
    for name, val in zip(lfcc_names, lfcc_feat):
        feat_dict[name] = float(val)

    res_names = residual_feature_names(str(lpc_order))
    for name, val in zip(res_names, res_feat):
        feat_dict[name] = float(val)

    mod_names = modulation_feature_names()
    for name, val in zip(mod_names, mod_feat):
        feat_dict[name] = float(val)

    feat_dict["residual_failure_rate"] = failure_rate
    feat_dict["n_segments"] = len(segments)

    return feat_dict


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def extract_all_features(
    manifest_csv: str = "manifests/split_manifest.csv",
    results_dir: str = "results",
    config_path: str = "configs/baseline.yaml",
    lpc_order: int = 16,
    smoke_test: bool = False,
    smoke_n: int = 100,
) -> None:
    """
    Ekstraksi fitur untuk semua utterance di manifest.

    Parameters
    ----------
    manifest_csv : path ke split_manifest.csv
    results_dir  : folder output CSV fitur
    config_path  : path ke YAML config
    lpc_order    : orde LPC (default 16; ablation: 12 atau 20)
    smoke_test   : bila True, hanya proses smoke_n utterance pertama
    smoke_n      : jumlah utterance untuk smoke test
    """
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    df = pd.read_csv(manifest_csv)
    print(f"[extract_features] Loaded {len(df)} utterances from {manifest_csv}")

    if smoke_test:
        df = df.head(smoke_n).copy()
        print(f"[extract_features] Smoke test mode: using {len(df)} utterances")

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_p{lpc_order}" if lpc_order != 16 else ""

    # Proses per split
    splits = df["split"].unique()
    for split_name in splits:
        df_split = df[df["split"] == split_name].copy()
        print(f"\n[extract_features] Processing split='{split_name}' ({len(df_split)} utterances)...")

        rows = []
        failed = 0

        for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=split_name):
            feat = extract_utterance_features(
                file_path=row["file_path"],
                cfg=cfg,
                lpc_order=lpc_order,
            )
            if feat is None:
                failed += 1
                continue

            # Tambahkan metadata
            entry = {
                "utterance_id": row["utterance_id"],
                "speaker_id":   row["speaker_id"],
                "label":        int(row["label"]),
                "split":        split_name,
                "dataset":      row.get("dataset", "unknown"),
                "generator_id": row.get("generator_id", "unknown"),
            }
            entry.update(feat)
            rows.append(entry)

        if rows:
            df_out = pd.DataFrame(rows)
            out_csv = out_dir / f"features{suffix}_{split_name}.csv"
            df_out.to_csv(str(out_csv), index=False)
            print(f"  Saved {len(df_out)} rows → {out_csv}  (failed: {failed})")

            # Laporan NaN
            nan_cols = df_out.isnull().sum()
            nan_cols = nan_cols[nan_cols > 0]
            if len(nan_cols) > 0:
                print(f"  NaN summary:\n{nan_cols.to_string()}")
        else:
            print(f"  [WARNING] No valid utterances for split '{split_name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract all features from split manifest")
    parser.add_argument("--manifest",    default="manifests/split_manifest.csv")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--config",      default="configs/baseline.yaml")
    parser.add_argument("--lpc_order",   type=int, default=16, choices=[12, 16, 20])
    parser.add_argument("--smoke_test",  action="store_true", help="Test dengan 100 utterance")
    parser.add_argument("--smoke_n",     type=int, default=100)
    args = parser.parse_args()

    extract_all_features(
        manifest_csv = args.manifest,
        results_dir  = args.results_dir,
        config_path  = args.config,
        lpc_order    = args.lpc_order,
        smoke_test   = args.smoke_test,
        smoke_n      = args.smoke_n,
    )
