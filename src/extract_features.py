"""
src/extract_features.py
========================
Script utama ekstraksi semua fitur dari manifest split (Versi Turbo + Resume).
"""

from __future__ import annotations
import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

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

def resolve_existing_path(file_path: str) -> str:
    if not file_path:
        return file_path
    if os.path.exists(file_path):
        return file_path
    
    p = Path(file_path)
    filename = p.name
    parent_dir = p.parent.name
    
    candidates = [
        Path("/content/dataset_processed") / parent_dir / filename,
        Path("/content/VoxCPM_processed") / parent_dir / filename,
        Path("data/processed") / parent_dir / filename,
        Path("/content/dataset_processed/2s") / filename,
        Path("/content/VoxCPM_processed/2s") / filename,
        Path("data/processed/2s") / filename,
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
            
    return file_path

def extract_utterance_features(file_path: str, cfg: dict, lpc_order: int = 16) -> dict | None:
    file_path = resolve_existing_path(file_path)
    if not file_path or not os.path.exists(file_path):
        return None

    sr      = cfg["sample_rate"]
    seconds = cfg["segment_seconds"]
    hop_s   = cfg["hop_seconds"]
    n_mfcc  = cfg["mfcc_n"]
    n_lfcc  = cfg["lfcc_n"]

    try:
        segments, _ = load_and_segment(
            file_path, target_sr=sr, seconds=seconds,
            hop_seconds=hop_s, min_duration_s=1.0,
        )
    except Exception:
        return None

    if len(segments) == 0:
        return None


    # MFCC
    mfcc_vecs = []
    for seg in segments:
        try:
            v = mfcc_features(seg, sr=sr, n_mfcc=n_mfcc, n_fft=cfg["mfcc_n_fft"], hop_length=cfg["mfcc_hop_length"], win_length=cfg["mfcc_win_length"])
            mfcc_vecs.append(v)
        except Exception: pass
    mfcc_feat = np.nanmean(mfcc_vecs, axis=0) if mfcc_vecs else np.full(4 * n_mfcc * 3, np.nan)

    # LFCC
    lfcc_vecs = []
    for seg in segments:
        try:
            v = lfcc_features(seg, sr=sr, n_lfcc=n_lfcc, n_filters=cfg["lfcc_n_filters"], fmax=cfg["lfcc_fmax"])
            lfcc_vecs.append(v)
        except Exception: pass
    lfcc_feat = np.nanmean(lfcc_vecs, axis=0) if lfcc_vecs else np.full(2 * n_lfcc, np.nan)

    # Residual
    res_vecs = []
    for seg in segments:
        try:
            v = residual_feature_vector(seg, order=lpc_order, frame_ms=cfg["lpc_frame_ms"], hop_ms=cfg["lpc_hop_ms"], sr=sr, min_energy=cfg["lpc_min_energy"])
            res_vecs.append(v)
        except Exception: pass
    if res_vecs:
        res_feat = np.nanmean(res_vecs, axis=0)
        failure_rate = float(np.isnan(np.array(res_vecs)).all(axis=1).mean())
    else:
        res_feat = np.full(15, np.nan)
        failure_rate = 1.0

    # Modulasi
    mod_vecs = []
    for seg in segments:
        try:
            v = modulation_features(seg, sr=sr, envelope_sr=cfg["envelope_sr"], lowpass_hz=cfg["modulation_lowpass_hz"], max_hz=cfg["modulation_max_hz"])
            mod_vecs.append(v)
        except Exception: pass
    mod_feat = np.nanmean(mod_vecs, axis=0) if mod_vecs else np.full(7, np.nan)

    feat_dict = {}
    for name, val in zip(mfcc_feature_names(n_mfcc), mfcc_feat): feat_dict[name] = float(val)
    for name, val in zip(lfcc_feature_names(n_lfcc), lfcc_feat): feat_dict[name] = float(val)
    for name, val in zip(residual_feature_names(str(lpc_order)), res_feat): feat_dict[name] = float(val)
    for name, val in zip(modulation_feature_names(), mod_feat): feat_dict[name] = float(val)
    
    feat_dict["residual_failure_rate"] = failure_rate
    feat_dict["n_segments"] = len(segments)
    return feat_dict

def process_row(row, cfg, lpc_order, split_name):
    """Helper multiprocessing"""
    feat = extract_utterance_features(row["file_path"], cfg, lpc_order)
    if feat is None: return None
    entry = {
        "utterance_id":     row["utterance_id"],
        "file_path":        row.get("file_path", ""),
        "speaker_id":       row["speaker_id"],
        "label":            int(row["label_idx"]) if "label_idx" in row and pd.notna(row["label_idx"]) else (0 if str(row["label"]).lower()=="real" else 1),
        "split":            split_name,
        "generator_source": row.get("generator_source", "unknown"),
        "intent":           row.get("intent", "unknown"),
        "dataset":          row.get("dataset", "unknown"),
        "generator_id":     row.get("generator_id", "unknown"),
    }
    entry.update(feat)
    return entry


def extract_all_features(manifest_csv="manifests/split_manifest.csv", results_dir="results", config_path="configs/baseline.yaml", lpc_order=16, smoke_test=False, smoke_n=100, duration=None):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    if duration and manifest_csv == "manifests/split_manifest.csv":
        manifest_csv = f"manifests/split_manifest_{duration}.csv"

    manifest_paths = [Path(manifest_csv)]
    dur_suffix = f"_{duration}" if duration else ""
    sep_manifest_csv = Path(f"manifests/separate_test_manifest{dur_suffix}.csv")
    if sep_manifest_csv.exists():
        manifest_paths.append(sep_manifest_csv)

    for m_path in manifest_paths:
        if not m_path.exists():
            continue
        print(f"\n[extract] Loading manifest: {m_path}")
        df = pd.read_csv(m_path, on_bad_lines='skip')
        if smoke_test: df = df.head(smoke_n).copy()

        out_dir = Path(results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_p{lpc_order}" if lpc_order != 16 else ""

        for split_name in df["split"].unique():
            df_split = df[df["split"] == split_name].copy()
            out_csv = out_dir / f"features{suffix}{dur_suffix}_{split_name}.csv"
            
            # LOGIKA RESUME CHECKPOINT
            processed_ids = set()
            if out_csv.exists():
                try:
                    existing_df = pd.read_csv(out_csv, on_bad_lines='skip')
                    if "utterance_id" in existing_df.columns:
                        processed_ids = set(existing_df["utterance_id"].astype(str))
                        print(f"\n[RESUME] Menemukan {len(processed_ids)} data di {out_csv.name}, melanjutkan sisanya...")
                except Exception as e:
                    print(f"\n[WARNING] CSV {out_csv.name} tidak valid ({e}). Memulai ulang penulisan untuk split '{split_name}'...")
                    out_csv.unlink()
                    processed_ids = set()

            to_process = df_split[~df_split["utterance_id"].astype(str).isin(processed_ids)]
            
            if len(to_process) == 0:
                print(f"\n[SKIP] Semua {len(df_split)} data '{split_name}' sudah selesai.")
                continue

            print(f"\n[extract] Memproses '{split_name}': sisa {len(to_process)} data (Multiprocessing ON)...")
            
            # EKSEKUSI PARALEL
            results = Parallel(n_jobs=-1, batch_size="auto")(
                delayed(process_row)(row, cfg, lpc_order, split_name)
                for _, row in tqdm(to_process.iterrows(), total=len(to_process), desc=split_name)
            )

            valid_results = [r for r in results if r is not None]
            failed = len(results) - len(valid_results)

            if valid_results:
                df_out = pd.DataFrame(valid_results)
                if out_csv.exists():
                    df_out.to_csv(str(out_csv), mode='a', header=False, index=False)
                else:
                    df_out.to_csv(str(out_csv), index=False)
                print(f"  Tersimpan {len(df_out)} baris -> {out_csv} (gagal: {failed})")
            else:
                print(f"  [WARNING] Data corrupt/gagal diproses pada split '{split_name}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/split_manifest.csv")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--lpc_order", type=int, default=16)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--smoke_n", type=int, default=100)
    args = parser.parse_args()

    extract_all_features(args.manifest, args.results_dir, args.config, args.lpc_order, args.smoke_test, args.smoke_n)

