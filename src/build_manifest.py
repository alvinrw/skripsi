"""
src/build_manifest.py
=====================
Scan folder data/raw/, kumpulkan metadata setiap file audio,
hitung SHA256, dan tulis ke manifests/source_manifest.csv.

Kolom output:
    utterance_id, file_path, speaker_id, label, generator_id,
    dataset, split, duration_s, sample_rate, sha256

Cara pakai:
    python src/build_manifest.py --data_dir data/raw --out manifests/source_manifest.csv

CATATAN:
    - Kolom speaker_id, label, generator_id wajib diisi secara manual
      atau via --label_map (file CSV: file_path,label,speaker_id)
    - Field split diisi "UNASSIGNED" dan diisi oleh make_splits.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm


# ──────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────

def sha256_file(path: Path, block: int = 1024 * 1024) -> str:
    """Hitung SHA256 dari file (streaming, memory-efficient)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def infer_label_from_path(path: Path) -> int | str:
    """
    Coba inferensi label dari nama folder.
    Konvensi DEEP-VOICE: folder 'real' atau 'fake'/'deepfake'.
    Kembalikan 0 (real) atau 1 (fake) bila berhasil, else 'FILL_REQUIRED'.
    """
    parts = [p.lower() for p in path.parts]
    if "real" in parts or "bonafide" in parts:
        return 0
    if "fake" in parts or "deepfake" in parts or "spoof" in parts:
        return 1
    return "FILL_REQUIRED"


def infer_speaker_from_path(path: Path) -> str:
    """
    Coba inferensi speaker_id dari nama folder induk.
    Kembalikan nama folder induk sebagai speaker ID, atau 'FILL_REQUIRED'.
    """
    # Heuristic: folder satu level atas file
    return path.parent.name if path.parent.name else "FILL_REQUIRED"


# ──────────────────────────────────────────────
# Main builder
# ──────────────────────────────────────────────

def build_manifest(
    data_dir: str,
    out_csv: str,
    label_map_csv: str | None = None,
    dataset_name: str = "DEEP_VOICE",
    min_duration_s: float = 1.0,
) -> pd.DataFrame:
    """
    Scan data_dir secara rekursif untuk file .wav / .flac.
    Buat manifest CSV dengan metadata setiap utterance.

    Parameters
    ----------
    data_dir      : root folder audio mentah
    out_csv       : path output manifest CSV
    label_map_csv : (opsional) CSV dengan kolom file_path,label,speaker_id,generator_id
    dataset_name  : nama dataset (DEEP_VOICE / ASVspoof2021_DF)
    min_duration_s: file lebih pendek dari ini dilewati
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"data_dir tidak ditemukan: {data_dir}")

    # Muat label map bila tersedia
    label_map: dict = {}
    if label_map_csv and Path(label_map_csv).exists():
        lm = pd.read_csv(label_map_csv)
        for _, row in lm.iterrows():
            label_map[str(row["file_path"])] = row.to_dict()
        print(f"[build_manifest] Loaded label map: {len(label_map)} entries")

    audio_extensions = {".wav", ".flac", ".mp3", ".ogg"}
    audio_files = [
        p for p in data_path.rglob("*")
        if p.is_file() and p.suffix.lower() in audio_extensions
    ]
    print(f"[build_manifest] Found {len(audio_files)} audio files in {data_dir}")

    rows = []
    skipped = 0

    for path in tqdm(audio_files, desc="Building manifest"):
        rel_path = path.as_posix()  # gunakan path as-is (e.g. data/raw/...)

        try:
            info = sf.info(str(path))
        except Exception as e:
            print(f"  [SKIP] Cannot read {path}: {e}")
            skipped += 1
            continue

        if info.duration < min_duration_s:
            print(f"  [SKIP] Too short ({info.duration:.2f}s): {path}")
            skipped += 1
            continue

        # Ambil metadata dari label_map bila ada
        lm_entry = label_map.get(rel_path, label_map.get(path.name, {}))

        label        = lm_entry.get("label",        infer_label_from_path(path))
        speaker_id   = lm_entry.get("speaker_id",   infer_speaker_from_path(path))
        generator_id = lm_entry.get("generator_id", "unknown")

        rows.append({
            "utterance_id": path.stem,
            "file_path":    rel_path,
            "speaker_id":   speaker_id,
            "label":        label,
            "generator_id": generator_id,
            "dataset":      dataset_name,
            "split":        "UNASSIGNED",
            "duration_s":   round(info.duration, 4),
            "sample_rate":  info.samplerate,
            "sha256":       sha256_file(path),
        })

    df = pd.DataFrame(rows)

    # Peringatan duplikasi
    dup_hash = df["sha256"].duplicated().sum()
    if dup_hash > 0:
        print(f"  [WARNING] {dup_hash} duplicate SHA256 detected! Periksa file duplikat.")

    dup_uid = df["utterance_id"].duplicated().sum()
    if dup_uid > 0:
        print(f"  [WARNING] {dup_uid} duplicate utterance_id! Pertimbangkan gunakan path sebagai ID.")

    # Simpan
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out_path), index=False)
    print(f"[build_manifest] Saved {len(df)} rows → {out_csv}  (skipped: {skipped})")

    # Ringkasan
    if "label" in df.columns:
        label_counts = df["label"].value_counts().to_dict()
        print(f"  Label distribution: {label_counts}")

    return df


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build source manifest from audio files")
    parser.add_argument("--data_dir",     default="data/raw",                    help="Folder audio mentah")
    parser.add_argument("--out",          default="manifests/source_manifest.csv", help="Output CSV")
    parser.add_argument("--label_map",    default=None,                           help="CSV label map (opsional)")
    parser.add_argument("--dataset_name", default="DEEP_VOICE",                  help="Nama dataset")
    parser.add_argument("--min_duration", type=float, default=1.0,               help="Durasi minimum (detik)")
    args = parser.parse_args()

    build_manifest(
        data_dir     = args.data_dir,
        out_csv      = args.out,
        label_map_csv= args.label_map,
        dataset_name = args.dataset_name,
        min_duration_s = args.min_duration,
    )
