"""
src/check_leakage.py
====================
Validasi bahwa tidak ada utterance_id, sha256, atau speaker_id yang
melintasi lebih dari satu split (train / validation / test).

Harus dijalankan SETIAP KALI sebelum memulai pelatihan model.

Cara pakai:
    python src/check_leakage.py [--manifest manifests/split_manifest.csv]

Exit code:
    0  → tidak ada leakage
    1  → leakage terdeteksi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def check_leakage(manifest_csv: str = "manifests/split_manifest.csv") -> bool:
    """
    Periksa leakage pada manifest.

    Returns
    -------
    True  bila pipeline bersih (tidak ada leakage)
    False bila ada leakage (raise SystemExit(1) saat CLI)
    """
    if not Path(manifest_csv).exists():
        print(f"[check_leakage] File tidak ditemukan: {manifest_csv}")
        return False

    df = pd.read_csv(manifest_csv)

    # Hanya periksa split dev (train/validation/test), abaikan external
    dev_splits = {"train", "validation", "test"}
    df = df[df["split"].isin(dev_splits)].copy()
    print(f"[check_leakage] Checking {len(df)} rows (dev splits only)")

    ok = True
    keys_to_check = ["utterance_id", "sha256", "speaker_id"]

    for key in keys_to_check:
        if key not in df.columns:
            print(f"  [SKIP] Kolom '{key}' tidak ada di manifest")
            continue

        # Hitung berapa split unik yang dimiliki setiap nilai key
        overlap = df.groupby(key)["split"].nunique()
        bad = overlap[overlap > 1]

        if len(bad) > 0:
            ok = False
            print(f"  [FAIL] Leakage detected on '{key}': {len(bad)} entries cross splits")
            print(bad.head(10).to_string())
        else:
            print(f"  [OK  ] No leakage on '{key}'")

    # Periksa distribusi label per split
    print("\n[check_leakage] Label distribution per split:")
    summary = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(summary.to_string())

    # Periksa distribusi speaker per split
    print("\n[check_leakage] Speaker count per split:")
    spk_summary = df.groupby("split")["speaker_id"].nunique()
    print(spk_summary.to_string())

    if ok:
        print("\n[check_leakage] ✓ All leakage checks PASSED. Safe to train.")
    else:
        print("\n[check_leakage] ✗ LEAKAGE DETECTED. Fix split sebelum training!")

    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check data leakage in split manifest")
    parser.add_argument("--manifest", default="manifests/split_manifest.csv")
    args = parser.parse_args()

    clean = check_leakage(args.manifest)
    sys.exit(0 if clean else 1)
