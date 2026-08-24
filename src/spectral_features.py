"""
src/spectral_features.py
=========================
Ekstraksi fitur spektral:
  - MFCC (+ delta + delta²)  -> 20*4 * 3 = 240 dimensi
  - Statistik: mean, std, Q25, Q75 per koefisien

Semua parameter dibaca dari config YAML.
"""

from __future__ import annotations

import numpy as np
import librosa


# ──────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────

def summarize_matrix(mat: np.ndarray) -> np.ndarray:
    """
    Hitung statistik agregat dari matriks fitur (n_coef × n_frames).
    Output: vektor [mean, std, Q25, Q75] per koefisien -> panjang 4 * n_coef.
    """
    return np.concatenate([
        np.mean(mat, axis=1),
        np.std(mat, axis=1),
        np.percentile(mat, 25, axis=1),
        np.percentile(mat, 75, axis=1),
    ]).astype(np.float32)


# ──────────────────────────────────────────────
# MFCC
# ──────────────────────────────────────────────

def mfcc_features(
    x: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 20,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
) -> np.ndarray:
    """
    Ekstraksi MFCC + delta + delta2, dirangkum dengan statistik.

    Output dimensi: 4 * n_mfcc * 3  (mean/std/Q25/Q75 × n_mfcc × 3 order)
    Default: 4 * 20 * 3 = 240 fitur
    """
    m = librosa.feature.mfcc(
        y=x, sr=sr, n_mfcc=n_mfcc,
        n_fft=n_fft, hop_length=hop_length, win_length=win_length,
    )
    delta  = librosa.feature.delta(m)
    delta2 = librosa.feature.delta(m, order=2)

    feat = np.concatenate([
        summarize_matrix(m),
        summarize_matrix(delta),
        summarize_matrix(delta2),
    ])
    return feat.astype(np.float32)


def mfcc_feature_names(n_mfcc: int = 20) -> list[str]:
    """Nama fitur MFCC untuk kolom DataFrame."""
    names = []
    for prefix in ["mfcc", "mfcc_d1", "mfcc_d2"]:
        for stat in ["mean", "std", "q25", "q75"]:
            for i in range(n_mfcc):
                names.append(f"{prefix}_{stat}_{i:02d}")
    return names


if __name__ == "__main__":
    # Quick sanity check
    dummy = np.random.randn(32000).astype(np.float32)  # 2 detik @ 16kHz
    feat  = mfcc_features(dummy)
    names = mfcc_feature_names()
    print(f"MFCC feature vector: {feat.shape[0]} dims")
    print(f"Feature names: {len(names)}")
    assert feat.shape[0] == len(names), "Dimensi tidak cocok!"
    print("OK - MFCC features verified.")
