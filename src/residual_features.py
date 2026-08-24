"""
src/residual_features.py
=========================
Ekstraksi bukti residual berbasis prediksi linear (LPC).

Interpretasi (dari panduan):
    Residual LPC adalah estimasi komponen yang TIDAK dapat diprediksi oleh
    model all-pole. Disebut sebagai "excitation-related evidence" atau
    "prediction residual" — BUKAN glottal source murni.

    e[n] = x[n] + Σ(k=1..p) a_k * x[n-k]

Fitur per frame (5 fitur):
    1. residual_energy       — energi rata-rata residual
    2. residual_entropy      — Shannon entropy distribusi residual
    3. prediction_error      — rasio energi residual / energi sinyal
    4. excitation_irregularity — mean |Δe| / std(e)  (ketidakberaturan eksitasi)
    5. kurtosis              — kurtosis residual (peakedness)

Agregasi ke utterance: mean, std, median -> 15 fitur per utterance
Ablation LPC order: p ∈ {12, 16, 20}
"""

from __future__ import annotations

import numpy as np
import librosa
from scipy.signal import lfilter
from scipy.stats import kurtosis as scipy_kurtosis


# ──────────────────────────────────────────────
# Frame-level utilities
# ──────────────────────────────────────────────

def frame_signal(
    x: np.ndarray,
    frame_len: int = 400,
    hop_len: int = 160,
) -> np.ndarray:
    """
    Bagi sinyal menjadi frame overlapping.
    Returns (n_frames, frame_len) — copy aman.
    """
    frames = librosa.util.frame(x, frame_length=frame_len, hop_length=hop_len)
    return frames.T.copy()  # (n_frames, frame_len)


def shannon_entropy(x: np.ndarray, bins: int = 64) -> float:
    """Shannon entropy dari distribusi histogram residual."""
    hist, _ = np.histogram(x, bins=bins, density=True)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def residual_from_frame(
    frame: np.ndarray,
    order: int = 16,
    min_energy: float = 1e-8,
) -> np.ndarray | None:
    """
    Hitung residual LPC dari satu frame.

    Parameters
    ----------
    frame      : sinyal 1-D (satu frame)
    order      : orde LPC (p)
    min_energy : threshold energi minimum; return None bila frame terlalu senyap

    Returns
    -------
    residual 1-D float32, atau None bila frame dilewati
    """
    frame_win = frame * np.hamming(len(frame))
    energy = float(np.mean(frame_win ** 2))

    if energy < min_energy:
        return None  # frame terlalu senyap -> koefisien tidak stabil

    # Koefisien LPC menggunakan autocorrelation method (Levinson-Durbin)
    a = librosa.lpc(frame_win.astype(float), order=order)

    # Filter all-pole: e[n] = x[n] + a1*x[n-1] + ... + ap*x[n-p]
    # lfilter(b, a, x): b=koefisien LPC, a=[1.0] (FIR filter)
    residual = lfilter(a, [1.0], frame_win)

    return residual.astype(np.float32)


# ──────────────────────────────────────────────
# Utterance-level features
# ──────────────────────────────────────────────

RESIDUAL_FEATURE_NAMES_BASE = [
    "res_energy",
    "res_entropy",
    "res_pred_error",
    "res_irregularity",
    "res_kurtosis",
]


def residual_feature_vector(
    x: np.ndarray,
    order: int = 16,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    sr: int = 16000,
    min_energy: float = 1e-8,
) -> np.ndarray:
    """
    Ekstraksi fitur residual untuk satu utterance/segmen.

    Alur:
    1. Bagi sinyal menjadi frame (25ms / hop 10ms)
    2. Hitung residual LPC per frame
    3. Hitung 5 fitur per frame
    4. Agregasi: mean, std, median -> 15 fitur

    Parameters
    ----------
    x         : sinyal audio 1-D float32
    order     : orde LPC
    frame_ms  : panjang frame dalam milidetik
    hop_ms    : hop dalam milidetik
    sr        : sample rate
    min_energy: threshold energi frame minimum

    Returns
    -------
    np.ndarray shape (15,) — [mean×5, std×5, median×5]
    """
    frame_len = int(sr * frame_ms / 1000)
    hop_len   = int(sr * hop_ms / 1000)

    frames = frame_signal(x, frame_len=frame_len, hop_len=hop_len)

    feats = []
    n_failed = 0

    for frame in frames:
        e = residual_from_frame(frame, order=order, min_energy=min_energy)
        if e is None:
            n_failed += 1
            continue

        sig_energy = float(np.mean(frame ** 2))
        energy     = float(np.mean(e ** 2))
        pred_error = energy / (sig_energy + 1e-9)
        irregularity = float(np.mean(np.abs(np.diff(e)))) / (float(np.std(e)) + 1e-9)
        kurt       = float(scipy_kurtosis(e, fisher=False, bias=False))

        feats.append([energy, shannon_entropy(e), pred_error, irregularity, kurt])

    # Catat failure rate (opsional; dapat disimpan sebagai feature QC indicator)
    # failure_rate = n_failed / max(len(frames), 1)

    arr = np.asarray(feats, dtype=float)

    if len(arr) == 0:
        # Semua frame gagal -> return NaN vector
        return np.full(15, np.nan, dtype=np.float32)

    return np.concatenate([
        np.nanmean(arr, axis=0),
        np.nanstd(arr, axis=0),
        np.nanmedian(arr, axis=0),
    ]).astype(np.float32)


def residual_feature_names(order_suffix: str = "16") -> list[str]:
    """
    Nama fitur residual untuk kolom DataFrame.
    Suffix order digunakan dalam ablation (p12/p16/p20).
    """
    names = []
    for stat in ["mean", "std", "median"]:
        for base in RESIDUAL_FEATURE_NAMES_BASE:
            names.append(f"{base}_{stat}_p{order_suffix}")
    return names


# ──────────────────────────────────────────────
# Ablation helper
# ──────────────────────────────────────────────

def residual_features_ablation(
    x: np.ndarray,
    orders: list[int] = [12, 16, 20],
    **kwargs,
) -> dict[int, np.ndarray]:
    """
    Hitung residual features untuk beberapa orde LPC sekaligus.
    Returns dict: {order: feature_vector}
    """
    return {
        p: residual_feature_vector(x, order=p, **kwargs)
        for p in orders
    }


if __name__ == "__main__":
    dummy = np.random.randn(32000).astype(np.float32)  # 2 detik @ 16kHz
    for order in [12, 16, 20]:
        feat  = residual_feature_vector(dummy, order=order)
        names = residual_feature_names(str(order))
        print(f"LPC order={order}: {feat.shape[0]} dims | NaN={np.isnan(feat).sum()}")
        assert feat.shape[0] == 15 == len(names), "Dimensi tidak cocok!"
    print("OK - Residual features verified.")
