"""
src/modulation_features.py
==========================
Ekstraksi bukti modulasi dari envelope amplitudo.

Analisis modulasi menilai perubahan envelope dan ritme jangka menengah.

Alur (dari panduan):
    1. Analytical signal (Hilbert) -> envelope a(t)
    2. Lowpass filter 40 Hz -> smooth envelope
    3. Resample ke envelope_sr=100 Hz
    4. Kurangi mean -> de-mean envelope
    5. FFT dengan Hanning window -> modulation spectrum M(f_m)
    6. Fokus 0–20 Hz (dinamika temporal ujaran)

Fitur (7 dimensi):
    1. band_0.5_2   — energi pita 0.5–2 Hz   (ritme suku kata)
    2. band_2_4     — energi pita 2–4 Hz     (ritme bergetar)
    3. band_4_8     — energi pita 4–8 Hz
    4. band_8_20    — energi pita 8–20 Hz    (modulasi cepat)
    5. centroid     — pusat massa frekuensi modulasi
    6. mod_entropy  — entropi spektrum modulasi
    7. mod_depth    — std(env) / mean(|env|) (kedalaman modulasi)

    M(f_m) = |FFT{ a(t) - mean(a(t)) }|²
"""

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert, butter, sosfiltfilt, resample_poly
from scipy.stats import entropy as scipy_entropy


# ──────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────

def band_energy(
    freq: np.ndarray,
    power: np.ndarray,
    low: float,
    high: float,
) -> float:
    """
    Hitung energi relatif dalam pita frekuensi [low, high) Hz.
    Ternormalisasi terhadap total energi -> rentang [0, 1].
    """
    idx = (freq >= low) & (freq < high)
    return float(power[idx].sum() / (power.sum() + 1e-12))


# ──────────────────────────────────────────────
# Main extraction
# ──────────────────────────────────────────────

MODULATION_FEATURE_NAMES = [
    "mod_band_0.5_2",
    "mod_band_2_4",
    "mod_band_4_8",
    "mod_band_8_20",
    "mod_centroid",
    "mod_entropy",
    "mod_depth",
]


def modulation_features(
    x: np.ndarray,
    sr: int = 16000,
    envelope_sr: int = 100,
    lowpass_hz: float = 40.0,
    max_hz: float = 20.0,
    bands: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """
    Ekstraksi fitur modulasi dari sinyal audio.

    Parameters
    ----------
    x            : sinyal audio 1-D float32
    sr           : sample rate input
    envelope_sr  : sample rate setelah resample envelope (default 100 Hz)
    lowpass_hz   : cutoff lowpass filter pada envelope (Hz)
    max_hz       : batas atas analisis modulasi (Hz)
    bands        : list pita [(low, high), ...]; default sesuai panduan

    Returns
    -------
    np.ndarray shape (7,) — 4 band energies + centroid + entropy + depth
    """
    if bands is None:
        bands = [(0.5, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 20.0)]

    # 1. Analytical signal -> envelope
    env = np.abs(hilbert(x.astype(np.float64)))

    # 2. Lowpass filter (menghapus komponen cepat sebelum downsample)
    sos = butter(4, lowpass_hz, btype="lowpass", fs=sr, output="sos")
    env = sosfiltfilt(sos, env)

    # 3. Resample ke envelope_sr (misal 16000 Hz -> 100 Hz)
    # resample_poly memerlukan bilangan bulat up/down
    from math import gcd
    g = gcd(envelope_sr, sr)
    env = resample_poly(env, up=envelope_sr // g, down=sr // g)

    # 4. De-mean
    env = env - env.mean()

    # 5. FFT dengan Hanning window -> power spectrum
    win    = np.hanning(len(env))
    spec   = np.abs(np.fft.rfft(env * win)) ** 2
    freq   = np.fft.rfftfreq(len(env), d=1.0 / envelope_sr)

    # 6. Batasi ke 0 < f <= max_hz
    keep = (freq > 0) & (freq <= max_hz)
    freq = freq[keep]
    spec = spec[keep]

    if spec.sum() < 1e-20:
        # Envelope hampir nol -> return NaN
        return np.full(len(bands) + 3, np.nan, dtype=np.float32)

    # 7. Distribusi probabilitas untuk entropi
    p = spec / (spec.sum() + 1e-12)

    # Fitur band energi
    band_feats = [band_energy(freq, spec, lo, hi) for lo, hi in bands]

    # Centroid
    centroid = float((freq * p).sum())

    # Spectral entropy modulasi
    mod_entropy = float(scipy_entropy(p + 1e-12))

    # Modulation depth: std(env_full) / mean(|env_full|)
    env_full = np.abs(hilbert(x.astype(np.float64)))
    mod_depth = float(np.std(env_full) / (np.mean(np.abs(env_full)) + 1e-9))

    result = np.asarray(
        band_feats + [centroid, mod_entropy, mod_depth],
        dtype=np.float32,
    )
    return result


def modulation_feature_names() -> list[str]:
    """Nama fitur modulasi untuk kolom DataFrame."""
    return list(MODULATION_FEATURE_NAMES)


if __name__ == "__main__":
    dummy = np.random.randn(32000).astype(np.float32)  # 2 detik @ 16kHz
    feat  = modulation_features(dummy)
    names = modulation_feature_names()
    print(f"Modulation feature vector: {feat.shape[0]} dims")
    print(dict(zip(names, feat)))
    assert feat.shape[0] == len(names), "Dimensi tidak cocok!"
    print("OK - Modulation features verified.")
