"""
src/lfcc.py
===========
Ekstraksi Linear Frequency Cepstral Coefficients (LFCC).

Perbedaan MFCC vs LFCC:
- MFCC : filterbank mel-scale (log-spaced)
- LFCC : filterbank linear-scale (equally-spaced dalam Hz)

Implementasi:
1. STFT → power spectrum
2. Linear filterbank (equally spaced dari 0 Hz ke fmax)
3. Log compression
4. DCT → cepstral coefficients
5. Statistik mean & std per koefisien
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct
from scipy.signal import stft


# ──────────────────────────────────────────────
# Linear Filterbank
# ──────────────────────────────────────────────

def linear_filterbank(
    power: np.ndarray,
    freqs: np.ndarray,
    n_filters: int = 40,
    fmax: float = 8000.0,
) -> np.ndarray:
    """
    Buat linear filterbank dan terapkan ke power spectrum.

    Parameters
    ----------
    power    : (n_freq, n_frames) power spectrum dari STFT
    freqs    : (n_freq,) frekuensi bin dari STFT
    n_filters: jumlah filter linear
    fmax     : frekuensi maksimum analisis (Hz)

    Returns
    -------
    filterbank_output : (n_filters, n_frames)
    """
    # Titik tepi filter: equally spaced dari 0 Hz ke fmax
    edges = np.linspace(0, fmax, n_filters + 2)

    out = []
    for i in range(n_filters):
        left, center, right = edges[i], edges[i + 1], edges[i + 2]
        # Triangular window
        w = np.maximum(
            0.0,
            np.minimum(
                (freqs - left)   / (center - left   + 1e-9),
                (right  - freqs) / (right  - center + 1e-9),
            )
        )
        # Dot product: (n_freq,) · (n_freq, n_frames) → (n_frames,)
        out.append(np.sum(power * w[:, None], axis=0))

    return np.asarray(out, dtype=np.float32)  # (n_filters, n_frames)


# ──────────────────────────────────────────────
# LFCC
# ──────────────────────────────────────────────

def lfcc_features(
    x: np.ndarray,
    sr: int = 16000,
    n_lfcc: int = 20,
    n_filters: int = 40,
    fmax: float = 8000.0,
    nperseg: int = 400,
    noverlap: int = 240,
    nfft: int = 512,
) -> np.ndarray:
    """
    Ekstraksi LFCC dengan statistik mean dan std.

    Output dimensi: 2 * n_lfcc = 40 (default)

    Parameters
    ----------
    x        : sinyal audio 1-D float32
    sr       : sample rate
    n_lfcc   : jumlah LFCC yang diambil
    n_filters: jumlah filter dalam linear filterbank
    fmax     : frekuensi maksimum filterbank (Hz)
    nperseg  : panjang jendela STFT (sampel)
    noverlap : tumpang tindih STFT (sampel)
    nfft     : ukuran FFT
    """
    # STFT
    freqs, _, z = stft(
        x.astype(np.float64),
        fs=sr,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
    )

    power = np.abs(z) ** 2  # (n_freq, n_frames)

    # Linear filterbank
    fb = linear_filterbank(power, freqs, n_filters=n_filters, fmax=fmax)  # (n_filters, n_frames)

    # Log compression
    log_fb = np.log(fb + 1e-8)

    # DCT Type-II → ambil n_lfcc koefisien pertama
    cep = dct(log_fb, type=2, axis=0, norm="ortho")[:n_lfcc]  # (n_lfcc, n_frames)

    # Statistik agregat: mean + std per koefisien
    feat = np.concatenate([cep.mean(axis=1), cep.std(axis=1)]).astype(np.float32)
    return feat


def lfcc_feature_names(n_lfcc: int = 20) -> list[str]:
    """Nama fitur LFCC untuk kolom DataFrame."""
    names = []
    for stat in ["mean", "std"]:
        for i in range(n_lfcc):
            names.append(f"lfcc_{stat}_{i:02d}")
    return names


if __name__ == "__main__":
    dummy = np.random.randn(32000).astype(np.float32)  # 2 detik @ 16kHz
    feat  = lfcc_features(dummy)
    names = lfcc_feature_names()
    print(f"LFCC feature vector: {feat.shape[0]} dims")
    assert feat.shape[0] == len(names), "Dimensi tidak cocok!"
    print("OK - LFCC features verified.")
