"""
src/audio_io.py
===============
Fungsi untuk:
- Load audio (mono, resample ke target_sr)
- Normalisasi amplitudo per utterance
- Segmentasi audio menjadi frame-frame tetap

Semua parameter dibaca dari config YAML via argumen, bukan hardcode.
"""

from __future__ import annotations

import numpy as np
import librosa
from typing import List, Tuple


def load_audio(
    path: str,
    target_sr: int = 16000,
    min_duration_s: float = 1.0,
) -> Tuple[np.ndarray, int]:
    """
    Load file audio, konversi ke mono, resample ke target_sr,
    dan normalisasi amplitudo per utterance (peak normalization).

    Parameters
    ----------
    path        : path ke file audio (.wav / .flac / .mp3)
    target_sr   : sample rate target (default 16000)
    min_duration_s : durasi minimum dalam detik; raise ValueError bila terlalu pendek

    Returns
    -------
    x   : sinyal audio float32, sudah dinormalisasi
    sr  : sample rate (= target_sr)

    Raises
    ------
    ValueError  : sinyal mengandung NaN/Inf, atau lebih pendek dari min_duration_s
    """
    x, sr = librosa.load(path, sr=target_sr, mono=True)

    # Validasi
    if not np.isfinite(x).all():
        raise ValueError(f"Sinyal mengandung NaN/Inf: {path}")
    if len(x) < int(target_sr * min_duration_s):
        raise ValueError(
            f"Audio terlalu pendek ({len(x)/target_sr:.2f}s < {min_duration_s}s): {path}"
        )

    # Peak normalization per utterance
    peak = max(float(np.max(np.abs(x))), 1e-6)
    x = (x / peak).astype(np.float32)

    return x, sr


def segment_audio(
    x: np.ndarray,
    sr: int = 16000,
    seconds: float = 2.0,
    hop_seconds: float = 1.0,
) -> List[np.ndarray]:
    """
    Bagi sinyal x menjadi segmen dengan panjang tetap.
    Segmen terakhir di-pad dengan nol bila terlalu pendek.

    Parameters
    ----------
    x           : sinyal audio (1-D float32)
    sr          : sample rate
    seconds     : panjang setiap segmen dalam detik
    hop_seconds : jarak antar awal segmen

    Returns
    -------
    list of np.ndarray, setiap elemen panjang = int(sr * seconds)
    """
    frame_len = int(sr * seconds)
    hop_len   = int(sr * hop_seconds)

    # Pad bila sinyal lebih pendek dari satu frame
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))

    segments = []
    for start in range(0, len(x) - frame_len + 1, hop_len):
        seg = x[start : start + frame_len].copy()
        segments.append(seg)

    return segments


def load_and_segment(
    path: str,
    target_sr: int = 16000,
    seconds: float = 2.0,
    hop_seconds: float = 1.0,
    min_duration_s: float = 1.0,
) -> Tuple[List[np.ndarray], int]:
    """
    Convenience wrapper: load_audio + segment_audio.

    Returns
    -------
    (segments, sr)
    """
    x, sr = load_audio(path, target_sr=target_sr, min_duration_s=min_duration_s)
    segs  = segment_audio(x, sr=sr, seconds=seconds, hop_seconds=hop_seconds)
    return segs, sr


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path:
        segs, sr = load_and_segment(path)
        print(f"Loaded {len(segs)} segments @ {sr} Hz, each {segs[0].shape[0]} samples")
    else:
        print("Usage: python audio_io.py <path_to_audio>")
