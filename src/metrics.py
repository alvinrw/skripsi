"""
src/metrics.py
==============
Fungsi evaluasi performa model:
  - EER (Equal Error Rate) dan threshold EER
  - ROC-AUC
  - Precision, Recall, F1 (per kelas dan macro)
  - Accuracy
  - Bootstrap 95% CI untuk AUC dan EER

ATURAN:
    Threshold ditentukan pada validation set.
    Threshold TIDAK boleh dihitung ulang pada test set atau dataset eksternal.

Konvensi:
    label 0 = real/bonafide
    label 1 = deepfake
    Skor yang lebih besar = lebih deepfake
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
)


# ──────────────────────────────────────────────
# EER
# ──────────────────────────────────────────────

def compute_eer(
    y_true: np.ndarray,
    score: np.ndarray,
) -> tuple[float, float]:
    """
    Hitung Equal Error Rate (EER) dan threshold-nya.

    EER adalah titik di mana FPR ≈ FNR.

    Returns
    -------
    (eer, threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, score, pos_label=1)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float(0.5 * (fpr[idx] + fnr[idx]))
    return eer, float(thresholds[idx])


# ──────────────────────────────────────────────
# Full metric report
# ──────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float | None = None,
    split: str = "test",
    model_id: str = "model",
    seed: int = 2026,
) -> dict:
    """
    Hitung semua metrik utama.

    Parameters
    ----------
    y_true    : label ground truth (0/1)
    score     : skor deepfake (lebih besar = lebih deepfake)
    threshold : threshold operasi (dari validation set). Bila None → gunakan threshold EER.
    split     : nama split untuk logging
    model_id  : nama model untuk logging
    seed      : seed yang digunakan

    Returns
    -------
    dict dengan semua metrik
    """
    y_true = np.asarray(y_true)
    score  = np.asarray(score)

    auc = float(roc_auc_score(y_true, score))
    eer, eer_threshold = compute_eer(y_true, score)

    # Gunakan threshold yang diberikan atau EER threshold
    thr = threshold if threshold is not None else eer_threshold
    y_pred = (score >= thr).astype(int)

    acc = float(accuracy_score(y_true, y_pred))

    # Per kelas + macro
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    )
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    return {
        "model":        model_id,
        "split":        split,
        "seed":         seed,
        "auc":          round(auc, 5),
        "eer":          round(eer, 5),
        "eer_threshold": round(eer_threshold, 5),
        "used_threshold": round(thr, 5),
        "accuracy":     round(acc, 5),
        "precision_real":  round(float(p[0]), 5),
        "recall_real":     round(float(r[0]), 5),
        "f1_real":         round(float(f1[0]), 5),
        "precision_fake":  round(float(p[1]), 5),
        "recall_fake":     round(float(r[1]), 5),
        "f1_fake":         round(float(f1[1]), 5),
        "precision_macro": round(float(p_mac), 5),
        "recall_macro":    round(float(r_mac), 5),
        "f1_macro":        round(float(f1_mac), 5),
        "n_samples":    int(len(y_true)),
        "n_real":       int((y_true == 0).sum()),
        "n_fake":       int((y_true == 1).sum()),
    }


def print_metrics(metrics: dict) -> None:
    """Print ringkasan metrik ke stdout."""
    print(f"\n{'='*50}")
    print(f"Model : {metrics['model']} | Split: {metrics['split']} | Seed: {metrics['seed']}")
    print(f"{'='*50}")
    print(f"  AUC      : {metrics['auc']:.4f}")
    print(f"  EER      : {metrics['eer']:.4f}  (threshold={metrics['eer_threshold']:.4f})")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  F1 Fake  : {metrics['f1_fake']:.4f}  | F1 Real: {metrics['f1_real']:.4f}")
    print(f"  F1 Macro : {metrics['f1_macro']:.4f}")
    print(f"  Samples  : {metrics['n_samples']} (real={metrics['n_real']}, fake={metrics['n_fake']})")


def save_metrics(metrics_list: list[dict], out_csv: str) -> None:
    """Simpan list metrik ke CSV (append bila sudah ada)."""
    from pathlib import Path
    df_new = pd.DataFrame(metrics_list)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        df_old = pd.read_csv(str(out_path))
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv(str(out_path), index=False)
    print(f"[metrics] Saved {len(df_new)} rows → {out_csv}")


if __name__ == "__main__":
    # Quick sanity check dengan data random
    rng = np.random.default_rng(2026)
    y   = rng.integers(0, 2, size=200)
    s   = rng.random(size=200)  # random score

    eer, thr = compute_eer(y, s)
    print(f"Random EER: {eer:.4f}  (expected ~0.5)")

    m = compute_metrics(y, s, threshold=thr, split="test", model_id="random_baseline")
    print_metrics(m)
