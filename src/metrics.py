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
    accuracy_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


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
    threshold : threshold operasi (dari validation set). Bila None -> gunakan threshold EER.
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
    print(f"[metrics] Saved {len(df_new)} rows -> {out_csv}")


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, model_name: str) -> None:
    """Simpan gambar confusion matrix (PNG) dan tabel matriks (CSV)."""
    from pathlib import Path
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real (0)', 'Fake (1)'], 
                yticklabels=['Real (0)', 'Fake (1)'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save PNG image
    png_path = str(out_file) if str(out_file).endswith(".png") else f"{out_file}.png"
    plt.savefig(png_path)
    plt.close()
    
    # Save CSV table
    csv_path = png_path.replace(".png", ".csv")
    cm_df = pd.DataFrame(cm, index=['True_Real', 'True_Fake'], columns=['Pred_Real', 'Pred_Fake'])
    cm_df.to_csv(csv_path)
    print(f"[metrics] Confusion Matrix saved -> PNG: {png_path} | CSV: {csv_path}")


def save_prediction_mapping(
    df_meta: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    out_csv_path: str,
) -> pd.DataFrame:
    """
    Buat dan simpan CSV Mapping hasil prediksi per file audio.
    """
    from pathlib import Path
    
    mapping_df = df_meta.copy()
    if "file_path" in mapping_df.columns and "file_name" not in mapping_df.columns:
        mapping_df["file_name"] = mapping_df["file_path"].apply(lambda p: Path(str(p)).name)
        
    mapping_df["true_label"] = np.asarray(y_true)
    mapping_df["predicted_label"] = np.asarray(y_pred)
    mapping_df["confidence_score_fake"] = np.asarray(y_score)
    mapping_df["is_correct"] = (mapping_df["true_label"] == mapping_df["predicted_label"])
    
    out_p = Path(out_csv_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(str(out_p), index=False)
    print(f"[metrics] CSV Prediction Mapping saved -> {out_csv_path} ({len(mapping_df)} baris)")
    return mapping_df


def compute_per_generator_metrics(
    df_mapping: pd.DataFrame,
    out_csv_path: str | None = None,
) -> pd.DataFrame:
    """
    Hitung metrik per-generator (Voxcpm, OpenVoice, F5TTS, E2TTS, Real, dst).
    """
    if "generator_source" not in df_mapping.columns:
        print("[metrics] Kolom 'generator_source' tidak ditemukan untuk per-generator metrics.")
        return pd.DataFrame()
        
    results = []
    for gen, group in df_mapping.groupby("generator_source"):
        if len(group) == 0:
            continue
        y_t = group["true_label"].values
        y_s = group["confidence_score_fake"].values
        y_p = group["predicted_label"].values
        
        acc = float(accuracy_score(y_t, y_p))
        
        # Calculate AUC and EER if both classes present in group
        if len(np.unique(y_t)) > 1:
            try:
                auc = float(roc_auc_score(y_t, y_s))
                eer, _ = compute_eer(y_t, y_s)
            except Exception:
                auc, eer = np.nan, np.nan
        else:
            auc, eer = np.nan, np.nan
            
        p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
            y_t, y_p, average="macro", zero_division=0
        )
        
        results.append({
            "generator_source": gen,
            "n_samples": int(len(group)),
            "accuracy": round(acc, 5),
            "auc": round(auc, 5) if not np.isnan(auc) else None,
            "eer": round(eer, 5) if not np.isnan(eer) else None,
            "f1_macro": round(f1_mac, 5),
            "correct_count": int(group["is_correct"].sum()),
        })
        
    res_df = pd.DataFrame(results)
    if out_csv_path and not res_df.empty:
        from pathlib import Path
        p = Path(out_csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(str(p), index=False)
        print(f"[metrics] Breakdown per generator saved -> {out_csv_path}")
        
    return res_df


if __name__ == "__main__":
    # Quick sanity check dengan data random
    rng = np.random.default_rng(2026)
    y   = rng.integers(0, 2, size=200)
    s   = rng.random(size=200)  # random score

    eer, thr = compute_eer(y, s)
    print(f"Random EER: {eer:.4f}  (expected ~0.5)")

    m = compute_metrics(y, s, threshold=thr, split="test", model_id="random_baseline")
    print_metrics(m)

