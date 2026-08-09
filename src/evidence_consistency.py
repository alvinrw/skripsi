"""
src/evidence_consistency.py
============================
Analisis konsistensi antara jalur klasifikasi (baseline MFCC/LFCC) dan
jalur bukti forensik (residual-modulasi).

Tahapan (dari panduan):
    1. Baca skor baseline dan evidence dari utterance_scores.csv
    2. Spearman correlation antara skor baseline dan skor evidence
    3. Cohen's Kappa — agreement label pada threshold validasi
    4. Analisis 4 kategori kasus:
        a. Baseline benar & bukti benar   (consistent correct)
        b. Baseline benar & bukti salah   (baseline-only correct)
        c. Baseline salah & bukti benar   (evidence-only correct)
        d. Baseline salah & bukti salah   (both wrong)
    5. Error analysis: list kasus dengan metadata

Output:
    results/evidence_consistency.csv
    results/error_analysis.csv

Cara pakai:
    python src/evidence_consistency.py \
        --scores results/utterance_scores.csv \
        --baseline_col score_B2 \
        --evidence_col score_E4c \
        --baseline_threshold 0.5 \
        --evidence_threshold 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


# ──────────────────────────────────────────────
# Main consistency analysis
# ──────────────────────────────────────────────

def analyze_consistency(
    scores_csv: str = "results/utterance_scores.csv",
    baseline_col: str = "score_B2",
    evidence_col: str = "score_E4c",
    baseline_threshold: float = 0.5,
    evidence_threshold: float = 0.5,
    out_consistency: str = "results/evidence_consistency.csv",
    out_error: str = "results/error_analysis.csv",
    split: str = "test",
) -> dict:
    """
    Analisis konsistensi dua jalur: baseline vs evidence.

    Returns
    -------
    dict ringkasan hasil konsistensi
    """
    df = pd.read_csv(scores_csv)

    if "split" in df.columns:
        df = df[df["split"] == split].copy()
        print(f"[consistency] Using split='{split}': {len(df)} utterances")
    else:
        print(f"[consistency] No 'split' column. Using all {len(df)} rows.")

    # Validasi kolom
    for col in [baseline_col, evidence_col, "label"]:
        if col not in df.columns:
            available = df.columns.tolist()
            raise ValueError(f"Kolom '{col}' tidak ditemukan. Tersedia: {available}")

    df = df.dropna(subset=[baseline_col, evidence_col]).copy()
    print(f"  After dropna: {len(df)} utterances")

    y_true         = df["label"].values
    baseline_score = df[baseline_col].values
    evidence_score = df[evidence_col].values

    # ── Spearman correlation ──
    rho, p_rho = spearmanr(baseline_score, evidence_score)

    # ── Label berdasarkan threshold ──
    baseline_pred = (baseline_score >= baseline_threshold).astype(int)
    evidence_pred = (evidence_score >= evidence_threshold).astype(int)

    # ── Cohen's Kappa ──
    kappa = cohen_kappa_score(baseline_pred, evidence_pred)

    # ── Kategorisasi 4 kasus ──
    b_correct = (baseline_pred == y_true).astype(int)
    e_correct = (evidence_pred == y_true).astype(int)

    df["baseline_correct"] = b_correct
    df["evidence_correct"]  = e_correct
    df["baseline_pred"]     = baseline_pred
    df["evidence_pred"]     = evidence_pred

    def categorize(bc, ec) -> str:
        if bc == 1 and ec == 1: return "consistent_correct"
        if bc == 1 and ec == 0: return "baseline_only_correct"
        if bc == 0 and ec == 1: return "evidence_only_correct"
        return "both_wrong"

    df["consistency_category"] = [
        categorize(bc, ec)
        for bc, ec in zip(b_correct, e_correct)
    ]

    # ── Ringkasan ──
    cat_counts = df["consistency_category"].value_counts().to_dict()
    n = len(df)

    results = {
        "split":              split,
        "n_utterances":       n,
        "baseline_col":       baseline_col,
        "evidence_col":       evidence_col,
        "baseline_threshold": baseline_threshold,
        "evidence_threshold": evidence_threshold,
        "spearman_rho":       round(float(rho), 5),
        "spearman_p":         round(float(p_rho), 6),
        "cohen_kappa":        round(float(kappa), 5),
        "consistent_correct":      cat_counts.get("consistent_correct", 0),
        "baseline_only_correct":   cat_counts.get("baseline_only_correct", 0),
        "evidence_only_correct":   cat_counts.get("evidence_only_correct", 0),
        "both_wrong":              cat_counts.get("both_wrong", 0),
        "pct_consistent_correct":  round(cat_counts.get("consistent_correct", 0) / n * 100, 2),
        "pct_both_wrong":          round(cat_counts.get("both_wrong", 0) / n * 100, 2),
    }

    # ── Print ──
    print(f"\n{'='*55}")
    print(f"Consistency Analysis: {baseline_col} vs {evidence_col}")
    print(f"{'='*55}")
    print(f"  Spearman ρ       : {results['spearman_rho']:.4f}  (p={results['spearman_p']:.4f})")
    print(f"  Cohen's κ        : {results['cohen_kappa']:.4f}")
    print(f"\n  Category breakdown (n={n}):")
    for cat in ["consistent_correct", "baseline_only_correct", "evidence_only_correct", "both_wrong"]:
        cnt = cat_counts.get(cat, 0)
        print(f"    {cat:30s}: {cnt:4d}  ({cnt/n*100:.1f}%)")

    # ── Simpan ──
    Path(out_consistency).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([results]).to_csv(out_consistency, index=False)
    print(f"\n  Saved consistency summary → {out_consistency}")

    # Error analysis: kasus dengan keputusan salah
    error_df = df[df["consistency_category"] != "consistent_correct"].copy()
    cols_keep = ["utterance_id", "speaker_id", "label",
                 baseline_col, evidence_col,
                 "baseline_pred", "evidence_pred",
                 "baseline_correct", "evidence_correct",
                 "consistency_category"]
    cols_keep = [c for c in cols_keep if c in error_df.columns]
    error_df[cols_keep].to_csv(out_error, index=False)
    print(f"  Saved error analysis  → {out_error}  ({len(error_df)} cases)")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evidence consistency analysis")
    parser.add_argument("--scores",               default="results/utterance_scores.csv")
    parser.add_argument("--baseline_col",         default="score_B2")
    parser.add_argument("--evidence_col",         default="score_E4c")
    parser.add_argument("--baseline_threshold",   type=float, default=0.5)
    parser.add_argument("--evidence_threshold",   type=float, default=0.5)
    parser.add_argument("--out_consistency",      default="results/evidence_consistency.csv")
    parser.add_argument("--out_error",            default="results/error_analysis.csv")
    parser.add_argument("--split",                default="test")
    args = parser.parse_args()

    analyze_consistency(
        scores_csv          = args.scores,
        baseline_col        = args.baseline_col,
        evidence_col        = args.evidence_col,
        baseline_threshold  = args.baseline_threshold,
        evidence_threshold  = args.evidence_threshold,
        out_consistency     = args.out_consistency,
        out_error           = args.out_error,
        split               = args.split,
    )
