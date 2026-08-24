"""
src/bootstrap_difference.py
============================
Bootstrap paired AUC difference test antar model.

Gunakan untuk membandingkan apakah model A secara statistik lebih baik
dari model B (menggunakan sampel utterance yang SAMA).

Dari panduan:
    - Hindari klaim "lebih baik" bila confidence interval selisih melintasi nol
    - Pisahkan analisis confirmatory (direncanakan) dari eksplorasi tambahan
    - Laporkan mean dan std untuk tiga seed atau seluruh fold

Cara pakai:
    python src/bootstrap_difference.py \
        --scores results/utterance_scores.csv \
        --col_a score_B2 --col_b score_E4c \
        --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ──────────────────────────────────────────────
# Bootstrap AUC difference
# ──────────────────────────────────────────────

def bootstrap_auc_diff(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 2026,
    alpha: float = 0.05,
) -> dict:
    """
    Bootstrap paired AUC difference test.

    H0: AUC(A) == AUC(B)
    H1: AUC(A) != AUC(B)

    Parameters
    ----------
    y        : label ground truth
    score_a  : skor model A
    score_b  : skor model B
    n_boot   : jumlah bootstrap iterasi
    seed     : random seed
    alpha    : level signifikansi (default 0.05 -> 95% CI)

    Returns
    -------
    dict: auc_a, auc_b, diff_point, ci_lo, ci_hi, significant
    """
    y       = np.asarray(y)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)

    # Point estimates
    auc_a = float(roc_auc_score(y, score_a))
    auc_b = float(roc_auc_score(y, score_b))
    diff_point = auc_a - auc_b

    # Bootstrap
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(y)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]

        # Butuh setidaknya dua kelas dalam bootstrap sample
        if len(np.unique(y_b)) < 2:
            continue

        d = roc_auc_score(y_b, score_a[idx]) - roc_auc_score(y_b, score_b[idx])
        diffs.append(d)

    diffs = np.array(diffs)
    ci_lo, ci_hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # Significant bila CI tidak melintasi nol
    significant = bool(ci_lo > 0 or ci_hi < 0)

    return {
        "auc_a":        round(auc_a, 5),
        "auc_b":        round(auc_b, 5),
        "diff_point":   round(diff_point, 5),
        "ci_lo":        round(float(ci_lo), 5),
        "ci_hi":        round(float(ci_hi), 5),
        "n_boot_used":  len(diffs),
        "significant":  significant,
        "note":         "significant bila CI tidak melintasi nol",
    }


# ──────────────────────────────────────────────
# Convenience: compare multiple pairs
# ──────────────────────────────────────────────

def compare_models(
    scores_csv: str = "results/utterance_scores.csv",
    pairs: list[tuple[str, str]] | None = None,
    split: str = "test",
    n_boot: int = 2000,
    seed: int = 2026,
    out_csv: str = "results/bootstrap_comparisons.csv",
) -> pd.DataFrame:
    """
    Bandingkan beberapa pasang model secara bootstrap.

    Parameters
    ----------
    pairs : list of (col_a, col_b); bila None, buat semua kombinasi 2
    """
    df = pd.read_csv(scores_csv)

    if "split" in df.columns:
        df = df[df["split"] == split].copy()
        print(f"[bootstrap] Using split='{split}': {len(df)} utterances")

    df = df.dropna(subset=["label"]).copy()
    y = df["label"].values

    score_cols = [c for c in df.columns if c.startswith("score_")]

    if pairs is None:
        from itertools import combinations
        pairs = list(combinations(score_cols, 2))

    rows = []
    for col_a, col_b in pairs:
        if col_a not in df.columns or col_b not in df.columns:
            print(f"  [SKIP] {col_a} vs {col_b}: kolom tidak ditemukan")
            continue

        valid = df[[col_a, col_b]].dropna()
        if len(valid) < 10:
            print(f"  [SKIP] {col_a} vs {col_b}: terlalu sedikit data")
            continue

        result = bootstrap_auc_diff(
            y=y[df[[col_a, col_b]].notna().all(axis=1).values],
            score_a=df.loc[df[[col_a, col_b]].notna().all(axis=1), col_a].values,
            score_b=df.loc[df[[col_a, col_b]].notna().all(axis=1), col_b].values,
            n_boot=n_boot, seed=seed,
        )
        result["col_a"] = col_a
        result["col_b"] = col_b
        result["split"] = split
        rows.append(result)

        print(f"  {col_a} vs {col_b}: "
              f"AUC diff={result['diff_point']:+.4f}  "
              f"95%CI=[{result['ci_lo']:.4f}, {result['ci_hi']:.4f}]  "
              f"{'SIGNIFICANT' if result['significant'] else 'not significant'}")

    if rows:
        out = pd.DataFrame(rows)
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
        print(f"\n[bootstrap] Saved {len(out)} comparisons -> {out_csv}")
        return out

    return pd.DataFrame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap AUC difference test")
    parser.add_argument("--scores",   default="results/utterance_scores.csv")
    parser.add_argument("--col_a",    default=None, help="Kolom skor model A")
    parser.add_argument("--col_b",    default=None, help="Kolom skor model B")
    parser.add_argument("--split",    default="test")
    parser.add_argument("--n_boot",   type=int, default=2000)
    parser.add_argument("--seed",     type=int, default=2026)
    parser.add_argument("--out",      default="results/bootstrap_comparisons.csv")
    args = parser.parse_args()

    if args.col_a and args.col_b:
        pairs = [(args.col_a, args.col_b)]
    else:
        pairs = None  # semua kombinasi

    compare_models(
        scores_csv = args.scores,
        pairs      = pairs,
        split      = args.split,
        n_boot     = args.n_boot,
        seed       = args.seed,
        out_csv    = args.out,
    )
