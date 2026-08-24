"""
src/statistical_tests.py
========================
Analisis statistik bukti forensik (residual dan modulasi).

Tahapan (dari panduan):
    1. Uji normalitas (Shapiro-Wilk) — hanya informasi, tidak menentukan metode
    2. Uji Mann-Whitney U (two-sided) sebagai uji utama
    3. Effect size: rank-biserial correlation rb = 1 - 2U / (n1 * n2)
    4. Koreksi multiple comparisons: Benjamini-Hochberg FDR
    5. Bootstrap 95% CI untuk selisih median

Interpretasi (dari panduan):
    - Laporkan effect size (rb), BUKAN hanya p-value
    - Hasil non-signifikan tetap dilaporkan
    - Hasil negatif tetap ilmiah bila pipeline valid

Output:
    results/statistical_tests.csv

Cara pakai:
    python src/statistical_tests.py [--features_csv results/features_train.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, shapiro
from statsmodels.stats.multitest import multipletests


# ──────────────────────────────────────────────
# Core tests
# ──────────────────────────────────────────────

def rank_biserial_correlation(u: float, n1: int, n2: int) -> float:
    """
    Rank-biserial correlation (effect size untuk Mann-Whitney U).
    rb ∈ [-1, 1]:
        rb = 0    -> tidak ada perbedaan
        |rb| = 1  -> total separation
        |rb| ≥ 0.3 -> kecil; ≥ 0.5 -> sedang; ≥ 0.7 -> besar
    """
    return float(1.0 - 2.0 * u / (n1 * n2))


def bootstrap_median_diff_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 2000,
    seed: int = 2026,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """
    Bootstrap 95% CI untuk selisih median (median_real - median_fake).

    Returns (lower, median_diff, upper)
    """
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        bx = rng.choice(x, size=len(x), replace=True)
        by = rng.choice(y, size=len(y), replace=True)
        diffs.append(np.median(bx) - np.median(by))
    lo, med, hi = np.percentile(diffs, [100 * alpha / 2, 50, 100 * (1 - alpha / 2)])
    return float(lo), float(med), float(hi)


# ──────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────

def run_statistical_tests(
    features_csv: str = "results/features_train.csv",
    out_csv: str = "results/statistical_tests.csv",
    feature_prefixes: tuple[str, ...] = ("res_", "mod_"),
    n_boot: int = 2000,
    seed: int = 2026,
) -> pd.DataFrame:
    """
    Jalankan uji statistik untuk semua fitur bukti forensik.

    Parameters
    ----------
    features_csv     : CSV fitur dari split train (atau train+val)
    out_csv          : path output
    feature_prefixes : prefix kolom fitur yang diuji
    n_boot           : jumlah iterasi bootstrap CI
    seed             : random seed

    Returns
    -------
    DataFrame hasil uji statistik (sudah di-FDR-koreksi)
    """
    df = pd.read_csv(features_csv)
    print(f"[statistical_tests] Loaded {len(df)} rows from {features_csv}")

    features = [c for c in df.columns if c.startswith(feature_prefixes)]
    print(f"  Testing {len(features)} features...")

    real_df = df[df["label"] == 0]
    fake_df = df[df["label"] == 1]
    print(f"  Real: {len(real_df)} | Fake: {len(fake_df)}")

    rows = []
    for f in features:
        real = real_df[f].dropna().values
        fake = fake_df[f].dropna().values

        if len(real) < 5 or len(fake) < 5:
            print(f"  [SKIP] {f}: insufficient data (real={len(real)}, fake={len(fake)})")
            continue

        # Shapiro-Wilk (informasi saja, max 5000 sampel)
        try:
            sw_real = float(shapiro(real[:5000]).pvalue)
            sw_fake = float(shapiro(fake[:5000]).pvalue)
        except Exception:
            sw_real = sw_fake = np.nan

        # Mann-Whitney U
        u_stat, p_val = mannwhitneyu(real, fake, alternative="two-sided")
        rb = rank_biserial_correlation(u_stat, len(real), len(fake))

        # Bootstrap CI selisih median
        ci_lo, ci_med, ci_hi = bootstrap_median_diff_ci(
            real, fake, n_boot=n_boot, seed=seed
        )

        rows.append({
            "feature":         f,
            "n_real":          len(real),
            "n_fake":          len(fake),
            "median_real":     round(float(np.median(real)), 6),
            "median_fake":     round(float(np.median(fake)), 6),
            "mean_real":       round(float(np.mean(real)), 6),
            "mean_fake":       round(float(np.mean(fake)), 6),
            "std_real":        round(float(np.std(real)), 6),
            "std_fake":        round(float(np.std(fake)), 6),
            "u_stat":          float(u_stat),
            "p_mwu":           float(p_val),
            "rank_biserial":   round(rb, 6),
            "abs_rb":          round(abs(rb), 6),
            "median_diff":     round(ci_med, 6),
            "ci_lo_95":        round(ci_lo, 6),
            "ci_hi_95":        round(ci_hi, 6),
            "sw_p_real":       round(sw_real, 6),
            "sw_p_fake":       round(sw_fake, 6),
        })

    if not rows:
        print("[statistical_tests] Tidak ada fitur yang diuji.")
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    # FDR correction (Benjamini-Hochberg)
    _, p_fdr, _, _ = multipletests(out["p_mwu"].values, method="fdr_bh")
    out["p_fdr"] = p_fdr
    out["significant_fdr"] = (out["p_fdr"] < 0.05)

    # Interpretasi effect size
    def interpret_rb(rb: float) -> str:
        a = abs(rb)
        if a >= 0.7:   return "large"
        if a >= 0.5:   return "medium"
        if a >= 0.3:   return "small"
        return "negligible"

    out["effect_size_label"] = out["rank_biserial"].apply(interpret_rb)

    # Urutkan berdasarkan |rb|
    out = out.sort_values("abs_rb", ascending=False).reset_index(drop=True)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[statistical_tests] Saved {len(out)} results -> {out_csv}")

    # Ringkasan
    n_sig = out["significant_fdr"].sum()
    print(f"\n  Significant (FDR < 0.05): {n_sig}/{len(out)}")
    print("\n  Top 10 by |rank_biserial|:")
    print(out[["feature", "rank_biserial", "p_fdr", "significant_fdr",
               "effect_size_label"]].head(10).to_string(index=False))

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Statistical tests for forensic evidence features")
    parser.add_argument("--features_csv", default="results/features_train.csv")
    parser.add_argument("--out",          default="results/statistical_tests.csv")
    parser.add_argument("--n_boot",       type=int, default=2000)
    parser.add_argument("--seed",         type=int, default=2026)
    args = parser.parse_args()

    run_statistical_tests(
        features_csv = args.features_csv,
        out_csv      = args.out,
        n_boot       = args.n_boot,
        seed         = args.seed,
    )
