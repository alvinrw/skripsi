"""
src/aggregate_scores.py
=======================
Agregasi skor segmen menjadi skor utterance.

Input : results/segment_scores.csv
         Kolom: utterance_id, speaker_id, label, split, score, [experiment_id]

Output: results/utterance_scores.csv
         Kolom: utterance_id, speaker_id, label, split, score_mean, score_median, n_segments

Catatan (dari panduan):
    - Metrik utama dilaporkan pada tingkat UTTERANCE
    - Skor segmen hanya untuk analisis lokal
    - Segmen dari utterance yang sama BUKAN sampel independen

Cara pakai:
    python src/aggregate_scores.py [--segment_csv results/segment_scores.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def aggregate_scores(
    segment_csv: str = "results/segment_scores.csv",
    out_csv: str = "results/utterance_scores_agg.csv",
    score_col: str = "score",
) -> pd.DataFrame:
    """
    Agregasi skor segmen ke tingkat utterance.

    Returns DataFrame dengan skor utterance.
    """
    if not Path(segment_csv).exists():
        raise FileNotFoundError(f"File tidak ditemukan: {segment_csv}")

    seg = pd.read_csv(segment_csv)
    print(f"[aggregate] Loaded {len(seg)} segment rows from {segment_csv}")

    if score_col not in seg.columns:
        raise ValueError(f"Kolom '{score_col}' tidak ada. Kolom tersedia: {seg.columns.tolist()}")

    group_keys = ["utterance_id", "speaker_id", "label", "split"]
    group_keys = [k for k in group_keys if k in seg.columns]

    utt = (
        seg.groupby(group_keys, as_index=False)
        .agg(
            score_mean   = (score_col, "mean"),
            score_median = (score_col, "median"),
            score_min    = (score_col, "min"),
            score_max    = (score_col, "max"),
            score_std    = (score_col, "std"),
            n_segments   = (score_col, "size"),
        )
    )

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    utt.to_csv(out_csv, index=False)
    print(f"[aggregate] Saved {len(utt)} utterance rows → {out_csv}")

    # Ringkasan per split
    if "split" in utt.columns:
        for sp in utt["split"].unique():
            sub = utt[utt["split"] == sp]
            print(f"  {sp}: {len(sub)} utterances | n_segments avg={sub['n_segments'].mean():.1f}")

    return utt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate segment scores to utterance level")
    parser.add_argument("--segment_csv", default="results/segment_scores.csv")
    parser.add_argument("--out",         default="results/utterance_scores_agg.csv")
    parser.add_argument("--score_col",   default="score")
    args = parser.parse_args()

    aggregate_scores(
        segment_csv = args.segment_csv,
        out_csv     = args.out,
        score_col   = args.score_col,
    )
