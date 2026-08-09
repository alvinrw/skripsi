"""
src/train_baseline.py
=====================
Pelatihan model baseline dan evidence-only pada fitur yang sudah diekstraksi.

Eksperimen wajib (dari panduan):
    B0: MFCC             + SVM RBF           → baseline utama
    B1: LFCC             + SVM RBF           → representasi linear
    B2: MFCC + LFCC      + SVM RBF           → fusi spektral
    B3: MFCC + LFCC      + Random Forest     → non-linear pohon
    B4: MFCC + LFCC      + XGBoost           → boosting
    B5: MFCC + LFCC      + KNN/MLP           → target tambahan

    E4a: Residual-only   + SVM/RF/XGBoost    → ablation domain
    E4b: Modulation-only + SVM/RF/XGBoost    → ablation domain
    E4c: Residual+Mod    + SVM/RF/XGBoost    → evidence-only model

ATURAN:
    - Scaler HANYA difit pada training set
    - Threshold EER HANYA dihitung pada validation set
    - Model disimpan ke checkpoints/ beserta daftar fitur dan threshold

Cara pakai:
    python src/train_baseline.py [--config configs/baseline.yaml] [--lpc_order 16]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent))

from reproducibility import load_config, set_seed
from metrics import compute_metrics, compute_eer, print_metrics, save_metrics


# ──────────────────────────────────────────────
# Feature group definitions
# ──────────────────────────────────────────────

def get_feature_groups(df: pd.DataFrame, lpc_order: int = 16) -> dict[str, list[str]]:
    """
    Tentukan grup fitur berdasarkan prefix kolom.
    """
    all_cols = df.columns.tolist()

    mfcc_cols = [c for c in all_cols if c.startswith("mfcc_")]
    lfcc_cols = [c for c in all_cols if c.startswith("lfcc_")]
    res_cols  = [c for c in all_cols if c.startswith("res_") and f"_p{lpc_order}" in c]
    mod_cols  = [c for c in all_cols if c.startswith("mod_")]

    return {
        "mfcc":        mfcc_cols,
        "lfcc":        lfcc_cols,
        "mfcc_lfcc":   mfcc_cols + lfcc_cols,
        "residual":    res_cols,
        "modulation":  mod_cols,
        "evidence":    res_cols + mod_cols,
    }


# ──────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────

def make_pipeline(model_type: str, cfg: dict, seed: int) -> Pipeline:
    """
    Buat sklearn Pipeline: Imputer → Scaler → Classifier.

    model_type: 'svm_rbf' | 'rf' | 'xgb' | 'knn' | 'mlp'
    """
    if model_type == "svm_rbf":
        clf = SVC(
            C=cfg["svm_C"], kernel=cfg["svm_kernel"],
            probability=True, class_weight="balanced",
            random_state=seed,
        )
    elif model_type == "rf":
        clf = RandomForestClassifier(
            n_estimators=cfg["rf_n_estimators"],
            max_depth=cfg["rf_max_depth"] or None,
            class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )
    elif model_type == "xgb":
        clf = XGBClassifier(
            n_estimators=cfg["xgb_n_estimators"],
            max_depth=cfg["xgb_max_depth"],
            learning_rate=cfg["xgb_learning_rate"],
            eval_metric="logloss",
            random_state=seed, n_jobs=-1,
            verbosity=0,
        )
    elif model_type == "knn":
        clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    elif model_type == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            max_iter=300,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     clf),
    ])


# ──────────────────────────────────────────────
# Train + evaluate single experiment
# ──────────────────────────────────────────────

def run_experiment(
    experiment_id: str,
    feature_group: str,
    model_type: str,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict,
    seed: int,
    out_scores: str = "results/utterance_scores.csv",
    out_checkpoints: str = "checkpoints",
    out_metrics: str = "results/metrics.csv",
) -> dict:
    """
    Latih satu model, evaluasi pada val dan test, simpan ke disk.

    Returns metrik validation sebagai dict.
    """
    print(f"\n[train] Experiment {experiment_id}: features={feature_group}, model={model_type}, seed={seed}")

    if not feature_cols:
        print(f"  [SKIP] Tidak ada fitur untuk grup '{feature_group}'")
        return {}

    X_train = df_train[feature_cols].values
    y_train = df_train["label"].values
    X_val   = df_val[feature_cols].values
    y_val   = df_val["label"].values
    X_test  = df_test[feature_cols].values
    y_test  = df_test["label"].values

    # Latih model
    pipeline = make_pipeline(model_type, cfg, seed=seed)
    pipeline.fit(X_train, y_train)

    # Skor pada validation dan test
    score_val  = pipeline.predict_proba(X_val)[:, 1]
    score_test = pipeline.predict_proba(X_test)[:, 1]

    # Threshold dari validation set SAJA
    _, val_threshold = compute_eer(y_val, score_val)

    # Metrik
    m_val  = compute_metrics(y_val,  score_val,  threshold=val_threshold,
                              split="validation", model_id=experiment_id, seed=seed)
    m_test = compute_metrics(y_test, score_test, threshold=val_threshold,
                              split="test",       model_id=experiment_id, seed=seed)

    print_metrics(m_val)
    print_metrics(m_test)

    # Simpan metrik
    save_metrics([m_val, m_test], out_metrics)

    # Simpan skor per utterance (untuk konsistensi analysis)
    score_col = f"score_{experiment_id}"
    rows_val  = df_val[["utterance_id", "speaker_id", "label"]].copy()
    rows_val[score_col] = score_val
    rows_val["split"]   = "validation"

    rows_test = df_test[["utterance_id", "speaker_id", "label"]].copy()
    rows_test[score_col] = score_test
    rows_test["split"]   = "test"

    scores_df = pd.concat([rows_val, rows_test], ignore_index=True)

    out_scores_path = Path(out_scores)
    out_scores_path.parent.mkdir(parents=True, exist_ok=True)
    if out_scores_path.exists():
        existing = pd.read_csv(str(out_scores_path))
        if score_col not in existing.columns:
            scores_df_merge = existing.merge(
                scores_df[["utterance_id", "split", score_col]],
                on=["utterance_id", "split"], how="left"
            )
            scores_df_merge.to_csv(str(out_scores_path), index=False)
    else:
        scores_df.to_csv(str(out_scores_path), index=False)

    # Simpan model ke checkpoints
    ckpt_dir = Path(out_checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model":         pipeline,
        "features":      feature_cols,
        "feature_group": feature_group,
        "model_type":    model_type,
        "seed":          seed,
        "val_threshold": val_threshold,
        "experiment_id": experiment_id,
    }
    ckpt_path = ckpt_dir / f"{experiment_id}_seed{seed}.joblib"
    joblib.dump(ckpt, str(ckpt_path))
    print(f"  Saved model → {ckpt_path}")

    return m_val


# ──────────────────────────────────────────────
# Main: jalankan semua eksperimen
# ──────────────────────────────────────────────

def train_all(
    results_dir: str = "results",
    config_path: str = "configs/baseline.yaml",
    lpc_order: int = 16,
    smoke_test: bool = False,
) -> None:
    cfg = load_config(config_path)

    suffix = f"_p{lpc_order}" if lpc_order != 16 else ""

    # Load fitur
    train_csv = Path(results_dir) / f"features{suffix}_train.csv"
    val_csv   = Path(results_dir) / f"features{suffix}_validation.csv"
    test_csv  = Path(results_dir) / f"features{suffix}_test.csv"

    for p in [train_csv, val_csv, test_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Feature CSV tidak ditemukan: {p}. Jalankan extract_features.py lebih dulu.")

    df_train = pd.read_csv(str(train_csv))
    df_val   = pd.read_csv(str(val_csv))
    df_test  = pd.read_csv(str(test_csv))

    if smoke_test:
        df_train = df_train.head(50).copy()
        df_val   = df_val.head(20).copy()
        df_test  = df_test.head(20).copy()
        print("[train] Smoke test mode")

    seeds = cfg["seeds"]
    feat_groups = get_feature_groups(df_train, lpc_order=lpc_order)

    # Daftar eksperimen wajib (panduan Tabel Baseline)
    experiments = [
        # (id, feature_group, model_type)
        ("B0", "mfcc",      "svm_rbf"),
        ("B1", "lfcc",      "svm_rbf"),
        ("B2", "mfcc_lfcc", "svm_rbf"),
        ("B3", "mfcc_lfcc", "rf"),
        ("B4", "mfcc_lfcc", "xgb"),
        # Evidence-only (ablation E4)
        ("E4a", "residual",   "svm_rbf"),
        ("E4b", "modulation", "svm_rbf"),
        ("E4c", "evidence",   "svm_rbf"),
        ("E4d", "evidence",   "rf"),
        ("E4e", "evidence",   "xgb"),
    ]

    all_metrics = []
    for exp_id, fg, mt in experiments:
        feat_cols = feat_groups.get(fg, [])
        for seed in seeds:
            set_seed(seed)
            m = run_experiment(
                experiment_id   = f"{exp_id}",
                feature_group   = fg,
                model_type      = mt,
                df_train        = df_train,
                df_val          = df_val,
                df_test         = df_test,
                feature_cols    = feat_cols,
                cfg             = cfg,
                seed            = seed,
                out_scores      = f"{results_dir}/utterance_scores{suffix}.csv",
                out_checkpoints = "checkpoints",
                out_metrics     = f"{results_dir}/metrics{suffix}.csv",
            )
            all_metrics.append(m)

    print("\n[train_baseline] All experiments completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline and evidence models")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--config",      default="configs/baseline.yaml")
    parser.add_argument("--lpc_order",   type=int, default=16, choices=[12, 16, 20])
    parser.add_argument("--smoke_test",  action="store_true")
    args = parser.parse_args()

    train_all(
        results_dir = args.results_dir,
        config_path = args.config,
        lpc_order   = args.lpc_order,
        smoke_test  = args.smoke_test,
    )
