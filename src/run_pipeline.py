"""
src/run_pipeline.py
===================
Master script pipeline skripsi residual-modulasi speech deepfake.

Urutan tahapan:
    Step 0 (smoke)     : Validasi dependency dan sanity check
    Step 1 (manifest)  : Build source manifest dari data/raw/
    Step 2 (splits)    : Group split berdasarkan speaker
    Step 3 (leakage)   : Validasi tidak ada leakage
    Step 4 (features)  : Ekstraksi semua fitur (MFCC/LFCC/Residual/Mod)
    Step 5 (train)     : Latih semua model baseline dan evidence
    Step 6 (stats)     : Uji statistik bukti forensik
    Step 7 (consist)   : Analisis konsistensi baseline vs evidence
    Step 8 (bootstrap) : Bootstrap AUC comparison antar model

Cara pakai:
    # Jalankan smoke test tanpa dataset (validasi pipeline)
    python src/run_pipeline.py --smoke_test

    # Jalankan pipeline penuh (setelah dataset tersedia)
    python src/run_pipeline.py --steps all

    # Jalankan tahap tertentu
    python src/run_pipeline.py --steps manifest splits leakage features

    # Ablation LPC order
    python src/run_pipeline.py --steps features train --lpc_order 12
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
import traceback

# Tambah src ke path
sys.path.insert(0, str(Path(__file__).parent))

from reproducibility import load_config, set_seed


# ──────────────────────────────────────────────
# Step runners
# ──────────────────────────────────────────────

def step_smoke_test(cfg: dict) -> bool:
    """Validasi dependency dan sanity check modul."""
    print("\n[Step 0] Smoke test: validasi import dan dependency...")

    errors = []
    modules = [
        ("numpy",            "import numpy as np; print('numpy', np.__version__)"),
        ("pandas",           "import pandas as pd; print('pandas', pd.__version__)"),
        ("scipy",            "import scipy; print('scipy', scipy.__version__)"),
        ("sklearn",          "import sklearn; print('sklearn', sklearn.__version__)"),
        ("librosa",          "import librosa; print('librosa', librosa.__version__)"),
        ("soundfile",        "import soundfile; print('soundfile OK')"),
        ("xgboost",          "import xgboost as xgb; print('xgboost', xgb.__version__)"),
        ("yaml",             "import yaml; print('pyyaml OK')"),
        ("statsmodels",      "from statsmodels.stats.multitest import multipletests; print('statsmodels OK')"),
        ("tqdm",             "from tqdm import tqdm; print('tqdm OK')"),
    ]

    for name, code in modules:
        try:
            exec(code)
        except ImportError as e:
            errors.append(f"  MISSING: {name} — {e}")

    # Sanity check modul skripsi
    skripsi_modules = [
        "audio_io",
        "spectral_features",
        "lfcc",
        "residual_features",
        "modulation_features",
        "metrics",
        "reproducibility",
    ]
    for mod in skripsi_modules:
        try:
            __import__(mod)
            print(f"  [OK] {mod}")
        except Exception as e:
            errors.append(f"  FAIL: {mod} — {e}")

    # Quick numeric sanity check
    import numpy as np
    dummy = np.random.randn(32000).astype(np.float32)

    from spectral_features import mfcc_features
    from lfcc import lfcc_features
    from residual_features import residual_feature_vector
    from modulation_features import modulation_features

    try:
        m = mfcc_features(dummy)
        assert m.shape[0] == 240, f"MFCC dim wrong: {m.shape[0]}"
        print(f"  [OK] MFCC: {m.shape[0]} dims")
    except Exception as e:
        errors.append(f"  FAIL: MFCC — {e}")

    try:
        l = lfcc_features(dummy)
        assert l.shape[0] == 40, f"LFCC dim wrong: {l.shape[0]}"
        print(f"  [OK] LFCC: {l.shape[0]} dims")
    except Exception as e:
        errors.append(f"  FAIL: LFCC — {e}")

    try:
        r = residual_feature_vector(dummy, order=16)
        assert r.shape[0] == 15, f"Residual dim wrong: {r.shape[0]}"
        print(f"  [OK] Residual (p=16): {r.shape[0]} dims")
    except Exception as e:
        errors.append(f"  FAIL: Residual — {e}")

    try:
        mo = modulation_features(dummy)
        assert mo.shape[0] == 7, f"Modulation dim wrong: {mo.shape[0]}"
        print(f"  [OK] Modulation: {mo.shape[0]} dims")
    except Exception as e:
        errors.append(f"  FAIL: Modulation — {e}")

    if errors:
        print("\n[Step 0] FAILED:")
        for e in errors:
            print(e)
        return False

    print("\n[Step 0] [OK] All smoke test checks PASSED")
    return True


def step_prepare(cfg: dict, args: argparse.Namespace) -> bool:
    import sys
    from subprocess import run
    print("\n[Step 0b] Menjalankan persiapan dataset VoxCPM dari Google Drive...")
    cmd = [sys.executable, "src/prepare_dataset.py", "--drive_dir", getattr(args, "drive_dir", "data/raw"), "--out_dir", getattr(args, "out_dir", "data/processed")]
    if getattr(args, "zip_out", None):
        cmd += ["--zip_out", args.zip_out]
    res = run(cmd)
    return res.returncode == 0


def step_manifest(cfg: dict, args: argparse.Namespace) -> bool:
    from build_manifest import build_manifest
    try:
        build_manifest(
            data_dir     = "data/raw",
            out_csv      = "manifests/source_manifest.csv",
            label_map_csv= getattr(args, "label_map", None),
            dataset_name = "DEEP_VOICE",
        )
        return True
    except Exception as e:
        print(f"[Step 1] ERROR: {e}")
        traceback.print_exc()
        return False


def step_splits(cfg: dict, args: argparse.Namespace) -> bool:
    from make_splits import make_splits
    try:
        make_splits(
            source_csv  = "manifests/source_manifest.csv",
            out_csv     = "manifests/split_manifest.csv",
            config_path = args.config,
        )
        return True
    except Exception as e:
        print(f"[Step 2] ERROR: {e}")
        traceback.print_exc()
        return False


def step_leakage(cfg: dict, args: argparse.Namespace) -> bool:
    from check_leakage import check_leakage
    try:
        ok = check_leakage("manifests/split_manifest.csv")
        if not ok:
            print("[Step 3] LEAKAGE DETECTED — pipeline dihentikan.")
        return ok
    except Exception as e:
        print(f"[Step 3] ERROR: {e}")
        return False


def step_features(cfg: dict, args: argparse.Namespace) -> bool:
    from extract_features import extract_all_features
    try:
        for order in ([args.lpc_order] if args.lpc_order else [16]):
            extract_all_features(
                manifest_csv = "manifests/split_manifest.csv",
                results_dir  = "results",
                config_path  = args.config,
                lpc_order    = order,
                smoke_test   = args.smoke_test,
                duration     = getattr(args, "duration", None),
            )
        return True
    except Exception as e:
        print(f"[Step 4] ERROR: {e}")
        traceback.print_exc()
        return False


def step_train(cfg: dict, args: argparse.Namespace) -> bool:
    from train_baseline import train_all
    try:
        train_all(
            results_dir = "results",
            config_path = args.config,
            lpc_order   = args.lpc_order or 16,
            smoke_test  = args.smoke_test,
            duration    = getattr(args, "duration", None),
        )
        return True
    except Exception as e:
        print(f"[Step 5] ERROR: {e}")
        traceback.print_exc()
        return False


def step_stats(cfg: dict, args: argparse.Namespace) -> bool:
    from statistical_tests import run_statistical_tests
    try:
        run_statistical_tests(
            features_csv = "results/features_train.csv",
            out_csv      = "results/statistical_tests.csv",
            n_boot       = cfg.get("n_bootstrap", 2000),
            seed         = cfg["seed"],
        )
        return True
    except Exception as e:
        print(f"[Step 6] ERROR: {e}")
        traceback.print_exc()
        return False


def step_consistency(cfg: dict, args: argparse.Namespace) -> bool:
    from evidence_consistency import analyze_consistency
    try:
        analyze_consistency(
            scores_csv         = "results/utterance_scores.csv",
            baseline_col       = "score_B2",
            evidence_col       = "score_E4c",
            out_consistency    = "results/evidence_consistency.csv",
            out_error          = "results/error_analysis.csv",
            split              = "test",
        )
        return True
    except Exception as e:
        print(f"[Step 7] ERROR: {e}")
        traceback.print_exc()
        return False


def step_bootstrap(cfg: dict, args: argparse.Namespace) -> bool:
    from bootstrap_difference import compare_models
    try:
        compare_models(
            scores_csv = "results/utterance_scores.csv",
            split      = "test",
            n_boot     = cfg.get("n_bootstrap", 2000),
            seed       = cfg["seed"],
            out_csv    = "results/bootstrap_comparisons.csv",
        )
        return True
    except Exception as e:
        print(f"[Step 8] ERROR: {e}")
        traceback.print_exc()
        return False


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

STEP_MAP = {
    "smoke":      step_smoke_test,
    "prepare":    step_prepare,
    "manifest":   step_manifest,
    "splits":     step_splits,
    "leakage":    step_leakage,
    "features":   step_features,
    "train":      step_train,
    "stats":      step_stats,
    "consistency": step_consistency,
    "bootstrap":  step_bootstrap,
}

ALL_STEPS = ["smoke", "prepare", "manifest", "splits", "leakage",
             "features", "train", "stats", "consistency", "bootstrap"]


def main():
    parser = argparse.ArgumentParser(
        description="Master pipeline skripsi residual-modulasi speech deepfake",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--steps", nargs="+", default=["smoke"],
        choices=ALL_STEPS + ["all"],
        help="Tahapan yang akan dijalankan. Gunakan 'all' untuk semua tahapan.",
    )
    parser.add_argument("--config",     default="configs/baseline.yaml")
    parser.add_argument("--lpc_order",  type=int, default=16, choices=[12, 16, 20])
    parser.add_argument("--smoke_test", action="store_true",
                        help="Gunakan subset kecil data untuk validasi pipeline cepat")
    parser.add_argument("--label_map",  default=None,
                        help="Path ke CSV label map (opsional)")
    parser.add_argument("--duration", type=str, default=None,
                        help="Durasi audio untuk diekstrak fiturnya (misal: '2s', '3s')")
    parser.add_argument("--drive_dir", type=str, default="data/raw",
                        help="Path ke data mentah (digunakan oleh step prepare)")
    parser.add_argument("--out_dir", type=str, default="data/processed",
                        help="Path ke output chunking (digunakan oleh step prepare)")
    parser.add_argument("--zip_out", type=str, default=None,
                        help="Path zip output opsional (digunakan oleh step prepare)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    steps = ALL_STEPS if "all" in args.steps else args.steps

    print(f"\n{'='*60}")
    print("Pipeline Skripsi: Residual & Modulasi Speech Deepfake")
    print(f"{'='*60}")
    print(f"Steps   : {steps}")
    print(f"Config  : {args.config}")
    print(f"Seed    : {cfg['seed']}")
    print(f"LPC ord : {args.lpc_order}")
    print(f"Smoke   : {args.smoke_test}")
    print(f"{'='*60}\n")

    for step_name in steps:
        print(f"\n{'='*50}")
        print(f">> Step: {step_name.upper()}")
        print(f"{'='*50}")

        step_fn = STEP_MAP[step_name]

        if step_name == "smoke":
            ok = step_fn(cfg)
        else:
            ok = step_fn(cfg, args)

        if not ok:
            print(f"\n[FAIL] Step '{step_name}' FAILED. Pipeline dihentikan.")
            sys.exit(1)

        print(f"[OK] Step '{step_name}' selesai.")

    print(f"\n{'='*60}")
    print("[OK] Pipeline selesai!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
