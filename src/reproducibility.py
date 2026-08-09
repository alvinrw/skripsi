"""
src/reproducibility.py
======================
Set random seed untuk seluruh library agar eksperimen dapat direproduksi.
Dipanggil di awal SETIAP script sebelum import lain yang menggunakan random.
"""

import os
import random
import numpy as np


def set_seed(seed: int = 2026) -> None:
    """
    Set seed untuk os, random, dan numpy.
    Catatan: scikit-learn dan XGBoost menerima seed via parameter random_state,
    bukan global seed. Pastikan random_state=seed diteruskan ke setiap model.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    print(f"[reproducibility] Seed set to {seed}")


def load_config(config_path: str = "configs/baseline.yaml") -> dict:
    """
    Muat file YAML konfigurasi dan kembalikan sebagai dict.
    """
    import yaml  # pyyaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


if __name__ == "__main__":
    cfg = load_config()
    set_seed(cfg["seed"])
    print("Config loaded:", cfg)
