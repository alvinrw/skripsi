import json
from pathlib import Path

paths = [
    Path("../Pipeline_Skripsi_Colab_Barufase2.ipynb"),
    Path("../Pipeline_Skripsi_Colab_Baru.ipynb"),
    Path("Pipeline_Skripsi_Colab_Barufase2.ipynb"),
    Path("Pipeline_Skripsi_Colab_Baru.ipynb"),
]

cell18_source = [
    "import os, shutil\n",
    "print('Syncing figures and results to Google Drive...')\n",
    "!mkdir -p /content/drive/MyDrive/skripsi_results/figures\n",
    "!mkdir -p /content/drive/MyDrive/skripsi_results/results\n",
    "!cp -rf figures/* /content/drive/MyDrive/skripsi_results/figures/ 2>/dev/null || true\n",
    "!cp -rf results/* /content/drive/MyDrive/skripsi_results/results/ 2>/dev/null || true\n",
    "print('Seluruh file CSV, Confusion Matrix, dan Gambar Akademik G1-G7 telah tersimpan di Google Drive:')\n",
    "print('-> /content/drive/MyDrive/skripsi_results/figures/')\n"
]

for p in paths:
    if p.exists():
        with open(p, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        if len(nb['cells']) >= 18:
            nb['cells'][17]['source'] = cell18_source
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            print(f"Updated {p}")
