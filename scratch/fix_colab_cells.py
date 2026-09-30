import json
from pathlib import Path

paths = [
    Path("../Pipeline_Skripsi_Colab_Barufase2.ipynb"),
    Path("../Pipeline_Skripsi_Colab_Baru.ipynb"),
    Path("Pipeline_Skripsi_Colab_Barufase2.ipynb"),
    Path("Pipeline_Skripsi_Colab_Baru.ipynb"),
]

section6_md = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 6. Sync Gambar Akademik (figures/) & Results ke Google Drive 📁\n",
        "Men-generate 7 Gambar Akademik (G1 s.d. G7) dan menyalin seluruh file hasil ke Google Drive (`/content/drive/MyDrive/skripsi_results/figures/`)."
    ]
}

section6_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "if os.path.exists(\"/content/skripsi_fase2\"):\n",
        "    %cd /content/skripsi_fase2\n\n",
        "print(\"🎨 Men-generate Gambar Akademik G1 s.d. G7...\")\n",
        "!python3 -B src/generate_figures.py\n\n",
        "print(\"☁️ Menyimpan ke Google Drive /content/drive/MyDrive/skripsi_results/...\")\n",
        "!mkdir -p /content/drive/MyDrive/skripsi_results/figures\n",
        "!mkdir -p /content/drive/MyDrive/skripsi_results/results\n",
        "!cp -rf figures/* /content/drive/MyDrive/skripsi_results/figures/\n",
        "!cp -rf results/* /content/drive/MyDrive/skripsi_results/results/\n\n",
        "print(\"✅ Seluruh Gambar Akademik G1-G7 & File Hasil tersimpan di Google Drive:\")\n",
        "print(\"👉 /content/drive/MyDrive/skripsi_results/figures/\")\n"
    ]
}

for p in paths:
    if p.exists():
        with open(p, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        # Keep up to cell 17 (index 16)
        new_cells = nb['cells'][:17]
        new_cells.append(section6_md)
        new_cells.append(section6_code)
        
        nb['cells'] = new_cells
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Successfully fixed cells in {p} (Total cells: {len(new_cells)})")
