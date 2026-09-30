import json
from pathlib import Path

paths = [
    Path("../Pipeline_Skripsi_Colab_Barufase2.ipynb"),
    Path("../Pipeline_Skripsi_Colab_Baru.ipynb"),
    Path("Pipeline_Skripsi_Colab_Barufase2.ipynb"),
    Path("Pipeline_Skripsi_Colab_Baru.ipynb"),
]

for p in paths:
    if p.exists():
        with open(p, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        # In cell 17 (index 16, Section 5 code), add figure generation and sync at the end
        if len(nb['cells']) >= 17:
            cell17_src = nb['cells'][16]['source']
            sync_lines = [
                "\n\n# 5. Generate Gambar Akademik (G1-G7) & Sync ke Google Drive\n",
                "print(\"\\n🎨 Men-generate Gambar Akademik G1 s.d. G7...\")\n",
                "!python3 -B src/generate_figures.py\n",
                "print(\"☁️ Menyimpan seluruh file & gambar ke Google Drive /content/drive/MyDrive/skripsi_results/...\")\n",
                "!mkdir -p /content/drive/MyDrive/skripsi_results/figures\n",
                "!mkdir -p /content/drive/MyDrive/skripsi_results/results\n",
                "!cp -rf figures/* /content/drive/MyDrive/skripsi_results/figures/\n",
                "!cp -rf results/* /content/drive/MyDrive/skripsi_results/results/\n",
                "print(\"✅ Seluruh Gambar Akademik G1-G7 & File Results tersimpan di Google Drive:\")\n",
                "print(\"👉 /content/drive/MyDrive/skripsi_results/figures/\")\n"
            ]
            
            # Avoid duplicate if already added
            src_str = "".join(cell17_src)
            if "generate_figures.py" not in src_str:
                cell17_src.extend(sync_lines)
                nb['cells'][16]['source'] = cell17_src
                
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Updated {p}")
