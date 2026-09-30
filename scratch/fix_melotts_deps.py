import json
from pathlib import Path

nb_paths = [
    Path(r"c:\Users\alvin\Documents\Coolyeah\Skirpsi\Teknikal\fase 2\Generate_Voice_Cloning_Colab.ipynb"),
    Path(r"c:\Users\alvin\Documents\Coolyeah\Skirpsi\Teknikal\fase 2\skripsi_residual_modulasi\Generate_Voice_Cloning_Colab.ipynb")
]

new_install_line = '        install("mecab-python3", "unidic", "num2words", "pythainlp", "subword_nmt", "cached_path", "cn2an", "pypinyin", "pykakasi", "g2p_en", "g2pM", "fugashi", "ipadic", "unidic_lite", "anyascii", "jamo")\n'

for p in nb_paths:
    if not p.exists():
        continue
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    modified = False
    for cell in data.get("cells", []):
        if cell.get("cell_type") == "code":
            source_lines = cell.get("source", [])
            new_lines = []
            for line in source_lines:
                if 'install("mecab-python3"' in line:
                    line = new_install_line
                    modified = True
                new_lines.append(line)
            cell["source"] = new_lines
    
    if modified:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Updated: {p}")
