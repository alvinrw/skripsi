import json
from pathlib import Path

fase2_dir = Path(r"c:\Users\alvin\Documents\Coolyeah\Skirpsi\Teknikal\fase 2")
repo_dir = fase2_dir / "skripsi_residual_modulasi"

# List of all pip packages needed for MeloTTS / OpenVoice / F5-TTS
pip_pkgs = "f5-tts ffmpeg-python soundfile pydub mecab-python3 unidic num2words pythainlp subword_nmt cached_path cn2an pypinyin pykakasi g2p_en g2pM fugashi ipadic unidic_lite anyascii jamo gruut gruut_ipa huggingface_hub"

nb_paths = [
    fase2_dir / "Batch_Voice_Cloning_Colab.ipynb",
    fase2_dir / "OpenVoice_Colab.ipynb",
    fase2_dir / "F5_TTS_Colab.ipynb",
    fase2_dir / "Generate_Voice_Cloning_Colab.ipynb",
    fase2_dir / "Generate_Voice_Cloning_Colab_(3).ipynb",
    repo_dir / "Batch_Voice_Cloning_Colab.ipynb",
    repo_dir / "OpenVoice_Colab.ipynb",
    repo_dir / "F5_TTS_Colab.ipynb",
    repo_dir / "Generate_Voice_Cloning_Colab.ipynb",
]

for p in nb_paths:
    if not p.exists():
        continue
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    modified = False
    for cell in data.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            new_source = []
            for line in source:
                if 'install' in line and ('mecab-python3' in line or 'pip install' in line):
                    if 'gruut' not in line:
                        if line.startswith('!pip install'):
                            indent = ''
                            line = f'!pip install --no-build-isolation {pip_pkgs}\n'
                        elif 'install(' in line:
                            indent = line[:len(line) - len(line.lstrip(' '))]
                            line = f'{indent}install("mecab-python3", "unidic", "num2words", "pythainlp", "subword_nmt", "cached_path", "cn2an", "pypinyin", "pykakasi", "g2p_en", "g2pM", "fugashi", "ipadic", "unidic_lite", "anyascii", "jamo", "gruut", "gruut_ipa")\n'
                        modified = True
                new_source.append(line)
            cell["source"] = new_source
            
    if modified:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Updated gruut & gruut_ipa in: {p.name}")
