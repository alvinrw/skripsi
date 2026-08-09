"""
colab_pipeline.py
=================
Script pipeline lengkap untuk Google Colab.
Jalankan setiap cell secara berurutan di Colab.

Cara pakai:
1. Buka Google Colab: https://colab.research.google.com
2. File > New Notebook
3. Copy-paste setiap cell di bawah ke Colab
   ATAU upload file ini lalu jalankan:
   exec(open('colab_pipeline.py').read())
"""

# ============================================================
# CELL 1: Clone repo dari GitHub + install library
# ============================================================
"""
Paste di Colab Cell 1:

!git clone https://github.com/alvinrw/skripsi.git
%cd skripsi

!pip install -q numpy pandas scipy scikit-learn librosa soundfile \
    xgboost pyyaml joblib tqdm statsmodels kaggle

print("Setup selesai!")
"""

# ============================================================
# CELL 2: Setup Kaggle API (AMAN via Colab Secrets)
# ============================================================
"""
Paste di Colab Cell 2:

# === CARA AMAN: Gunakan Colab Secrets ===
# 1. Klik ikon KUNCI di sidebar kiri Colab (atau Tools > Secrets)
# 2. Tambahkan dua secret:
#    - Name: KAGGLE_USERNAME  Value: fawwasaliy
#    - Name: KAGGLE_KEY       Value: KGAT_3ca3a1537e8a5408a16d00551d9775bc
# 3. Aktifkan toggle "Notebook access" untuk keduanya
# 4. Jalankan cell ini

from google.colab import userdata
import os, json
from pathlib import Path

# Ambil dari Colab Secrets (aman, tidak muncul di notebook)
username = userdata.get('KAGGLE_USERNAME')
key      = userdata.get('KAGGLE_KEY')

# Simpan ke kaggle.json
kaggle_dir = Path('/root/.kaggle')
kaggle_dir.mkdir(exist_ok=True)
kaggle_json = kaggle_dir / 'kaggle.json'
kaggle_json.write_text(json.dumps({"username": username, "key": key}))
kaggle_json.chmod(0o600)

print(f"Kaggle credentials set for user: {username}")
print("Siap download dataset!")
"""

# ============================================================
# CELL 3: Download dataset DEEP-VOICE
# ============================================================
"""
Paste di Colab Cell 3:

import os
os.makedirs('data/raw', exist_ok=True)

# Download dan langsung extract (~1-2 GB, tunggu beberapa menit)
!kaggle datasets download -d birdy654/deep-voice-deepfake-voice-recognition \
    -p data/raw --unzip

# Verifikasi
import glob
wavs = glob.glob('data/raw/**/*.wav', recursive=True)
print(f"Total file audio: {len(wavs)}")
print("Contoh file:", wavs[:3])
"""

# ============================================================
# CELL 4: (OPSIONAL) Mount Google Drive untuk simpan hasil
# ============================================================
"""
Paste di Colab Cell 4 (opsional tapi SANGAT disarankan):

# Simpan hasil ke Google Drive agar tidak hilang saat sesi Colab berakhir
from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_DIR = '/content/drive/MyDrive/skripsi_results'
os.makedirs(DRIVE_DIR, exist_ok=True)

# Symlink folder results, checkpoints, manifests ke Drive
for folder in ['results', 'checkpoints', 'manifests']:
    os.makedirs(f'{DRIVE_DIR}/{folder}', exist_ok=True)
    if not os.path.exists(folder):
        os.symlink(f'{DRIVE_DIR}/{folder}', folder)
    else:
        print(f'{folder} sudah ada, skip symlink')

print("Google Drive terhubung! Hasil akan tersimpan di:", DRIVE_DIR)
"""

# ============================================================
# CELL 5: Smoke test - validasi semua modul berjalan
# ============================================================
"""
Paste di Colab Cell 5:

import sys
sys.path.insert(0, 'src')

!python src/run_pipeline.py --steps smoke
"""

# ============================================================
# CELL 6: Build manifest dari data yang sudah didownload
# ============================================================
"""
Paste di Colab Cell 6:

!python src/run_pipeline.py --steps manifest
"""

# ============================================================
# CELL 7: Split data berdasarkan speaker
# ============================================================
"""
Paste di Colab Cell 7:

!python src/run_pipeline.py --steps splits
"""

# ============================================================
# CELL 8: Validasi leakage (WAJIB sebelum training)
# ============================================================
"""
Paste di Colab Cell 8:

!python src/run_pipeline.py --steps leakage
"""

# ============================================================
# CELL 9: Ekstraksi fitur (proses paling lama ~30-60 menit)
# ============================================================
"""
Paste di Colab Cell 9:

# Gunakan GPU bila tersedia untuk mempercepat (librosa bisa pakai GPU)
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Tidak tersedia (CPU mode)")

# Ekstraksi semua fitur: MFCC, LFCC, Residual, Modulasi
!python src/extract_features.py --lpc_order 16

# Cek output
import pandas as pd
df = pd.read_csv('results/features_train.csv')
print(f"Train features: {df.shape}")
print(df.head(2))
"""

# ============================================================
# CELL 10: Training semua model (B0-B4, E4a-E4e)
# ============================================================
"""
Paste di Colab Cell 10:

!python src/train_baseline.py

# Lihat hasil
import pandas as pd
metrics = pd.read_csv('results/metrics.csv')
print(metrics[['model', 'split', 'seed', 'auc', 'eer', 'f1_fake']].to_string())
"""

# ============================================================
# CELL 11: Analisis statistik fitur bukti forensik
# ============================================================
"""
Paste di Colab Cell 11:

!python src/statistical_tests.py

import pandas as pd
stats = pd.read_csv('results/statistical_tests.csv')
print("Top 10 fitur berdasarkan effect size:")
print(stats[['feature', 'rank_biserial', 'p_fdr', 'effect_size_label']].head(10))
"""

# ============================================================
# CELL 12: Analisis konsistensi + bootstrap
# ============================================================
"""
Paste di Colab Cell 12:

!python src/run_pipeline.py --steps consistency bootstrap

# Lihat hasil konsistensi
import pandas as pd
consist = pd.read_csv('results/evidence_consistency.csv')
print(consist.T)
"""

# ============================================================
# CELL 13: Ablation LPC order (E2)
# ============================================================
"""
Paste di Colab Cell 13 (eksperimen E2):

for order in [12, 20]:
    print(f"\n=== Ablation LPC order p={order} ===")
    !python src/extract_features.py --lpc_order {order}
    !python src/train_baseline.py --lpc_order {order}
"""

# ============================================================
# CELL 14: Backup semua hasil ke Google Drive
# ============================================================
"""
Paste di Colab Cell 14:

import shutil, os
from datetime import datetime

# Buat folder backup dengan timestamp
ts = datetime.now().strftime('%Y%m%d_%H%M')
backup_dir = f'/content/drive/MyDrive/skripsi_backup_{ts}'
os.makedirs(backup_dir, exist_ok=True)

# Copy hasil
for folder in ['results', 'checkpoints', 'manifests', 'configs']:
    if os.path.exists(folder):
        shutil.copytree(folder, f'{backup_dir}/{folder}', dirs_exist_ok=True)

print(f"Backup selesai: {backup_dir}")
"""
