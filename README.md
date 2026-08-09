# 🎙️ Skripsi: Analisis Forensik Bukti melalui Residual dan Modulasi pada Deteksi Speech Deepfake

> **Pelaksana:** Alvin Rifky Wahyudi (235150300111005)  
> **Universitas Brawijaya | 2026**  
> **Bidang:** Speech Signal Processing & Audio Forensics

Pipeline penelitian skripsi ini membangun **dua jalur independen** untuk mendeteksi speech deepfake:
- **Jalur Klasifikasi** — menggunakan MFCC/LFCC + Machine Learning (SVM, Random Forest, XGBoost)
- **Jalur Forensik** — mengekstraksi bukti residual prediksi linear dan dinamika modulasi untuk analisis karakteristik fisik sinyal

---

## 📋 Daftar Isi

1. [Arsitektur Pipeline](#arsitektur-pipeline)
2. [Struktur Folder](#struktur-folder)
3. [Penjelasan Per File](#penjelasan-per-file)
4. [Quick Start](#quick-start)
5. [Cara Menjalankan Pipeline](#cara-menjalankan-pipeline)
6. [Fitur yang Diekstraksi](#fitur-yang-diekstraksi)
7. [Model yang Digunakan](#model-yang-digunakan)
8. [Dataset](#dataset)
9. [Konfigurasi](#konfigurasi)
10. [Output & Hasil](#output--hasil)
11. [Konvensi Penting](#konvensi-penting)
12. [Rencana Kerja 12 Minggu](#rencana-kerja-12-minggu)
13. [Referensi](#referensi)

---

## Arsitektur Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        Audio Input                              │
│                   (DEEP-VOICE Dataset)                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
   ┌──────────────────┐      ┌────────────────────────┐
   │  JALUR KLASIFIKASI│      │    JALUR FORENSIK       │
   │                  │      │                        │
   │  MFCC (240 dim)  │      │  Residual LPC (15 dim) │
   │  LFCC  (40 dim)  │      │  Modulasi    ( 7 dim)  │
   │                  │      │                        │
   │  SVM / RF / XGB  │      │  Mann-Whitney U + FDR  │
   │  (B0 – B4)       │      │  Evidence-only Model   │
   │                  │      │  (E4a – E4e)           │
   └────────┬─────────┘      └──────────┬─────────────┘
            │                           │
            │    Skor per utterance     │
            └─────────────┬─────────────┘
                          │
                          ▼
             ┌────────────────────────┐
             │  CONSISTENCY ANALYSIS  │
             │  Spearman ρ, Cohen κ   │
             │  4 kategori kasus      │
             └────────────────────────┘
```

---

## Struktur Folder

```
skripsi_residual_modulasi/
│
├── 📄 README.md                    ← Dokumentasi lengkap ini
├── 📄 requirements-lock.txt        ← Versi library yang dipakai (pip freeze)
│
├── 📁 configs/                     ← Konfigurasi eksperimen
│   └── baseline.yaml               ← SEMUA parameter terpusat di sini
│
├── 📁 src/                         ← Seluruh source code Python
│   ├── reproducibility.py
│   ├── audio_io.py
│   ├── build_manifest.py
│   ├── make_splits.py
│   ├── check_leakage.py
│   ├── spectral_features.py
│   ├── lfcc.py
│   ├── residual_features.py
│   ├── modulation_features.py
│   ├── extract_features.py
│   ├── train_baseline.py
│   ├── metrics.py
│   ├── aggregate_scores.py
│   ├── statistical_tests.py
│   ├── evidence_consistency.py
│   ├── bootstrap_difference.py
│   ├── run_pipeline.py             ← MASTER SCRIPT (jalankan ini)
│   ├── download_data.py            ← Download DEEP-VOICE dari Kaggle
│   └── download_asvspoof.py        ← Panduan download ASVspoof 2021
│
├── 📁 data/                        ← DATA (tidak di-commit ke Git)
│   ├── raw/                        ← Audio mentah, JANGAN diubah
│   │   ├── real/                   ← Audio real/bonafide
│   │   └── fake/                   ← Audio deepfake/spoofed
│   └── processed/                  ← Audio hasil QC dan resampling
│
├── 📁 manifests/                   ← Metadata dan split data
│   ├── source_manifest.csv         ← Daftar semua file audio + SHA256
│   └── split_manifest.csv          ← File yang sama + kolom split (train/val/test)
│
├── 📁 checkpoints/                 ← Model yang sudah dilatih (tidak di-commit)
│   └── *.joblib                    ← Model, scaler, threshold per eksperimen
│
├── 📁 results/                     ← Semua output numerik eksperimen
│   ├── features_train.csv
│   ├── features_validation.csv
│   ├── features_test.csv
│   ├── utterance_scores.csv
│   ├── metrics.csv
│   ├── statistical_tests.csv
│   ├── evidence_consistency.csv
│   ├── error_analysis.csv
│   └── bootstrap_comparisons.csv
│
├── 📁 figures/                     ← Gambar untuk Bab 4 skripsi
│   └── (G1–G7 akan dihasilkan dari notebooks)
│
└── 📁 notebooks/                   ← Eksplorasi & visualisasi saja
    └── (bukan bagian pipeline utama)
```

> **Catatan Git:** Folder `data/`, `checkpoints/`, dan `results/` **tidak di-commit** karena ukurannya besar. Sudah didaftarkan di `.gitignore`.

---

## Penjelasan Per File

### 📁 `configs/`

#### `configs/baseline.yaml`
File konfigurasi utama. **Semua parameter eksperimen ada di sini** — tidak ada angka yang di-hardcode di source code.

```yaml
seed: 2026                  # Random seed untuk reproduksi
sample_rate: 16000          # Hz — semua audio diresample ke ini
segment_seconds: 2.0        # Panjang segmen audio
hop_seconds: 1.0            # Jarak antar segmen (overlap 50%)

mfcc_n: 20                  # Jumlah koefisien MFCC
lfcc_n: 20                  # Jumlah koefisien LFCC
lfcc_n_filters: 40          # Jumlah filter dalam filterbank LFCC

lpc_order: 16               # Orde LPC default (ablation: 12, 16, 20)
lpc_frame_ms: 25            # Panjang frame LPC (milidetik)
lpc_hop_ms: 10              # Hop LPC (milidetik)

svm_C: 10                   # Regularisasi SVM
rf_n_estimators: 200        # Jumlah pohon Random Forest
xgb_n_estimators: 200       # Boosting rounds XGBoost

seeds: [2026, 2027, 2028]   # 3 seed untuk stabilitas hasil
```

---

### 📁 `src/` — Source Code

#### `src/run_pipeline.py` ⭐ MASTER SCRIPT
**File utama yang dijalankan.** Mengorkestrasi semua 8 tahapan pipeline secara berurutan.

```
Tahapan:
  smoke      → validasi import dan dependency
  manifest   → scan data/raw/, hitung SHA256, buat CSV
  splits     → bagi data berdasarkan speaker (tanpa leakage)
  leakage    → validasi tidak ada speaker yang melintasi split
  features   → ekstraksi MFCC/LFCC/Residual/Modulasi
  train      → latih semua model dan simpan checkpoint
  stats      → uji statistik Mann-Whitney U + FDR
  consistency → analisis konsistensi dua jalur
  bootstrap  → paired AUC comparison antar model
```

---

#### `src/reproducibility.py`
Set global random seed agar semua eksperimen dapat direproduksi.
- `set_seed(seed)` → set seed untuk `os`, `random`, `numpy`
- `load_config(path)` → baca file YAML sebagai dict

---

#### `src/download_data.py`
Script download dataset **DEEP-VOICE** dari Kaggle.

```bash
# Opsi 1: Download otomatis via Kaggle API
python src/download_data.py

# Opsi 2: Verifikasi bila sudah download manual
python src/download_data.py --skip_kaggle

# Opsi 3: Tampilkan panduan download manual
python src/download_data.py --manual_guide
```

Prasyarat: File `~/.kaggle/kaggle.json` harus ada (dari [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create Token).

---

#### `src/download_asvspoof.py`
Helper untuk dataset **ASVspoof 2021 DF** (dataset eksternal).

ASVspoof **tidak bisa di-download otomatis** — harus request akses di [asvspoof.org](https://www.asvspoof.org/index2021.html). File ini berisi panduan registrasi, parser `trial_metadata.txt`, dan script untuk menambahkan ASVspoof ke manifest pipeline.

```bash
python src/download_asvspoof.py                  # tampilkan panduan
python src/download_asvspoof.py --verify         # cek apakah sudah ada
python src/download_asvspoof.py --parse_metadata # buat manifest dari metadata
```

> ⚠️ Dataset ini **hanya digunakan di minggu 10** sebagai cross-dataset test. Tidak boleh dipakai untuk training atau tuning apapun.

---

#### `src/build_manifest.py`
Scan seluruh file audio di `data/raw/`, hitung SHA256 setiap file, dan buat `manifests/source_manifest.csv`.

**Kolom output:**

| Kolom | Isi |
|-------|-----|
| `utterance_id` | Nama file (tanpa ekstensi) |
| `file_path` | Path relatif dari root proyek |
| `speaker_id` | ID speaker (inferensi dari nama folder) |
| `label` | 0 = real, 1 = deepfake |
| `generator_id` | Nama generator/model TTS (atau "unknown") |
| `dataset` | DEEP_VOICE atau ASVspoof2021_DF |
| `split` | UNASSIGNED (diisi oleh `make_splits.py`) |
| `duration_s` | Durasi audio dalam detik |
| `sample_rate` | Sample rate asli file |
| `sha256` | Hash SHA256 untuk deteksi duplikat |

```bash
python src/build_manifest.py --data_dir data/raw --out manifests/source_manifest.csv
```

---

#### `src/make_splits.py`
Membagi data menjadi train / validation / test **berdasarkan `speaker_id`** menggunakan `GroupShuffleSplit`. Ini mencegah speaker yang sama muncul di lebih dari satu split (speaker leakage).

```
Split:  Train 64% | Validation 16% | Test 20%
Metode: GroupShuffleSplit(groups=speaker_id)
```

```bash
python src/make_splits.py --config configs/baseline.yaml
```

---

#### `src/check_leakage.py`
Validasi bahwa tidak ada `utterance_id`, `sha256`, atau `speaker_id` yang muncul di lebih dari satu split. **Wajib dijalankan sebelum setiap training.**

```bash
python src/check_leakage.py
# Exit code 0 = aman, Exit code 1 = ada leakage (pipeline berhenti)
```

---

#### `src/audio_io.py`
Fungsi dasar untuk load dan memproses file audio.

- `load_audio(path)` → load mono, resample ke 16kHz, normalisasi peak per utterance
- `segment_audio(x)` → bagi sinyal menjadi segmen 2 detik dengan hop 1 detik
- `load_and_segment(path)` → gabungan keduanya (shortcut)

---

#### `src/spectral_features.py`
Ekstraksi **MFCC** (Mel-Frequency Cepstral Coefficients).

```
Output: 240 dimensi
Rumus : 20 koef × 3 order (MFCC + Δ + ΔΔ) × 4 statistik (mean, std, Q25, Q75)
```

- `mfcc_features(x)` → vektor fitur 240-dim
- `mfcc_feature_names()` → nama kolom untuk DataFrame
- `summarize_matrix(mat)` → agregasi statistik dari matriks koefisien

---

#### `src/lfcc.py`
Ekstraksi **LFCC** (Linear Frequency Cepstral Coefficients).

Berbeda dari MFCC yang menggunakan filterbank mel-scale (logaritmik), LFCC menggunakan filterbank **linear** (equally-spaced dalam Hz). Ini membuat LFCC lebih sensitif terhadap artefak TTS di frekuensi tinggi.

```
Output : 40 dimensi
Rumus  : 20 koef × 2 statistik (mean, std)
Internal: 40 filter triangular dari 0–8000 Hz → DCT → ambil 20 koef pertama
```

- `linear_filterbank(power, freqs)` → terapkan 40 filter linear
- `lfcc_features(x)` → vektor fitur 40-dim

---

#### `src/residual_features.py`
Ekstraksi **bukti residual** berbasis prediksi linear (LPC).

**Prinsip:** Sinyal ucapan dapat diprediksi menggunakan model all-pole (AR). Sisa yang tidak dapat diprediksi (residual) mengandung informasi tentang sumber eksitasi. Pada speech deepfake, pola residual ini berbeda dari ucapan asli.

```
Persamaan: e[n] = x[n] + Σ(k=1..p) a_k · x[n-k]
```

**Fitur per frame (5 fitur):**

| Fitur | Makna |
|-------|-------|
| `residual_energy` | Energi rata-rata sinyal residual |
| `residual_entropy` | Shannon entropy distribusi residual |
| `prediction_error` | Rasio energi residual / energi sinyal |
| `excitation_irregularity` | Ketidakberaturan temporal eksitasi |
| `kurtosis` | Peakedness distribusi residual |

**Agregasi ke utterance:** mean, std, median → **15 dimensi total**

**Ablation:** Orde LPC p ∈ {12, 16, 20} untuk menilai sensitivitas fitur.

> ⚠️ Residual LPC bukan "glottal source murni" — harus disebut sebagai "excitation-related evidence" atau "prediction residual".

---

#### `src/modulation_features.py`
Ekstraksi **bukti modulasi** dari envelope amplitudo.

Modulasi menangkap perubahan amplitudo jangka menengah (ritme bicara). Pada deepfake, pola modulasi ini sering tidak alami karena proses sintesis.

**Alur:**
```
Sinyal → Hilbert Transform → Envelope → Lowpass 40Hz
       → Resample ke 100Hz → De-mean → FFT (Hanning)
       → Modulation Spectrum [0–20 Hz]
```

**Fitur (7 dimensi):**

| Fitur | Rentang Frekuensi | Makna |
|-------|-------------------|-------|
| `mod_band_0.5_2` | 0.5–2 Hz | Ritme suku kata |
| `mod_band_2_4` | 2–4 Hz | Ritme bergetar |
| `mod_band_4_8` | 4–8 Hz | Modulasi menengah |
| `mod_band_8_20` | 8–20 Hz | Modulasi cepat |
| `mod_centroid` | — | Pusat massa spektrum modulasi |
| `mod_entropy` | — | Keteraturan spektrum modulasi |
| `mod_depth` | — | std(env) / mean(\|env\|) |

---

#### `src/extract_features.py`
Mengorkestrasi ekstraksi semua fitur untuk seluruh utterance di manifest.

- Membaca `split_manifest.csv`
- Untuk setiap utterance: load audio → segmentasi → ekstrak MFCC + LFCC + Residual + Modulasi
- Agregasi skor segmen menjadi representasi utterance (mean)
- Output: `results/features_{split}.csv` per split

```bash
python src/extract_features.py --lpc_order 16
python src/extract_features.py --lpc_order 12  # ablation
python src/extract_features.py --smoke_test     # uji cepat 100 utterance
```

---

#### `src/train_baseline.py`
Melatih semua model dan menyimpan checkpoint.

**Eksperimen baseline:**

| ID | Fitur | Model | Tujuan |
|----|-------|-------|--------|
| B0 | MFCC | SVM RBF | Baseline utama |
| B1 | LFCC | SVM RBF | Frekuensi linear |
| B2 | MFCC+LFCC | SVM RBF | Fusi spektral |
| B3 | MFCC+LFCC | Random Forest | Non-linear pohon |
| B4 | MFCC+LFCC | XGBoost | Gradient boosting |

**Eksperimen bukti forensik:**

| ID | Fitur | Model | Tujuan |
|----|-------|-------|--------|
| E4a | Residual | SVM | Ablation domain residual |
| E4b | Modulasi | SVM | Ablation domain modulasi |
| E4c | Residual+Mod | SVM | Evidence-only gabungan |
| E4d | Residual+Mod | RF | Non-linear evidence |
| E4e | Residual+Mod | XGBoost | Boosting evidence |

Setiap eksperimen dijalankan dengan **3 seed** (2026, 2027, 2028). Threshold EER ditentukan **hanya dari validation set**.

---

#### `src/metrics.py`
Fungsi evaluasi performa model.

- `compute_eer(y, score)` → EER dan threshold-nya
- `compute_metrics(y, score, threshold)` → dict lengkap semua metrik
- `save_metrics(list, path)` → append ke `results/metrics.csv`

**Metrik yang dihitung:**

| Metrik | Keterangan |
|--------|-----------|
| AUC | Area Under ROC Curve — kemampuan ranking |
| EER | Equal Error Rate — titik FPR = FNR |
| Accuracy | Deskriptif saja, bukan metrik utama |
| Precision/Recall/F1 | Per kelas (real, fake) dan macro |

---

#### `src/aggregate_scores.py`
Agregasi skor dari tingkat segmen ke tingkat utterance.

```
Kolom output: score_mean, score_median, score_std, score_min, score_max, n_segments
```

> Metrik utama **selalu dilaporkan di tingkat utterance**, bukan segmen. Segmen dari utterance yang sama bukan sampel independen.

---

#### `src/statistical_tests.py`
Uji statistik untuk menilai apakah fitur residual dan modulasi berbeda secara signifikan antara kelas real dan deepfake.

**Metode:**
1. **Mann-Whitney U** (two-sided) — uji utama, tidak mengasumsikan normalitas
2. **Rank-biserial correlation** — effect size (|rb| ≥ 0.3 kecil, ≥ 0.5 sedang, ≥ 0.7 besar)
3. **Benjamini-Hochberg FDR** — koreksi multiple comparisons
4. **Bootstrap 95% CI** — ketidakpastian selisih median

> Laporkan **effect size**, bukan hanya p-value. Hasil non-signifikan tetap dilaporkan.

---

#### `src/evidence_consistency.py`
Analisis konsistensi antara jalur baseline MFCC/LFCC dan jalur bukti residual-modulasi.

**4 kategori kasus:**

| Kategori | Makna |
|----------|-------|
| `consistent_correct` | Baseline ✓ & Bukti ✓ — mudah dijelaskan |
| `baseline_only_correct` | Baseline ✓ & Bukti ✗ — baseline lebih andal |
| `evidence_only_correct` | Baseline ✗ & Bukti ✓ — bukti menangkap apa yang baseline lewatkan |
| `both_wrong` | Baseline ✗ & Bukti ✗ — kasus sulit |

**Metrik:** Spearman ρ (korelasi skor), Cohen's κ (agreement keputusan)

---

#### `src/bootstrap_difference.py`
Bootstrap paired AUC difference test antara dua model.

- Gunakan untuk membandingkan apakah Model A **secara statistik** lebih baik dari Model B
- Klaim "lebih baik" hanya valid bila **95% CI tidak melintasi nol**
- Pisahkan analisis confirmatory dari eksplorasi

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/alvinrw/skripsi.git
cd skripsi
```

### 2. Setup Environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install numpy pandas scipy scikit-learn librosa soundfile `
    xgboost pyyaml joblib tqdm statsmodels kaggle

pip freeze > requirements-lock.txt
```

### 3. Validasi Pipeline (tanpa dataset)

```powershell
python src/run_pipeline.py --steps smoke
```

Output yang diharapkan:
```
[OK] MFCC: 240 dims
[OK] LFCC:  40 dims
[OK] Residual (p=16): 15 dims
[OK] Modulation: 7 dims
All smoke test checks PASSED
```

### 4. Download Dataset

```powershell
# Setup Kaggle API key dulu di ~/.kaggle/kaggle.json
python src/download_data.py
```

Atau download manual dari:
- **DEEP-VOICE:** https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition
- **ASVspoof 2021:** https://www.asvspoof.org/index2021.html

---

## Cara Menjalankan Pipeline

### Jalankan Bertahap (disarankan)

```powershell
# Step 1: Buat manifest dari folder audio
python src/run_pipeline.py --steps manifest

# Step 2: Split data berdasarkan speaker (tanpa leakage)
python src/run_pipeline.py --steps splits

# Step 3: WAJIB — validasi tidak ada speaker leakage
python src/run_pipeline.py --steps leakage

# Step 4: Ekstraksi semua fitur
python src/run_pipeline.py --steps features

# Step 5: Training semua model (B0–B4, E4a–E4e) dengan 3 seed
python src/run_pipeline.py --steps train

# Step 6: Uji statistik fitur bukti forensik
python src/run_pipeline.py --steps stats

# Step 7: Analisis konsistensi dua jalur
python src/run_pipeline.py --steps consistency

# Step 8: Bootstrap AUC comparison antar model
python src/run_pipeline.py --steps bootstrap
```

### Jalankan Semua Sekaligus

```powershell
python src/run_pipeline.py --steps all
```

### Ablation LPC Order (Eksperimen E2)

```powershell
python src/run_pipeline.py --steps features train --lpc_order 12
python src/run_pipeline.py --steps features train --lpc_order 16  # default
python src/run_pipeline.py --steps features train --lpc_order 20
```

---

## Fitur yang Diekstraksi

| Domain | Dimensi | Keterangan |
|--------|---------|------------|
| MFCC | 240 | 20 koef × 3 order × 4 statistik |
| LFCC | 40 | 20 koef × 2 statistik |
| Residual LPC | 15 | 5 fitur × 3 agregasi (p=16 default) |
| Modulasi | 7 | 4 band + centroid + entropy + depth |
| **Total** | **302** | Per utterance |

---

## Model yang Digunakan

Semua model menggunakan pipeline: **Imputer (median) → StandardScaler → Classifier**

| ID | Model | Fitur | Tujuan |
|----|-------|-------|--------|
| B0 | SVM RBF (C=10) | MFCC | Baseline utama stabil |
| B1 | SVM RBF | LFCC | Representasi linear |
| B2 | SVM RBF | MFCC+LFCC | Fusi spektral sederhana |
| B3 | Random Forest (200 trees) | MFCC+LFCC | Non-linear pohon |
| B4 | XGBoost (200 rounds) | MFCC+LFCC | Gradient boosting |
| E4a | SVM RBF | Residual only | Ablation domain |
| E4b | SVM RBF | Modulasi only | Ablation domain |
| E4c | SVM RBF | Residual+Mod | Evidence-only model |
| E4d | Random Forest | Residual+Mod | Non-linear evidence |
| E4e | XGBoost | Residual+Mod | Boosting evidence |

---

## Dataset

| Dataset | Peran | Sumber | Ukuran |
|---------|-------|--------|--------|
| **DEEP-VOICE** | Dataset utama (train/val/test) | [Kaggle](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition) | ~1-2 GB |
| **ASVspoof 2021 DF** | Cross-dataset test saja | [asvspoof.org](https://www.asvspoof.org/index2021.html) | ~11 GB |

### Aturan Dataset

- Split berdasarkan **`speaker_id`** (bukan random per file) → mencegah speaker leakage
- Scaler, threshold, feature selection **hanya** dari train set
- ASVspoof 2021 **tidak boleh** dipakai untuk tuning apapun
- ASVspoof 2021 hanya digunakan **sekali** setelah konfigurasi dibekukan (minggu 9)

---

## Konfigurasi

Semua parameter ada di [`configs/baseline.yaml`](configs/baseline.yaml). Tidak ada angka yang di-hardcode di source code.

Untuk mengubah parameter, edit file YAML — tidak perlu menyentuh source code.

---

## Output & Hasil

| File | Isi |
|------|-----|
| `results/features_train.csv` | Fitur per utterance (train) |
| `results/features_validation.csv` | Fitur per utterance (validation) |
| `results/features_test.csv` | Fitur per utterance (test) |
| `results/utterance_scores.csv` | Skor deepfake per utterance semua model |
| `results/metrics.csv` | AUC, EER, F1, Precision, Recall per model/split/seed |
| `results/statistical_tests.csv` | Mann-Whitney U, rank-biserial, p-FDR, CI 95% |
| `results/evidence_consistency.csv` | Spearman ρ, Cohen κ, 4 kategori kasus |
| `results/error_analysis.csv` | Kasus dengan keputusan salah + metadata |
| `results/bootstrap_comparisons.csv` | Paired AUC difference + 95% CI |
| `checkpoints/*.joblib` | Model, scaler, threshold, daftar fitur |

---

## Konvensi Penting

| Aturan | Detail |
|--------|--------|
| **Label** | Real/bonafide = `0`, Deepfake = `1` |
| **Skor** | Lebih besar = lebih deepfake (selalu) |
| **Threshold** | Ditentukan **hanya** dari validation set |
| **Split** | Berdasarkan **speaker_id**, bukan file |
| **Dataset eksternal** | **Hanya sekali**, setelah config dibekukan |
| **Reporting** | Metrik utama pada tingkat **utterance**, bukan segmen |
| **Residual** | Disebut "excitation-related evidence", bukan "glottal source" |

---

## Rencana Kerja 12 Minggu

| Minggu | Kegiatan | Target |
|--------|----------|--------|
| 1 | Setup environment, baca proposal, akses dataset | Environment terkunci, daftar pustaka inti |
| 2 | Manifest, audit, split, segmentasi | Manifest bebas leakage, 20 sampel lolos QC |
| 3 | MFCC/LFCC + smoke test baseline | B0–B2 berjalan, EER tersedia |
| 4 | Baseline lengkap + evaluasi internal | B0–B4, threshold validasi, tabel awal |
| 5 | Implementasi residual LPC | Residual features, plot QC, failure log |
| 6 | Implementasi modulasi | Modulation features, spektrum rata-rata |
| 7 | Analisis statistik bukti | Effect size, FDR, fitur kandidat — **bekukan daftar fitur** |
| 8 | Evidence-only model + ablation | Residual-only, mod-only, gabungan |
| 9 | Speaker-independent + 3 seed | Generalisasi, stabilitas — **bekukan konfigurasi** |
| 10 | Cross-dataset + consistency analysis | Skor eksternal, agreement, error cases |
| 11 | Finalisasi gambar/tabel + Bab 3-4 | Semua hasil reproducible dari config |
| 12 | Revisi, presentasi, arsip | Naskah final, README, checklist, backup |

---

## Referensi

1. Yamagishi et al. — [ASVspoof 2021](https://arxiv.org/abs/2109.00537) (arXiv:2109.00537)
2. Bird & Lotfi — [DEEP-VOICE](https://arxiv.org/abs/2308.12734) (arXiv:2308.12734)
3. Drugman et al. — Glottal source processing (Computer Speech & Language, 2014)
4. Kadiri et al. — Excitation information of speech (Proc. IEEE, 2021)
5. Gupta & Patil — Linear frequency residual cepstral features (EUSIPCO, 2022)
6. Borrelli et al. — Synthetic speech detection via prediction traces (EURASIP, 2021)
7. Tamiazzo et al. — Wiener-Hopf linear prediction untuk deepfake (IH&MMSec, 2026)
8. Sadashiv et al. — Modulation spectrogram + SSL (APSIPA ASC, 2025)
9. Lundberg & Lee — SHAP (NeurIPS, 2017)

---

## Checklist Sebelum Training

- [ ] `pip freeze > requirements-lock.txt` sudah disimpan
- [ ] `data/raw/` berisi file audio (tidak diubah)
- [ ] Kolom label di manifest tidak ada `FILL_REQUIRED`
- [ ] `check_leakage.py` exit code 0 (tidak ada leakage)
- [ ] `configs/baseline.yaml` sudah dikonfigurasi sesuai kebutuhan
- [ ] Seed, versi library, dan hash manifest terdokumentasi

---

*Definisi selesai: seluruh hasil utama dapat dihasilkan ulang dari manifest dan konfigurasi yang tersimpan — bukan ketika satu angka akurasi tinggi telah diperoleh.*
