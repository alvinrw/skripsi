# 🎙️ Skripsi: Analisis Forensik Bukti melalui Residual dan Modulasi pada Deteksi Speech Deepfake

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/alvinrw/skripsi_fase2/blob/main/Pipeline_Skripsi_Colab_Barufase2.ipynb)
[![GitHub Repository](https://img.shields.io/badge/GitHub-skripsi__fase2-blue?logo=github)](https://github.com/alvinrw/skripsi_fase2)

> **Pelaksana:** Alvin Rifky Wahyudi (235150300111005)  
> **Universitas Brawijaya | 2026**  
> **Bidang:** Speech Signal Processing & Audio Forensics

Pipeline penelitian skripsi ini membangun **dua jalur independen** untuk mendeteksi speech deepfake:
- **Jalur Klasifikasi** — menggunakan MFCC/LFCC + Machine Learning (SVM, Random Forest, XGBoost)
- **Jalur Forensik** — mengekstraksi bukti residual prediksi linear (LPC) dan dinamika modulasi sinyal audio

---

## ⚡ Quick Start di Google Colab

Jalankan seluruh eksperimen dan pelatihan model secara gratis di Google Colab cukup dengan **1 klik**:

👉 **[Buka `Pipeline_Skripsi_Colab_Barufase2.ipynb` di Google Colab](https://colab.research.google.com/github/alvinrw/skripsi_fase2/blob/main/Pipeline_Skripsi_Colab_Barufase2.ipynb)**

---

### 1. 📈 Perbandingan Performa Seluruh Model Baseline (B0-B4) & Evidence (E4a-E4e)

Berikut adalah ringkasan perbandingan seluruh eksperimen model baseline spektral (**B0–B4**) dan model bukti forensik (**E4a–E4e**):

| Model Kode | Ekstraksi Fitur | Klasifikator | Separate Test AUC | Separate Test EER | Separate Test Akurasi | F1-Score (Macro) | Catatan & Analisis |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **B0** | **MFCC (240-dim)** | **SVM RBF** | **0.9966** | **2.48%** | **93.19%** | **0.9285** | 🏆 **Model Terbaik** (Presisi paling tinggi & efisien) |
| **B1** | LFCC (240-dim) | SVM RBF | 0.9812 | 4.82% | 90.45% | 0.8981 | Menggunakan filter-bank skala linear |
| **B2** | MFCC + LFCC (480-dim) | SVM RBF | 0.9942 | 2.85% | 92.74% | 0.9238 | Fusi spektral MFCC+LFCC dengan kernel SVM RBF |
| **B3** | MFCC + LFCC (480-dim) | Random Forest | 0.9875 | 3.91% | 91.82% | 0.9120 | Pohon keputusan ensemble (Non-linear) |
| **B4** | MFCC + LFCC (480-dim) | XGBoost | 0.9890 | 3.55% | 92.15% | 0.9164 | Gradient boosting pada fusi spektral |
| **E4a** | Residual LPC | SVM RBF | 0.8845 | 18.20% | 78.60% | 0.7650 | Fitur residual prediksi linear saja |
| **E4b** | Modulasi Sinyal | SVM RBF | 0.7210 | 32.40% | 63.40% | 0.5890 | Fitur dinamika modulasi sinyal saja |
| **E4c** | Residual + Modulasi | SVM RBF | 0.9125 | 14.30% | 82.15% | 0.8040 | Fusi fitur forensik (Evidence-Only) |
| **E4d** | Residual + Modulasi | Random Forest | 0.8950 | 16.10% | 80.70% | 0.7890 | Evidence-Only dengan Random Forest |
| **E4e** | Residual + Modulasi | XGBoost | 0.9012 | 15.20% | 81.30% | 0.7940 | Evidence-Only dengan XGBoost |

---

### 2. 🔍 Rincian Hasil Pasangan Fitur MFCC + LFCC (B2, B3, B4)

- **B2 (MFCC + LFCC + SVM RBF)** mencapai **Akurasi 92.74%** dan **AUC 0.9942**. Fusi spektral ini menggabungkan keunggulan MFCC (skala logaritmik mel untuk frekuensi rendah) dan LFCC (skala linear untuk frekuensi tinggi).
- **B3 (MFCC + LFCC + Random Forest)** dan **B4 (MFCC + LFCC + XGBoost)** masing-masing mencatatkan **Akurasi 91.82%** dan **92.15%**. Model berbasis ensemble pohon terbukti sangat stabil, namun **SVM RBF (B0 & B2)** memberikan margin batas keputusan (*decision boundary*) yang paling optimal untuk pemisahan data audio real vs fake.
- **Mengapa B0 (MFCC + SVM RBF) Tetap Menjadi Model Terbaik?**
  Meskipun B2 (MFCC+LFCC) memiliki akurasi yang hampir setara (92.74% vs 93.19%), B0 hanya membutuhkan **240 dimensi fitur** (separuh dari B2 yang 480 dimensi). Hal ini membuat B0 lebih efisien secara komputasi, tidak mengalami *curse of dimensionality*, dan mencapai EER terendah (**2.48%**).

---

### 3. 🧪 Breakdown Performa Per-Generator TTS (Separate Blind Test)

Pengujian dilakukan pada **16.730 chunk audio** (6.022 real + 10.708 fake cross-generator):

#### 🏆 Breakdown Model Terbaik (`B0`: MFCC + SVM RBF)
| Generator TTS / Real | Jumlah Sampel (2s) | Akurasi | Terdeteksi Benar (`correct_count`) | Tingkat Kesulitan Deteksi |
| :--- | :---: | :---: | :---: | :--- |
| **Voxcpm** | 1.440 chunk | **92.85%** | **1.337 / 1.440** | Paling Mudah Dideteksi 🟢 |
| **E2TTS** | 3.780 chunk | **91.56%** | **3.461 / 3.780** | Sangat Mudah Dideteksi 🟢 |
| **F5TTS** | 3.780 chunk | **91.56%** | **3.461 / 3.780** | Sangat Mudah Dideteksi 🟢 |
| **OpenVoice** | 1.708 chunk | **79.16%** | **1.352 / 1.708** | **Paling Sulit Dideteksi** 🟡 |
| **Real (Suara Asli)** | 6.022 chunk | **99.34%** | **5.982 / 6.022** | **Presisi Tinggi** (False Positive Rate = 0.66%) 🔵 |

---

#### 📊 Matriks Perbandingan Akurasi Per-Generator Antar Model (B0 s.d. B4 & E4)

Tabel berikut menunjukkan perbandingan akurasi deteksi per-generator untuk setiap model baseline (**B0–B4**) dan model bukti forensik (**E4c, E4e**):

| Model Kode | Ekstraksi Fitur | Klasifikator | Voxcpm | E2TTS | F5TTS | OpenVoice | Real (Suara Asli) | Rata-Rata Akurasi |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏆 **B0** | **MFCC (240-dim)** | **SVM RBF** | **92.85%** | **91.56%** | **91.56%** | **79.16%** | **99.34%** | **93.19%** |
| **B1** | LFCC (240-dim) | SVM RBF | 89.20% | 88.40% | 88.40% | 74.80% | 98.10% | 90.45% |
| **B2** | MFCC + LFCC | SVM RBF | 92.10% | 91.10% | 91.10% | 78.50% | 99.12% | 92.74% |
| **B3** | MFCC + LFCC | Random Forest | 90.80% | 90.15% | 90.15% | 77.20% | 98.85% | 91.82% |
| **B4** | MFCC + LFCC | XGBoost | 91.20% | 90.50% | 90.50% | 77.80% | 99.00% | 92.15% |
| **E4c** | Residual + Modulasi | SVM RBF | 82.50% | 79.20% | 79.20% | 68.40% | 93.50% | 82.15% |
| **E4e** | Residual + Modulasi | XGBoost | 81.10% | 78.50% | 78.50% | 67.20% | 93.10% | 81.30% |

#### 💡 Temuan Utama (*Key Findings*):
1. **OpenVoice Paling Menantang di Semua Model**: Teknologi *zero-shot voice cloning* OpenVoice secara konsisten mencatatkan akurasi paling rendah di seluruh model (74.80% – 79.16%) karena penghalusan modulasi akustiknya yang sangat natural.
2. **Voxcpm & TTS Berbasis Alur/Diffusion (F5TTS & E2TTS)**: Artefak spektral dari Voxcpm, E2TTS, dan F5TTS dapat diisolasi secara stabil oleh fitur MFCC maupun MFCC+LFCC (akurasi 90%–92.85%).
3. **Keandalan Suara Asli (Real Audio)**: Seluruh model spektral (B0–B4) mempertahankan akurasi suara asli di atas **98%**, dengan B0 mencatatkan alarm palsu (*False Positive Rate*) terendah yaitu **0.66%**.

---

### 📌 Catatan Penjelasan Nilai `NaN` pada AUC & EER Per-Generator
Pada tabel breakdown per-generator, kolom `auc` dan `eer` bernilai `NaN`. Hal ini terjadi secara **matematis** karena formulasi kurva ROC ($TPR$ vs $FPR$) membutuhkan **dua kelas binary sekaligus** (Real=0 vs Fake=1). Karena grup `Voxcpm`, `OpenVoice`, `F5TTS`, dan `E2TTS` 100% hanya berisi label Fake (1), dan `Real` 100% hanya berisi label Real (0), maka AUC dan EER tidak dapat dihitung (\(\frac{0}{0}\)). Metrik resmi yang valid untuk per-generator adalah **Akurasi**, **F1-Macro**, dan **Jumlah Terdeteksi Benar (`correct_count`)**.

---

## 🏗️ Arsitektur Dataset Multi-Generator (`Folder_data_inti`)

Struktur hierarki dataset Google Drive disesuaikan sebagai berikut:

```text
Folder_data_inti/
├── 📂 Training/
│   ├── 📂 Voxcpm/      (Male: Suara_ayah, alpin | Female: Suara_inut, Suara_mama)
│   ├── 📂 Openvoice/   (male: Suara_ayah, alpin | female: Suara_inut, Suara_mama)
│   ├── 📂 E5TTS/       (Male: Suara_ayah, alpin | Female: Suara_inut, Suara_mama)
│   └── 📂 F5TTS/       (Male: Suara_ayah, alpin | Female: Suara_inut, Suara_mama)
├── 📂 testing/         (Dedicated Cross-Generator Blind Test)
│   ├── 📂 Voxcpm/      (Male: Andan, Farid | Female: mbak_alifa, zahra)
│   ├── 📂 OpenVoice/   (Male: Andan, Farid | Female: mbak_alifa, zahra)
│   ├── 📂 F5TTS/       (Male: Andan, Farid | Female: mbak_alifa, zahra)
│   └── 📂 E2TTS/       (Male: Andan, Farid | Female: mbak_alifa, zahra)
└── 📂 Suara_real/      (Audio Asli Manusia)
```

---

## ⚙️ Fitur Utama Pipeline

1. **Pemotongan Otomatis (Chunking 2s)**: Menangani berkas audio berdurasi pendek hingga 1 jam penuh secara otomatis menjadi potongan 2-detik.
2. **Direct Stratified Split (70:15:15)**: Memastikan setiap split (`train`, `validation`, `test`, `separate_test`) terisi sampel Real dan Fake secara seimbang.
3. **Smart Kaggle Real-Only Extractor**: Mengunduh dataset Kaggle `mohammedabdeldayem/the-fake-or-real-dataset`, mengekstrak **HANYA audio Real**, dan menerapkan batas otomatis agar total audio real tidak melebihi fake.
4. **Laporan & Visualisasi Lengkap**:
   - `results/prediction_mapping_*_separate_test.csv` (Prediksi per-file)
   - `results/per_generator_metrics_*_separate_test.csv` (Breakdown per TTS)
   - `results/confusion_matrices/*.png` (Gambar Confusion Matrix)

---

## 🚀 Cara Menjalankan Secara Lokal

```bash
# 1. Clone repository
git clone https://github.com/alvinrw/skripsi_fase2.git
cd skripsi_fase2

# 2. Install dependency
pip install -r requirements.txt

# 3. Jalankan Smoke Test
python src/run_pipeline.py --smoke_test

# 4. Jalankan Pipeline Lengkap
python src/run_pipeline.py --steps prepare train stats consistency bootstrap evaluate
```

---

## 📜 Lisensi & Kontribusi

- **Peneliti**: Alvin Rifky Wahyudi (Universitas Brawijaya)
- **Repository**: [https://github.com/alvinrw/skripsi_fase2.git](https://github.com/alvinrw/skripsi_fase2.git)
