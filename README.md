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

## 📊 Hasil Eksperimen & Analisis Akademik (Bab IV Skripsi)

### 1. 📈 Ringkasan Performa Model Baseline (`B0`: MFCC + SVM RBF)

| Datasets / Split | AUC | EER (Equal Error Rate) | Akurasi | F1-Score (Macro) | Interpretasi Akademik |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Validation Set** | **1.0000** | **0.00%** | **100.00%** | **1.0000** | Konvergensi sempurna pada data latih internal |
| **In-Domain Test Set** | **0.7760** | **27.12%** | **74.47%** | **0.7045** | Generalisasi pada pembicara baru (in-domain split) |
| **Separate Blind Test** | **0.9966** | **2.48%** | **93.19%** | **0.9285** | **Sangat Unggul & Konsisten** pada uji silang 4 generator TTS |

---

### 2. 🧪 Breakdown Performa Per-Generator TTS (Separate Blind Test)

Pengujian dilakukan pada **16.730 chunk audio** (6.022 real + 10.708 fake cross-generator):

| Generator TTS / Real | Jumlah Sampel (2s) | Akurasi | Terdeteksi Benar (`correct_count`) | Tingkat Kesulitan Deteksi |
| :--- | :---: | :---: | :---: | :--- |
| **Voxcpm** | 1.440 chunk | **92.85%** | **1.337 / 1.440** | Paling Mudah Dideteksi 🟢 |
| **E2TTS** | 3.780 chunk | **91.56%** | **3.461 / 3.780** | Sangat Mudah Dideteksi 🟢 |
| **F5TTS** | 3.780 chunk | **91.56%** | **3.461 / 3.780** | Sangat Mudah Dideteksi 🟢 |
| **OpenVoice** | 1.708 chunk | **79.16%** | **1.352 / 1.708** | **Paling Sulit Dideteksi** 🟡 |
| **Real (Suara Asli)** | 6.022 chunk | **99.34%** | **5.982 / 6.022** | **Presisi Tinggi** (False Positive Rate = 0.66%) 🔵 |

#### 💡 Temuan Utama (*Key Findings*):
1. **OpenVoice Paling Menantang**: Teknologi *zero-shot voice cloning* OpenVoice menghasilkan karakteristik akustik yang paling mendekati suara asli manusia, sehingga akurasi deteksinya berada di 79.16%.
2. **Artefak Voxcpm Sangat Menonjol**: Residual spektral Voxcpm paling mudah diisolasi oleh model (92.85%).
3. **Keandalan Suara Asli**: Model mencatatkan tingkat alarm palsu (*False Positive Rate*) yang sangat rendah yaitu hanya **0.66%** (hanya 40 dari 6.022 chunk suara asli yang terdegradasi salah).

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
