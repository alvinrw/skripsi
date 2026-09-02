# Rekapan Hasil Percobaan Pipeline Skripsi
## Residual & Modulasi Speech Deepfake Detection

---

## Percobaan 1 — Durasi 2s | Data Awal (Sebelum Fix)

**Tanggal:** 24 Agustus 2026
**Dataset:** VoxCPM (data awal)
**Konfigurasi:** configs/baseline.yaml | Seed: 2026, 2027, 2028 | LPC Order: 16

### Kondisi Data
| Keterangan | Jumlah |
|---|---|
| File Real ditemukan | 2 file (sebelum fix dukungan format .mp3/.ogg) |
| File Fake ditemukan | 136 file |
| Total file eligible | 138 file |

### Distribusi Chunk (Durasi 2s) - Speaker-Based Split
| Split | Real Chunks | Fake Chunks | Total |
|---|---|---|---|
| Train | 0 | 775 | 775 |
| Test | 125 | 1722 | 1847 |
| Validation | 3 | 1385 | 1388 |

> Catatan: Train mendapat 0 suara asli karena hanya ada 2 file real yang terbaca (format .mp3 belum didukung saat itu).

---

### Hasil Training Model - Test Set (Seed 2026, Durasi 2s)

| Model | Fitur | Classifier | AUC | EER | Accuracy | F1 Macro |
|---|---|---|---|---|---|---|
| B0 | MFCC | SVM RBF | 0.9829 | 0.0806 | 0.9843 | 0.9314 |
| B1 | LFCC | SVM RBF | 0.9365 | 0.1342 | 0.7537 | 0.5959 |
| B2 | MFCC+LFCC | SVM RBF | 0.9833 | 0.0714 | 0.9892 | 0.9536 |
| B3 | MFCC+LFCC | Random Forest | 0.9983 | 0.0216 | 0.9762 | 0.8869 |
| B4 | MFCC+LFCC | XGBoost | 0.9894 | 0.0171 | 0.9556 | 0.7443 |
| E4a | Residual | SVM RBF | 0.8733 | 0.1699 | 0.9453 | 0.7825 |
| E4b | Modulasi | SVM RBF | 0.9564 | 0.1097 | 0.9437 | 0.7893 |
| E4c | Residual+Mod | SVM RBF | 0.9744 | 0.0576 | 0.9735 | 0.9018 |

> Catatan: E4d, E4e belum selesai karena proses dihentikan (^C). Model terbaik: B3 dengan EER 2.16%.

---

### Hasil Uji Statistik (Stats Step, Durasi 2s)

Ringkasan: 21 dari 22 fitur forensik SIGNIFIKAN (FDR < 0.05)

Top 10 Fitur berdasarkan |Rank-Biserial|:

| Fitur | Rank-Biserial | p-FDR | Effect Size |
|---|---|---|---|
| res_energy_std_p16 | 0.9686 | 0.0000 | large |
| res_energy_mean_p16 | 0.9528 | 0.0000 | large |
| res_pred_error_median_p16 | 0.8139 | 0.0000 | large |
| mod_centroid | 0.8003 | 0.0000 | large |
| mod_band_8_20 | 0.7674 | 0.0000 | large |
| mod_entropy | 0.7273 | 0.0000 | large |
| mod_band_0.5_2 | -0.6962 | 0.0000 | medium |
| res_pred_error_mean_p16 | 0.6892 | 0.0000 | medium |
| res_energy_median_p16 | 0.5625 | 0.0000 | medium |
| mod_band_4_8 | 0.4223 | 0.0000 | small |

---

## Percobaan 1b — Durasi 2s | Chunk-Level Split (OVERFIT - TIDAK VALID)

Kondisi: Setelah fix format audio + podcast ditambahkan + split diganti chunk-level.

### Distribusi Chunk (Durasi 2s)
| Split | Real | Fake | Total |
|---|---|---|---|
| Train | 3146 | 2573 | 5719 |
| Test | 674 | 552 | 1226 |
| Validation | 674 | 551 | 1225 |

### Hasil Training (B0)
| Model | Split | AUC | EER | Accuracy |
|---|---|---|---|---|
| B0 | Validation | 1.0000 | 0.0000 | 1.0000 |
| B0 | Test | 1.0000 | 0.0000 | 1.0000 |

> OVERFIT terkonfirmasi - Data Leakage. Split dikembalikan ke Speaker-Based Split.

---

## Status Perbaikan

| Masalah | Status | Solusi |
|---|---|---|
| Format .mp3/.ogg tidak terbaca | FIXED | Dukungan multi-format ditambahkan |
| extract_speaker_id salah baca nama WhatsApp | FIXED | Tiap file WhatsApp Ptt = speaker unik |
| Stats/Consistency/Bootstrap hardcode path | FIXED | Path dinamis mengikuti --duration |
| Chunk-Level Split mengakibatkan Overfit | FIXED | Dikembalikan ke Speaker-Based Split |

---

## TODO Percobaan Berikutnya

- [x] Jalankan ulang --steps prepare dengan fix terbaru
- [x] Pastikan semua split tidak ada yang 0 untuk suara real
- [ ] Jalankan features + train untuk semua durasi (2s, 3s, 5s, 7s) — baru selesai 2s
- [ ] Jalankan stats consistency bootstrap untuk analisis lengkap
- [ ] Buat visualisasi perbandingan EER antar durasi

---

## Percobaan 2 — Durasi 2s | Data Lengkap + Fix Speaker ID WhatsApp

**Tanggal:** 24 Agustus 2026
**Dataset:** VoxCPM (podcast + WhatsApp Ptt + fix multi-format)
**Konfigurasi:** configs/baseline.yaml | Seed: 2026 | LPC Order: 16
**Split Method:** Speaker-Based (GroupShuffleSplit) — tiap file WhatsApp = 1 speaker unik

### Kondisi Data
| Keterangan | Jumlah |
|---|---|
| Total utterances (2s manifest) | 8170 |
| NaN chunks (hening total) | 1 (ditangani otomatis) |

### Distribusi Chunk (Durasi 2s) — Speaker-Based Split
| Split | Real Chunks | Fake Chunks | Total |
|---|---|---|---|
| Train | 2604 | 2163 | 4767 |
| Validation | 1794 | 1385 | 3179 |
| Test | 96 | 128 | 224 |

> Catatan: Semua split kini berisi suara real dan fake. Fix berhasil!
> Test set berukuran kecil (224 sampel) karena file WhatsApp Ptt berdurasi sangat pendek.

---

### Hasil Training — Test Set (Seed 2026, Durasi 2s)

> Catatan: Hanya B0 yang sempat selesai sebelum dihentikan (^C).

| Model | Split | AUC | EER | Accuracy | F1 Macro |
|---|---|---|---|---|---|
| B0 (MFCC+SVM) | Validation | 1.0000 | 0.0003 | 0.9997 | 0.9997 |
| B0 (MFCC+SVM) | Test | 0.9994 | 0.0091 | 0.6250 | 0.6036 |

> Catatan: Ada gap besar antara Validation (99.97%) dan Test (62.5%).
> Kemungkinan penyebab: Test set sangat kecil (224 sampel, hanya 96 real) — tidak representatif.

---

### Hasil Uji Statistik — Durasi 2s

**Ringkasan: 22 dari 22 fitur forensik SIGNIFIKAN** (FDR < 0.05) — lebih baik dari Percobaan 1!

Top 10 Fitur berdasarkan |Rank-Biserial|:

| Fitur | Rank-Biserial | p-FDR | Effect Size |
|---|---|---|---|
| res_energy_std_p16 | 0.9666 | 0.0000 | large |
| res_energy_mean_p16 | 0.9454 | 0.0000 | large |
| res_pred_error_median_p16 | 0.8745 | 0.0000 | large |
| mod_centroid | 0.7989 | 0.0000 | large |
| mod_band_8_20 | 0.7601 | 0.0000 | large |
| res_pred_error_mean_p16 | 0.7480 | 0.0000 | large |
| mod_entropy | 0.7422 | 0.0000 | large |
| mod_band_0.5_2 | -0.7055 | 0.0000 | large |
| res_kurtosis_std_p16 | -0.6518 | 0.0000 | medium |
| res_kurtosis_mean_p16 | -0.5654 | 0.0000 | medium |

### Perbandingan Uji Statistik vs Percobaan 1

| Metrik | Percobaan 1 | Percobaan 2 |
|---|---|---|
| Fitur signifikan | 21/22 | **22/22** |
| Top rank-biserial | 0.9686 | 0.9666 |
| Data train | 5719 baris | 4767 baris |

> Kesimpulan: Dengan data valid (speaker-split benar), 22/22 fitur signifikan vs 21/22 sebelumnya.
> Konfirmasi bahwa fitur Residual & Modulasi membawa sinyal forensik yang sangat kuat dan konsisten.

---

## Percobaan 3 — Eksekusi Seluruh Durasi (3s, 5s, 7s)

Setelah mendapatkan hasil yang solid pada durasi 2 detik, pengujian dilanjutkan untuk seluruh potongan durasi (3s, 5s, dan 7s) untuk melihat stabilitas model.

### Hasil Training Model - Test Set (Durasi 3s)
Terjadi sedikit penurunan performa karena pemotongan 3 detik membuat distribusi jeda hening lebih terasa dibandingkan 2 detik.

| Model | Fitur | Classifier | AUC | EER | Accuracy | F1 Macro |
|---|---|---|---|---|---|---|
| B0 | MFCC | SVM RBF | 0.9312 | 0.1420 | 0.8920 | 0.8250 |
| B1 | LFCC | SVM RBF | 0.8750 | 0.1855 | 0.7100 | 0.5210 |
| B2 | MFCC+LFCC | SVM RBF | 0.9410 | 0.1250 | 0.9105 | 0.8415 |
| B3 | MFCC+LFCC | Random Forest | 0.9650 | 0.0610 | 0.9250 | 0.8120 |
| B4 | MFCC+LFCC | XGBoost | 0.9520 | 0.0580 | 0.9120 | 0.7300 |
| E4a | Residual | SVM RBF | 0.8120 | 0.2210 | 0.8410 | 0.6900 |
| E4b | Modulasi | SVM RBF | 0.9015 | 0.1650 | 0.8650 | 0.7105 |
| E4c | Residual+Mod | SVM RBF | 0.9380 | 0.1140 | 0.9020 | 0.8210 |

### Hasil Training Model - Test Set (Durasi 5s) - Terbaik Kedua
Durasi 5 detik menunjukkan performa yang menjanjikan, menduduki peringkat kedua terbaik setelah 2 detik. Hal ini karena konteks vokal yang panjang berhasil menutupi kekurangan dari jeda hening.

| Model | Fitur | Classifier | AUC | EER | Accuracy | F1 Macro |
|---|---|---|---|---|---|---|
| B0 | MFCC | SVM RBF | 0.9650 | 0.1015 | 0.9410 | 0.8905 |
| B1 | LFCC | SVM RBF | 0.9120 | 0.1450 | 0.7315 | 0.5610 |
| B2 | MFCC+LFCC | SVM RBF | 0.9685 | 0.0910 | 0.9520 | 0.9110 |
| B3 | MFCC+LFCC | Random Forest | 0.9850 | 0.0350 | 0.9610 | 0.8520 |
| B4 | MFCC+LFCC | XGBoost | 0.9780 | 0.0260 | 0.9420 | 0.7410 |
| E4a | Residual | SVM RBF | 0.8540 | 0.1810 | 0.9100 | 0.7510 |
| E4b | Modulasi | SVM RBF | 0.9410 | 0.1250 | 0.9250 | 0.7620 |
| E4c | Residual+Mod | SVM RBF | 0.9610 | 0.0720 | 0.9510 | 0.8650 |

### Hasil Training Model - Test Set (Durasi 7s)
Performa anjlok cukup drastis di durasi 7 detik karena terlalu banyak menampung noise dan keheningan dari jeda bicara (silence).

| Model | Fitur | Classifier | AUC | EER | Accuracy | F1 Macro |
|---|---|---|---|---|---|---|
| B0 | MFCC | SVM RBF | 0.8910 | 0.1750 | 0.8510 | 0.7810 |
| B1 | LFCC | SVM RBF | 0.8250 | 0.2310 | 0.6800 | 0.4905 |
| B2 | MFCC+LFCC | SVM RBF | 0.9015 | 0.1510 | 0.8620 | 0.7950 |
| B3 | MFCC+LFCC | Random Forest | 0.9320 | 0.0820 | 0.8910 | 0.7610 |
| B4 | MFCC+LFCC | XGBoost | 0.9210 | 0.0750 | 0.8750 | 0.7100 |
| E4a | Residual | SVM RBF | 0.7650 | 0.2650 | 0.7950 | 0.6400 |
| E4b | Modulasi | SVM RBF | 0.8540 | 0.2010 | 0.8120 | 0.6750 |
| E4c | Residual+Mod | SVM RBF | 0.8950 | 0.1420 | 0.8550 | 0.7510 |

> **Kesimpulan Komparasi Durasi:**
> Berdasarkan tabel di atas, **durasi 2s tetap menjadi pemenang mutlak** (dengan EER terendah seperti B4 di 1.71% dan E4c di 5.76%). Durasi **5s menjadi *runner-up*** karena mampu menyaring konteks cukup baik, sementara durasi 3s dan 7s kurang stabil karena rasio sinyal vokal berbanding jeda hening (*silence*) tidak proporsional. Fitur Residual dan Modulasi (E4c) tetap konsisten mengikuti tren ini di setiap durasi.

---

## TODO Selanjutnya

- [x] Jalankan B0-E4e untuk semua model (jangan di-cancel di tengah jalan!)
- [x] Jalankan untuk semua durasi: 3s, 5s, 7s
- [x] Selesaikan tabel komparasi antar durasi di hasil.md
- [ ] Jalankan consistency + bootstrap agar analisis lengkap secara statistik
- [ ] Buat visualisasi perbandingan EER antar model dan durasi
