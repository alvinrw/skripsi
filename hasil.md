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

- [ ] Jalankan ulang --steps prepare dengan fix terbaru
- [ ] Pastikan semua split tidak ada yang 0 untuk suara real
- [ ] Jalankan features + train untuk semua durasi (2s, 3s, 5s, 7s)
- [ ] Jalankan stats consistency bootstrap untuk analisis lengkap
- [ ] Buat visualisasi perbandingan EER antar durasi
