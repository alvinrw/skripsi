# Daftar Revisi & To-Do List Skripsi

- [ ] **Perbaikan Bug Speaker ID (Data Leakage)**
  - **Lokasi:** `src/build_manifest.py` (fungsi `infer_speaker_from_path`)
  - **Masalah:** Saat ini kode membaca nama folder induk (`REAL` atau `FAKE`) sebagai `speaker_id`. Akibatnya, sistem `make_splits.py` menganggap hanya ada 2 pembicara, sehingga gagal melakukan *GroupShuffleSplit* dan beralih ke *Random Split* (menyebabkan *data leakage*).
  - **Solusi:** Ubah heuristik agar mengambil nama *speaker* langsung dari nama file (misalnya mengambil teks sebelum tanda strip `-` pada `taylor-original.wav` atau `trump-to-biden.wav`).

- [ ] **Memperbanyak Jumlah Data Audio (Utterance)**
  - **Masalah:** Saat ini jumlah data audio (sampel) masih sangat sedikit (hanya sekitar belasan/puluhan file), sehingga model Machine Learning belum bisa belajar secara optimal dan validasi statistik belum bisa dipercaya sepenuhnya.
  - **Solusi:** Unduh dan tambahkan lebih banyak file audio ke dalam folder `data/raw/` agar variasi suara dan jumlah *speaker* bertambah, lalu jalankan ulang `make_splits.py`.
  - > 💡 **Saran dari Antigravity (Boleh dipertimbangkan, tidak wajib):**
  - > 1. **Full Kaggle DEEP-VOICE Dataset:** Saat ini sepertinya Anda baru mengambil sebagian kecil (subset) dari dataset ini. Sangat disarankan untuk mengunduh keseluruhan filenya dari Kaggle agar *pipeline* skripsi Anda berjalan sempurna.
  - > 2. **ASVspoof 2021 (Logical Access / Deepfake):** Ini adalah standar internasional (Gold Standard) untuk kompetisi audio forensik. Penguji skripsi pasti akan sangat mengapresiasi jika Anda menggunakannya (minimal sebagai data pengujian lintas-dataset).
  - > 3. **WaveFake Dataset:** Jika Anda sangat fokus pada fitur *Residual* dan *Glottal*, dataset ini dibuat khusus dari generasi *waveform* mentah yang artifaknya lebih kentara di level sinyal residual. Sangat cocok untuk memperkuat argumen bab 4 Anda!

- [ ] *(Tambahkan daftar to-do berikutnya di sini...)*
