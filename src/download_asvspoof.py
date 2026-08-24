"""
src/download_asvspoof.py
========================
Panduan dan helper untuk mendapatkan dataset ASVspoof 2021 DF
(Deepfake track) sebagai dataset eksternal cross-dataset test.

PENTING (dari panduan skripsi):
    - ASVspoof 2021 DF HANYA digunakan untuk cross-dataset test
    - DILARANG dipakai untuk tuning, pemilihan fitur, scaler, atau threshold
    - Hanya digunakan SEKALI setelah konfigurasi final dibekukan (minggu 9-10)
    - Bila dataset ini belum tersedia, selesaikan semua eksperimen DEEP-VOICE dulu

Cara pakai:
    python src/download_asvspoof.py              # tampilkan panduan lengkap
    python src/download_asvspoof.py --verify     # verifikasi bila sudah didownload
    python src/download_asvspoof.py --add_manifest  # tambah ke manifest sebagai 'external'
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


# ──────────────────────────────────────────────
# Panduan download ASVspoof 2021
# ──────────────────────────────────────────────

GUIDE = """
============================================================
CARA MENDAPATKAN ASVspoof 2021 DF (Deepfake Track)
============================================================

ASVspoof 2021 tidak bisa di-download langsung seperti Kaggle.
Harus lewat proses registrasi resmi.

LANGKAH-LANGKAH:

1. Buka halaman resmi:
   https://www.asvspoof.org/index2021.html

2. Klik "Request Access" atau "Dataset Download"
   (biasanya perlu login/daftar dengan email institusi)

3. Isi form registrasi dengan data:
   - Nama lengkap: Alvin Rifky Wahyudi
   - Institusi: Universitas Brawijaya
   - Tujuan: Penelitian skripsi deteksi speech deepfake
   - Email: (gunakan email UB)

4. Tunggu email konfirmasi (biasanya 1-3 hari kerja)

5. Setelah mendapat link download, unduh bagian:
   *** ASVspoof2021_DF_eval (Deepfake track - evaluation set) ***
   Ukuran: ~11 GB

ALTERNATIF (bila akses lambat):
   Beberapa universitas menyediakan mirror. Tanyakan ke dosen pembimbing.

SETELAH DOWNLOAD:
   Ekstrak ke:
   data/raw/ASVspoof2021_DF/

   Struktur yang diharapkan:
   data/raw/ASVspoof2021_DF/
   ├── flac/              ← file audio (format .flac)
   └── trial_metadata.txt ← metadata label & generator

CATATAN PENTING:
   Sesuai panduan skripsi, bila dataset ini belum tersedia
   saat minggu ke-2, lanjutkan semua eksperimen dengan DEEP-VOICE.
   Dokumentasikan tanggal dan proses permintaan akses.
============================================================
"""


# ──────────────────────────────────────────────
# Parse metadata ASVspoof 2021
# ──────────────────────────────────────────────

def parse_asvspoof_metadata(
    metadata_path: str,
    audio_dir: str,
    out_csv: str = "manifests/asvspoof2021_manifest.csv",
) -> pd.DataFrame:
    """
    Parse file trial_metadata.txt dari ASVspoof 2021 DF
    dan buat manifest CSV yang kompatibel dengan pipeline.

    Format baris trial_metadata.txt:
        <speaker_id> <utterance_id> <env> <attack_id> <label>
        Contoh: LA_0001 LA_E_1000001 - - bonafide

    Parameters
    ----------
    metadata_path : path ke trial_metadata.txt
    audio_dir     : folder berisi file .flac
    out_csv       : output manifest CSV
    """
    if not Path(metadata_path).exists():
        raise FileNotFoundError(f"Metadata tidak ditemukan: {metadata_path}")

    audio_path = Path(audio_dir)
    rows = []

    with open(metadata_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            speaker_id   = parts[0]
            utterance_id = parts[1]
            attack_id    = parts[3]  # '-' bila bonafide
            label_str    = parts[4].lower()  # 'bonafide' atau 'spoof'

            # Konversi label ke konvensi pipeline (0=real, 1=fake)
            label = 0 if label_str == "bonafide" else 1
            generator_id = "bonafide" if label == 0 else attack_id

            # Cari file audio (.flac)
            audio_file = audio_path / f"{utterance_id}.flac"
            if not audio_file.exists():
                audio_file = audio_path / f"{utterance_id}.wav"

            rows.append({
                "utterance_id": utterance_id,
                "file_path":    str(audio_file),
                "speaker_id":   speaker_id,
                "label":        label,
                "generator_id": generator_id,
                "dataset":      "ASVspoof2021_DF",
                "split":        "external",   # SELALU external, tidak masuk train/val/test
                "duration_s":   None,         # akan diisi saat build_manifest
                "sample_rate":  None,
                "sha256":       None,
            })

    df = pd.DataFrame(rows)

    # Ringkasan
    n_real = (df["label"] == 0).sum()
    n_fake = (df["label"] == 1).sum()
    print(f"[asvspoof] Parsed {len(df)} utterances: real={n_real}, fake={n_fake}")

    # Simpan
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[asvspoof] Manifest saved -> {out_csv}")

    return df


# ──────────────────────────────────────────────
# Verifikasi folder ASVspoof
# ──────────────────────────────────────────────

def verify_asvspoof(data_dir: str = "data/raw/ASVspoof2021_DF") -> bool:
    """
    Periksa apakah dataset ASVspoof 2021 DF sudah tersedia dan valid.
    """
    base = Path(data_dir)

    print(f"\n[verify] Memeriksa: {base}")

    if not base.exists():
        print(f"  [NOT FOUND] Folder belum ada: {base}")
        print("  Jalankan: python src/download_asvspoof.py untuk panduan download")
        return False

    # Hitung file audio
    audio_files = list(base.rglob("*.flac")) + list(base.rglob("*.wav"))
    print(f"  File audio ditemukan: {len(audio_files)}")

    # Cek metadata
    meta_path = base / "trial_metadata.txt"
    if meta_path.exists():
        with open(str(meta_path)) as f:
            n_lines = sum(1 for l in f if l.strip())
        print(f"  trial_metadata.txt: {n_lines} baris")
    else:
        print("  [WARNING] trial_metadata.txt tidak ditemukan")

    if len(audio_files) > 0:
        print(f"\n  [OK] ASVspoof 2021 DF tersedia ({len(audio_files)} files)")
        return True
    else:
        print(f"\n  [NOT READY] Dataset belum lengkap")
        return False


# ──────────────────────────────────────────────
# Tambah ASVspoof ke manifest utama
# ──────────────────────────────────────────────

def add_to_main_manifest(
    asvspoof_manifest: str = "manifests/asvspoof2021_manifest.csv",
    main_manifest: str = "manifests/split_manifest.csv",
) -> None:
    """
    Gabungkan manifest ASVspoof ke split_manifest.csv sebagai 'external' split.
    Hanya dilakukan SETELAH konfigurasi final dibekukan (minggu 9).
    """
    if not Path(asvspoof_manifest).exists():
        print(f"[ERROR] {asvspoof_manifest} belum ada. Buat dulu dengan --parse_metadata")
        return

    if not Path(main_manifest).exists():
        print(f"[ERROR] {main_manifest} belum ada. Jalankan make_splits.py dulu.")
        return

    df_asv  = pd.read_csv(asvspoof_manifest)
    df_main = pd.read_csv(main_manifest)

    # Pastikan tidak ada duplikasi
    existing_ids = set(df_main["utterance_id"].tolist())
    df_asv_new = df_asv[~df_asv["utterance_id"].isin(existing_ids)]

    df_combined = pd.concat([df_main, df_asv_new], ignore_index=True)
    df_combined.to_csv(main_manifest, index=False)

    print(f"[add_manifest] Ditambahkan {len(df_asv_new)} utterances ASVspoof ke {main_manifest}")
    print(f"  Total manifest: {len(df_combined)} utterances")
    print(f"  INGAT: Split 'external' tidak boleh digunakan untuk training/tuning!")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Panduan dan helper ASVspoof 2021 DF")
    parser.add_argument("--verify",         action="store_true",
                        help="Verifikasi dataset sudah tersedia")
    parser.add_argument("--parse_metadata", action="store_true",
                        help="Parse trial_metadata.txt menjadi manifest CSV")
    parser.add_argument("--add_manifest",   action="store_true",
                        help="Tambah ASVspoof ke split_manifest.csv sebagai external")
    parser.add_argument("--data_dir",       default="data/raw/ASVspoof2021_DF")
    parser.add_argument("--metadata_file",  default="data/raw/ASVspoof2021_DF/trial_metadata.txt")
    parser.add_argument("--audio_dir",      default="data/raw/ASVspoof2021_DF/flac")
    args = parser.parse_args()

    if args.verify:
        verify_asvspoof(args.data_dir)

    elif args.parse_metadata:
        parse_asvspoof_metadata(
            metadata_path = args.metadata_file,
            audio_dir     = args.audio_dir,
        )

    elif args.add_manifest:
        add_to_main_manifest()

    else:
        # Default: tampilkan panduan
        print(GUIDE)
        print("Perintah tersedia:")
        print("  python src/download_asvspoof.py                # tampilkan panduan ini")
        print("  python src/download_asvspoof.py --verify       # cek apakah sudah tersedia")
        print("  python src/download_asvspoof.py --parse_metadata  # buat manifest dari metadata")
        print("  python src/download_asvspoof.py --add_manifest    # gabung ke split_manifest.csv")
