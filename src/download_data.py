"""
src/download_data.py
====================
Script download dataset DEEP-VOICE dari Kaggle dan (opsional)
persiapan struktur folder untuk pipeline.

PRASYARAT:
    1. Install Kaggle CLI:
       pip install kaggle

    2. Dapatkan API key dari https://www.kaggle.com/settings
       -> Account -> API -> Create New Token -> unduh kaggle.json

    3. Taruh kaggle.json di:
       Windows : C:\\Users\\<nama_anda>\\.kaggle\\kaggle.json
       Linux   : ~/.kaggle/kaggle.json

Cara pakai:
    python src/download_data.py
    python src/download_data.py --skip_kaggle    # bila sudah download manual
"""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path


# ──────────────────────────────────────────────
# Konfigurasi dataset
# ──────────────────────────────────────────────

KAGGLE_DATASET  = "birdy654/deep-voice-deepfake-voice-recognition"
DATA_RAW_DIR    = Path("data/raw")
DATA_ZIP_NAME   = "deep-voice-deepfake-voice-recognition.zip"


# ──────────────────────────────────────────────
# Cek Kaggle API key
# ──────────────────────────────────────────────

def check_kaggle_credentials() -> bool:
    """Periksa apakah kaggle.json sudah tersedia."""
    kaggle_path = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_path.exists():
        print(f"[download] Kaggle credentials ditemukan: {kaggle_path}")
        return True
    else:
        print(f"""
[ERROR] Kaggle API key tidak ditemukan di: {kaggle_path}

Langkah setup:
  1. Buka https://www.kaggle.com/settings
  2. Scroll ke bagian 'API'
  3. Klik 'Create New Token' -> file kaggle.json akan diunduh
  4. Pindahkan file kaggle.json ke:
       Windows: C:\\Users\\{os.environ.get('USERNAME', '<nama_anda>')}\\.kaggle\\kaggle.json
  5. Jalankan script ini lagi
""")
        return False


# ──────────────────────────────────────────────
# Download via Kaggle API
# ──────────────────────────────────────────────

def download_kaggle(dest_dir: Path) -> Path:
    """
    Download dataset dari Kaggle menggunakan kaggle-python library.
    Returns path ke file ZIP yang diunduh.
    """
    try:
        import kaggle  # noqa: F401  (trigger auth check)
    except ImportError:
        raise ImportError(
            "Library 'kaggle' belum terinstall.\n"
            "Jalankan: pip install kaggle"
        )

    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download] Mengunduh dataset: {KAGGLE_DATASET}")
    print(f"[download] Tujuan: {dest_dir}")
    print("  Ini mungkin memakan waktu beberapa menit (ukuran ~1-2 GB)...")

    # Kaggle CLI download
    os.system(
        f'kaggle datasets download -d {KAGGLE_DATASET} -p "{dest_dir}" --unzip'
    )

    print(f"[download] Download selesai -> {dest_dir}")
    return dest_dir


# ──────────────────────────────────────────────
# Verifikasi struktur folder hasil extract
# ──────────────────────────────────────────────

def verify_structure(data_dir: Path) -> dict:
    """
    Periksa struktur folder hasil extract dan hitung file audio.
    Returns ringkasan jumlah file per kategori.
    """
    audio_ext = {".wav", ".flac", ".mp3", ".ogg"}
    summary = {}

    print(f"\n[verify] Memeriksa struktur di: {data_dir}")

    # Cari semua folder dan hitung file audio
    if not data_dir.exists():
        print(f"  [ERROR] Folder tidak ditemukan: {data_dir}")
        return {}

    all_audio = list(data_dir.rglob("*"))
    audio_files = [f for f in all_audio if f.is_file() and f.suffix.lower() in audio_ext]

    print(f"  Total file audio ditemukan: {len(audio_files)}")

    # Coba deteksi label dari struktur folder
    real_files = [f for f in audio_files if any(
        p.lower() in ["real", "bonafide"] for p in f.parts
    )]
    fake_files = [f for f in audio_files if any(
        p.lower() in ["fake", "deepfake", "spoof", "synthetic"] for p in f.parts
    )]
    unknown    = len(audio_files) - len(real_files) - len(fake_files)

    summary = {
        "total":   len(audio_files),
        "real":    len(real_files),
        "fake":    len(fake_files),
        "unknown": unknown,
    }

    print(f"  Real/bonafide : {summary['real']}")
    print(f"  Fake/deepfake : {summary['fake']}")
    if unknown > 0:
        print(f"  Unknown label : {unknown}  ← perlu label_map.csv manual")

    # Tampilkan 5 folder top-level
    top_dirs = sorted({f.parent for f in audio_files})[:5]
    print(f"\n  Contoh folder (5 pertama):")
    for d in top_dirs:
        print(f"    {d}")

    return summary


# ──────────────────────────────────────────────
# Panduan manual (bila tidak pakai Kaggle API)
# ──────────────────────────────────────────────

def print_manual_guide() -> None:
    print("""
============================================================
PANDUAN DOWNLOAD MANUAL (tanpa Kaggle API)
============================================================
1. Buka browser, login ke Kaggle:
   https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition

2. Klik tombol "Download" (pojok kanan atas)
   File: deep-voice-deepfake-voice-recognition.zip (~1-2 GB)

3. Setelah download selesai, ekstrak ZIP ke folder:
   c:\\Users\\alvin\\Downloads\\test\\skripsi_residual_modulasi\\data\\raw\\

4. Pastikan struktur seperti ini:
   data/raw/
   ├── real/
   │   └── <file-file .wav>
   └── fake/
       └── <file-file .wav>

5. Lanjutkan dengan:
   python src/run_pipeline.py --steps manifest splits leakage
============================================================
""")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download dataset DEEP-VOICE dari Kaggle")
    parser.add_argument(
        "--skip_kaggle", action="store_true",
        help="Lewati download Kaggle (bila sudah download manual). Hanya verifikasi struktur."
    )
    parser.add_argument(
        "--data_dir", default=str(DATA_RAW_DIR),
        help="Folder tujuan data (default: data/raw)"
    )
    parser.add_argument(
        "--manual_guide", action="store_true",
        help="Tampilkan panduan download manual"
    )
    args = parser.parse_args()

    if args.manual_guide:
        print_manual_guide()
        return

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_kaggle:
        print("[download] Mode skip: langsung verifikasi struktur...")
        summary = verify_structure(data_dir)
    else:
        # Cek kredensial dulu
        if not check_kaggle_credentials():
            print("\nAlternatif: jalankan dengan --manual_guide untuk panduan download manual")
            print("Atau: python src/download_data.py --manual_guide")
            return

        # Download
        try:
            download_kaggle(data_dir)
            summary = verify_structure(data_dir)
        except Exception as e:
            print(f"\n[ERROR] Download gagal: {e}")
            print("\nCoba download manual:")
            print_manual_guide()
            return

    # Saran langkah selanjutnya
    if summary.get("total", 0) > 0:
        print("""
============================================================
Dataset sudah siap! Langkah selanjutnya:

  python src/run_pipeline.py --steps manifest
  python src/run_pipeline.py --steps splits
  python src/run_pipeline.py --steps leakage
  python src/run_pipeline.py --steps features
  python src/run_pipeline.py --steps train
============================================================
""")
    else:
        print("\n[WARNING] Tidak ada file audio ditemukan. Periksa folder data/raw/")
        print_manual_guide()


if __name__ == "__main__":
    main()
