import os
import sys
import glob
import librosa
import soundfile as sf
import pandas as pd
import numpy as np
import argparse
import shutil
import warnings
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit, train_test_split

warnings.filterwarnings("ignore")

def check_eligibility(filepath, sr=16000, min_duration=1.0):
    try:
        y, _ = librosa.load(filepath, sr=sr)
        if len(y) == 0:
            return False, "Empty audio"
        if np.isnan(y).any():
            return False, "NaN values detected"
        
        # Check non-silent duration
        non_mute_intervals = librosa.effects.split(y, top_db=30)
        valid_samples = sum([end - start for start, end in non_mute_intervals])
        valid_duration = valid_samples / sr
        
        if valid_duration < min_duration:
            return False, f"Too short non-silent duration ({valid_duration:.2f}s)"
            
        rms = librosa.feature.rms(y=y).mean()
        if rms < 0.001:
            return False, "Too quiet (RMS too low)"
            
        return True, "Eligible"
    except Exception as e:
        return False, f"Error: {e}"

def extract_speaker_id(filepath):
    """
    Ekstrak speaker ID dari nama file.
    Mendukung format VoxCeleb (id10001-real.wav), nama bebas, maupun
    pesan WhatsApp (WhatsApp Ptt 2026-07-28 ...) yang semuanya akan
    diperlakukan sebagai satu speaker berbeda per file.
    """
    stem = Path(filepath).stem.lower()

    # Format WhatsApp Ptt: gunakan nama file lengkap sebagai speaker unik
    if stem.startswith("whatsapp ptt") or stem.startswith("whatsapp"):
        # Pakai seluruh nama file agar setiap pesan suara = 1 speaker unik
        return stem

    # Format VoxCeleb: id10001-real, id10002-spoof, dst.
    if stem.startswith("id") and "-" in stem:
        return stem.split("-")[0]

    # Format dengan underscore: alvin_part1 -> alvin
    if "_" in stem:
        return stem.split("_")[0]

    # Default: nama file penuh = 1 speaker unik
    return stem

def process_dataset(drive_dir, out_dir, zip_out=None):
    print("\n[mount] Bukan lingkungan Google Colab. Mount dilewati.")
    print("[reproducibility] Seed set to 2026")
    np.random.seed(2026)
    
    drive_path = Path(drive_dir)
    out_path = Path(out_dir)
    manifests_dir = Path("manifests")
    results_dir = Path("results")
    
    out_path.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    real_files = [str(p) for p in (drive_path / "Suara_real").rglob("*") if p.is_file() and p.suffix.lower() in audio_exts]
    fake_files = [str(p) for p in (drive_path / "output_generate").rglob("*") if p.is_file() and p.suffix.lower() in audio_exts]
    
    print(f"[scan] Ditemukan {len(real_files)} file real dan {len(fake_files)} file fake.")
    
    all_files = []
    for f in real_files:
        all_files.append({"file_path": f, "label": "real", "label_idx": 0})
    for f in fake_files:
        all_files.append({"file_path": f, "label": "fake", "label_idx": 1})
        
    df = pd.DataFrame(all_files)
    if len(df) == 0:
        print("[FAIL] Tidak ada file audio ditemukan.")
        sys.exit(1)
        
    print("\n>> Tahap 1: Menjalankan Eligibility Check...")
    eligibility = []
    for f in tqdm(df["file_path"], desc="Checking eligibility"):
        is_ok, reason = check_eligibility(f)
        eligibility.append((is_ok, reason))
        
    df["eligible"] = [e[0] for e in eligibility]
    df["reason"] = [e[1] for e in eligibility]
    df["speaker_id"] = df["file_path"].apply(lambda x: extract_speaker_id(Path(x).stem))
    
    df.to_csv(results_dir / "dataset_recap_raw.csv", index=False)
    
    eligible_count = df["eligible"].sum()
    print(f"  [Hasil QC] Eligible: {eligible_count} | Non-Eligible (diabaikan): {len(df) - eligible_count}")
    print(f"  Laporan QC mentah disimpan ke: {results_dir / 'dataset_recap_raw.csv'}")
    
    df_valid = df[df["eligible"]].copy()
    if len(df_valid) == 0:
        print("[FAIL] Tidak ada file audio yang valid.")
        sys.exit(1)
        
    print("\n>> Tahap 2: Pembagian Dataset (70% Train, 15% Test, 15% Validation)...")
    unique_speakers = df_valid["speaker_id"].nunique()
    print(f"  Jumlah pembicara unik terdeteksi: {unique_speakers}")
    
    if unique_speakers >= 5:
        # GroupSplit
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=2026)
        train_idx, temp_idx = next(gss1.split(df_valid, groups=df_valid["speaker_id"]))
        df_train = df_valid.iloc[train_idx]
        df_temp = df_valid.iloc[temp_idx]
        
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=2026)
        val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp["speaker_id"]))
        df_val = df_temp.iloc[val_idx]
        df_test = df_temp.iloc[test_idx]
    else:
        print("  [WARNING] Pembicara terlalu sedikit. Fallback ke stratified random split.")
        test_size_n = int(len(df_valid) * 0.15)
        if test_size_n < 2:
            test_size_n = 2  # At least 2 for stratification
        
        # If dataset is really small (like the dummy one), avoid stratify failing
        try:
            df_train, df_temp = train_test_split(df_valid, test_size=test_size_n*2, stratify=df_valid["label"], random_state=2026)
            df_val, df_test = train_test_split(df_temp, test_size=0.5, stratify=df_temp["label"], random_state=2026)
        except ValueError:
            # Fallback again if stratify fails
            df_train, df_temp = train_test_split(df_valid, test_size=0.3, random_state=2026)
            df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=2026)
            
    df_valid.loc[df_train.index, "split"] = "train"
    df_valid.loc[df_val.index, "split"] = "validation"
    df_valid.loc[df_test.index, "split"] = "test"
    
    print("\n>> Tahap 3: Pemotongan Audio (Chunking)...")
    durations = [2, 3, 5, 7]
    sr = 16000
    
    recap_data = []
    
    for dur in durations:
        print(f"  Proses pemotongan durasi fixed: {dur} detik...")
        chunk_length = int(dur * sr)
        manifest_rows = []
        
        dur_out_dir = out_path / f"{dur}s"
        dur_out_dir.mkdir(exist_ok=True)
        
        for _, row in tqdm(df_valid.iterrows(), total=len(df_valid), desc=f"Chunking {dur}s"):
            y, _ = librosa.load(row["file_path"], sr=sr)
            
            # Non-overlapping chunking
            num_chunks = len(y) // chunk_length
            for i in range(num_chunks):
                chunk = y[i*chunk_length:(i+1)*chunk_length]
                orig_stem = Path(row["file_path"]).stem
                chunk_name = f"{orig_stem}_chunk{i}.wav"
                chunk_path = dur_out_dir / chunk_name
                
                sf.write(str(chunk_path), chunk, sr)
                
                manifest_rows.append({
                    "utterance_id": chunk_name.replace(".wav", ""),
                    "file_path": str(chunk_path).replace("\\", "/"),
                    "label": row["label"],
                    "label_idx": row["label_idx"],
                    "speaker_id": row["speaker_id"],
                    "split": row["split"]
                })
                
        manifest_df = pd.DataFrame(manifest_rows)
        manifest_csv = manifests_dir / f"split_manifest_{dur}s.csv"
        if not manifest_df.empty:
            manifest_df.to_csv(manifest_csv, index=False)
            print(f"    Manifest untuk durasi {dur}s disimpan ke: {manifest_csv}")
            
            total_dur_chunks = len(manifest_df)
            real_dur_chunks = len(manifest_df[manifest_df["label"] == "real"])
            fake_dur_chunks = len(manifest_df[manifest_df["label"] == "fake"])
            print(f"    -> Hasil potongan {dur}s: {total_dur_chunks} data ({real_dur_chunks} suara asli, {fake_dur_chunks} deepfake)")
            
            # Record recap
            for split in ["train", "validation", "test"]:
                subset = manifest_df[manifest_df["split"] == split]
                real_c = len(subset[subset["label"] == "real"])
                fake_c = len(subset[subset["label"] == "fake"])
                recap_data.append({
                    "duration": f"{dur}s",
                    "split": split,
                    "real_chunks": real_c,
                    "fake_chunks": fake_c,
                    "total_chunks": len(subset)
                })
        else:
            print(f"    [WARNING] Tidak ada chunk yang dihasilkan untuk durasi {dur}s.")
            recap_data.append({
                "duration": f"{dur}s",
                "split": "train",
                "real_chunks": 0, "fake_chunks": 0, "total_chunks": 0
            })
            recap_data.append({
                "duration": f"{dur}s",
                "split": "validation",
                "real_chunks": 0, "fake_chunks": 0, "total_chunks": 0
            })
            recap_data.append({
                "duration": f"{dur}s",
                "split": "test",
                "real_chunks": 0, "fake_chunks": 0, "total_chunks": 0
            })
            
    if zip_out:
        print(f"\n>> Tahap 4: Mengompresi dataset ke {zip_out}...")
        shutil.make_archive(zip_out.replace(".zip", ""), 'zip', str(out_path))
        print(f"  Berhasil membuat zip: {zip_out}")
        
    print("\n============================================================")
    print("Ringkasan Hasil Pemotongan & Pembagian Dataset:")
    print("============================================================")
    recap_df = pd.DataFrame(recap_data)
    print(recap_df.to_string(index=False))
    print("============================================================")
    
    recap_csv = results_dir / "dataset_recap_processed.csv"
    recap_df.to_csv(recap_csv, index=False)
    print(f"Laporan rekapitulasi diproses disimpan ke: {recap_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive_dir", type=str, default="data/raw")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--zip_out", type=str, default=None)
    args = parser.parse_args()
    process_dataset(args.drive_dir, args.out_dir, args.zip_out)
