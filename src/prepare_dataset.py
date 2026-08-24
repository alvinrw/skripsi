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

def extract_speaker_id(filename):
    filename = filename.lower()
    # Handle id10001-real.wav etc.
    if '-' in filename:
        return filename.split('-')[0]
    if '_' in filename:
        return filename.split('_')[0]
    return filename

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
        
    print("\n>> Tahap 2: Pemotongan Audio (Chunking) & Pengacakan (Split)...")
    print("  (Memotong audio dan membagi berdasarkan chunk secara acak untuk meratakan data)")
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
                    "split": "unassigned"
                })
                
        manifest_df = pd.DataFrame(manifest_rows)
        
        # Split chunk-level (Randomized)
        if not manifest_df.empty:
            try:
                train_df, temp_df = train_test_split(manifest_df, test_size=0.3, stratify=manifest_df["label"], random_state=2026)
                val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=2026)
            except ValueError:
                train_df, temp_df = train_test_split(manifest_df, test_size=0.3, random_state=2026)
                val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=2026)
                
            manifest_df.loc[train_df.index, "split"] = "train"
            manifest_df.loc[val_df.index, "split"] = "validation"
            manifest_df.loc[test_df.index, "split"] = "test"
            
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
