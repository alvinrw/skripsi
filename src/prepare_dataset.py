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

def detect_generator_source(filepath: str, label: str) -> str:
    path_clean = str(filepath).lower().replace("\\", "/").replace("_", "").replace("-", "").replace(" ", "")
    if label == "real" and not any(k in path_clean for k in ["openvoice", "f5tts", "e2tts", "voxcpm"]):
        return "Real"
    if "openvoice" in path_clean:
        return "OpenVoice"
    elif "f5tts" in path_clean:
        return "F5TTS"
    elif "e2tts" in path_clean:
        return "E2TTS"
    elif "voxcpm" in path_clean:
        return "Voxcpm"
    elif "kaggle" in path_clean:
        return "Kaggle"
    elif label == "real":
        return "Real"
    
    # Fallback to parent folder name
    parent_name = Path(filepath).parent.name
    return parent_name if parent_name else "Fake_Other"

def detect_intent(filepath: str) -> str:
    path_lower = str(filepath).lower().replace("\\", "/")
    parts = path_lower.split("/")
    for p in parts:
        p_clean = p.replace("_", "").replace("-", "").strip()
        if any(keyword in p_clean for keyword in ["test", "testing", "eval"]):
            return "testing"
        if any(keyword in p_clean for keyword in ["train", "training"]):
            return "training"
    return "training"


def process_dataset(drive_dir, out_dir, zip_out=None, kaggle_dirs=None):
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
    all_scanned_files = [str(p) for p in drive_path.rglob("*") if p.is_file() and p.suffix.lower() in audio_exts]
    
    real_files = []
    fake_files = []
    
    for f in all_scanned_files:
        path_lower = f.lower().replace("\\", "/")
        parts = path_lower.split("/")
        if any(r in parts for r in ["suara_real", "real", "bonafide"]):
            real_files.append(f)
        else:
            fake_files.append(f)
    
    if kaggle_dirs:
        for k_dir in kaggle_dirs:
            if k_dir and os.path.exists(k_dir):
                print(f"[scan] Scanning Kaggle dataset: {k_dir}")
                for p in Path(k_dir).rglob("*"):
                    if p.is_file() and p.suffix.lower() in audio_exts:
                        parts = [part.lower() for part in p.parts]
                        if "real" in parts:
                            real_files.append(str(p))
                        elif "fake" in parts:
                            fake_files.append(str(p))
                            
    print(f"[scan] Total ditemukan {len(real_files)} file real dan {len(fake_files)} file fake.")
    
    all_files = []
    for f in real_files:
        gen = detect_generator_source(f, "real")
        intent = detect_intent(f)
        all_files.append({"file_path": f, "label": "real", "label_idx": 0, "generator_source": gen, "intent": intent})
    for f in fake_files:
        gen = detect_generator_source(f, "fake")
        intent = detect_intent(f)
        all_files.append({"file_path": f, "label": "fake", "label_idx": 1, "generator_source": gen, "intent": intent})
        
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
        
    # Split into Training Pool vs Separate Testing Pool based on intent
    df_train_pool = df_valid[df_valid["intent"] == "training"].copy()
    df_test_pool = df_valid[df_valid["intent"] == "testing"].copy()
    
    # Fallback if no explicit intent folders were detected
    if len(df_train_pool) == 0:
        df_train_pool = df_valid.copy()
        df_test_pool = pd.DataFrame()
        
    # Shuffle Training Data pool
    df_train_pool = df_train_pool.sample(frac=1, random_state=2026).reset_index(drop=True)
    
    print("\n>> Tahap 2: Pembagian Dataset Training (70% Train, 15% Test, 15% Validation)...")
    unique_speakers = df_train_pool["speaker_id"].nunique()
    print(f"  Jumlah pembicara unik di training set: {unique_speakers}")
    
    if unique_speakers >= 5:
        # GroupSplit based on speaker_id
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=2026)
        train_idx, temp_idx = next(gss1.split(df_train_pool, groups=df_train_pool["speaker_id"]))
        df_train = df_train_pool.iloc[train_idx]
        df_temp = df_train_pool.iloc[temp_idx]
        
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=2026)
        val_idx, test_idx = next(gss2.split(df_temp, groups=df_temp["speaker_id"]))
        df_val = df_temp.iloc[val_idx]
        df_test = df_temp.iloc[test_idx]
    else:
        print("  [WARNING] Pembicara terlalu sedikit. Fallback ke stratified random split.")
        test_size_n = max(int(len(df_train_pool) * 0.15), 2)
        try:
            df_train, df_temp = train_test_split(df_train_pool, test_size=test_size_n*2, stratify=df_train_pool["label"], random_state=2026)
            df_val, df_test = train_test_split(df_temp, test_size=0.5, stratify=df_temp["label"], random_state=2026)
        except ValueError:
            df_train, df_temp = train_test_split(df_train_pool, test_size=0.3, random_state=2026)
            df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=2026)
            
    df_train_pool.loc[df_train.index, "split"] = "train"
    df_train_pool.loc[df_val.index, "split"] = "validation"
    df_train_pool.loc[df_test.index, "split"] = "test"
    
    if not df_test_pool.empty:
        df_test_pool["split"] = "separate_test"
        
    print("\n>> Tahap 3: Pemotongan Audio (Chunking)...")
    durations = [2]
    sr = 16000
    
    recap_data = []
    
    for dur in durations:
        print(f"  Proses pemotongan durasi fixed: {dur} detik...")
        chunk_length = int(dur * sr)
        
        dur_out_dir = out_path / f"{dur}s"
        dur_out_dir.mkdir(exist_ok=True)
        
        # 1. Manifest for Training splits (train/val/test)
        manifest_rows = []
        for _, row in tqdm(df_train_pool.iterrows(), total=len(df_train_pool), desc=f"Chunking Training Pool ({dur}s)"):
            try:
                y, _ = librosa.load(row["file_path"], sr=sr)
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
                        "generator_source": row["generator_source"],
                        "intent": row["intent"],
                        "split": row["split"]
                    })
            except Exception as e:
                print(f"  [ERROR] Gagal memotong {row['file_path']}: {e}")
                
        manifest_df = pd.DataFrame(manifest_rows)
        manifest_csv = manifests_dir / f"split_manifest_{dur}s.csv"
        if not manifest_df.empty:
            manifest_df.to_csv(manifest_csv, index=False)
            print(f"    Manifest Training Split ({dur}s) disimpan ke: {manifest_csv}")
            
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

        # 2. Manifest for Separate Test Pool
        sep_manifest_rows = []
        if not df_test_pool.empty:
            dur_sep_out_dir = out_path / f"separate_test_{dur}s"
            dur_sep_out_dir.mkdir(exist_ok=True)
            for _, row in tqdm(df_test_pool.iterrows(), total=len(df_test_pool), desc=f"Chunking Separate Test Pool ({dur}s)"):
                try:
                    y, _ = librosa.load(row["file_path"], sr=sr)
                    num_chunks = len(y) // chunk_length
                    for i in range(num_chunks):
                        chunk = y[i*chunk_length:(i+1)*chunk_length]
                        orig_stem = Path(row["file_path"]).stem
                        chunk_name = f"{orig_stem}_sepchunk{i}.wav"
                        chunk_path = dur_sep_out_dir / chunk_name
                        sf.write(str(chunk_path), chunk, sr)
                        
                        sep_manifest_rows.append({
                            "utterance_id": chunk_name.replace(".wav", ""),
                            "file_path": str(chunk_path).replace("\\", "/"),
                            "label": row["label"],
                            "label_idx": row["label_idx"],
                            "speaker_id": row["speaker_id"],
                            "generator_source": row["generator_source"],
                            "intent": row["intent"],
                            "split": "separate_test"
                        })
                except Exception as e:
                    print(f"  [ERROR] Gagal memotong testing file {row['file_path']}: {e}")
                    
            sep_manifest_df = pd.DataFrame(sep_manifest_rows)
            sep_manifest_csv = manifests_dir / f"separate_test_manifest_{dur}s.csv"
            if not sep_manifest_df.empty:
                sep_manifest_df.to_csv(sep_manifest_csv, index=False)
                print(f"    Manifest Separate Testing ({dur}s) disimpan ke: {sep_manifest_csv}")
                recap_data.append({
                    "duration": f"{dur}s",
                    "split": "separate_test",
                    "real_chunks": len(sep_manifest_df[sep_manifest_df["label"] == "real"]),
                    "fake_chunks": len(sep_manifest_df[sep_manifest_df["label"] == "fake"]),
                    "total_chunks": len(sep_manifest_df)
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
    parser.add_argument("--kaggle_dirs", type=str, nargs="*", default=None, help="Daftar direktori Kaggle dataset")
    args = parser.parse_args()
    process_dataset(args.drive_dir, args.out_dir, args.zip_out, args.kaggle_dirs)
