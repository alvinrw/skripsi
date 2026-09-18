"""
src/evaluate_results.py
=======================
Generate Confusion Matrix plot and accuracy breakdown by source generator.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def generate_report(scores_csv: str, manifest_csv: str, out_dir: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(scores_csv) or not os.path.exists(manifest_csv):
        print(f"[evaluate] File tidak ditemukan: {scores_csv} atau {manifest_csv}")
        return False
        
    df_scores = pd.read_csv(scores_csv)
    df_manifest = pd.read_csv(manifest_csv, sep=None, engine='python')
    
    df_test = df_scores[df_scores['split'] == 'test'].copy()
    df_merged = pd.merge(df_test, df_manifest[['utterance_id', 'dataset', 'generator_id']], on='utterance_id', how='left')
    
    # Cari semua kolom skor model
    score_cols = [c for c in df_merged.columns if c.startswith('score_')]
    
    if not score_cols:
        print("[evaluate] Tidak ada kolom score_ di utterance_scores.csv")
        return False
        
    print(f"\n[evaluate] Men-generate Confusion Matrix & Laporan untuk model: {', '.join(score_cols)}")
    
    summary_rows = []
    
    for model_col in score_cols:
        model_name = model_col.replace('score_', '')
        # Threshold default 0.5, atau harusnya baca dari metrics.csv
        df_merged['prediksi'] = (df_merged[model_col] > 0.5).astype(int)
        
        # 1. Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(df_merged['label'], df_merged['prediksi'])
        
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Asli (0)', 'Fake (1)'], 
                    yticklabels=['Benar Asli (0)', 'Benar Fake (1)'])
        plt.title(f'Confusion Matrix - Model {model_name}')
        plt.ylabel('Kebenaran (Ground Truth)')
        plt.xlabel('Tebakan AI (Prediction)')
        plt.tight_layout()
        cm_path = out_path / f"confusion_matrix_{model_name}.png"
        plt.savefig(str(cm_path))
        plt.close()
        
        # 2. Analisis Sumber Deteksi Deepfake
        df_fake = df_merged[df_merged['label'] == 1].copy()
        if len(df_fake) > 0:
            df_fake['tebakan_benar'] = (df_fake['prediksi'] == 1)
            akurasi_sumber = df_fake.groupby('dataset')['tebakan_benar'].mean() * 100
            for dataset, acc in akurasi_sumber.items():
                summary_rows.append({
                    'model': model_name,
                    'dataset_sumber': dataset,
                    'akurasi_deteksi_deepfake_persen': round(acc, 2)
                })
                
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        summary_path = out_path / "accuracy_by_source.csv"
        df_summary.to_csv(str(summary_path), index=False)
        print(f"  [OK] Tersimpan: {len(score_cols)} Gambar Confusion Matrix")
        print(f"  [OK] Tersimpan: Analisis akurasi per sumber di {summary_path.name}")
        
    return True
