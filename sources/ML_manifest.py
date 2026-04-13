import os
import numpy as np
import pandas as pd
from datetime import datetime
def generate_modality_manifest(processed_dir='data/processed'):
    """
    Generates an organized manifest, prints a summary, and saves a
    comprehensive machine-readable CSV as per Section 6. [cite: 94, 95]
    """
    if not os.path.exists(processed_dir):
        print(f"Error: Folder {processed_dir} not found.")
        return

    all_files = sorted(os.listdir(processed_dir))

    # Define groups based on filenames
    modalities = {
        'HAR': [f for f in all_files if any(x in f.lower() for x in ['har', 'pretrain', 'supervised'])],
        'ECG': [f for f in all_files if 'ecg' in f.lower()],
        'EEG': [f for f in all_files if 'eeg' in f.lower()]
    }

    full_manifest_list = []

    print("\nFINAL DATASET MANIFEST SUMMARY")
    print("==============================")

    for modality, files in modalities.items():
        if not files: continue

        print(f"\n{modality}")
        print("-" * len(modality))

        modality_rows = []
        for file_name in files:
            # Skip the manifest itself to avoid recursion
            if 'pipeline_manifest.csv' in file_name:
                continue

            path = os.path.join(processed_dir, file_name)
            size_mb = os.path.getsize(path) / (1024 * 1024)

            info = {
                'modality': modality,
                'file_name': file_name,
                'file_size_mb': round(size_mb, 2),
            }

            if file_name.endswith('.npz'):
                data = np.load(path)
                shape = data['X'].shape
                info['row_or_window_count'] = shape[0]
                info['shape_n_c_t'] = str(shape)
            elif file_name.endswith('.csv'):
                df = pd.read_csv(path)
                info['row_or_window_count'] = len(df)
                info['shape_n_c_t'] = 'Metadata CSV'
            else:
                # Skip any other random files like .DS_Store or logs
                continue

            modality_rows.append(info)
            full_manifest_list.append(info)

        if modality_rows:
            summary_df = pd.DataFrame(modality_rows)
            print(
                summary_df[['file_name', 'file_size_mb', 'row_or_window_count', 'shape_n_c_t']].to_string(index=False))

    # Save the Machine-Readable CSV [cite: 95]
    if full_manifest_list:
        final_manifest_df = pd.DataFrame(full_manifest_list)
        manifest_out_path = os.path.join(processed_dir, 'pipeline_manifest.csv')
        final_manifest_df.to_csv(manifest_out_path, index=False)
        print(f"\n[SUCCESS] Machine-readable manifest saved to: {manifest_out_path}")

