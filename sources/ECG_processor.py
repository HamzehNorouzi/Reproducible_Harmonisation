import pandas as pd
import numpy as np
import glob
import os
import wfdb
from sources.actual_path import get_actual_path

# load function
def load_ecg_waveforms(df, data_dir, sampling_rate=100):
    """
    Loads raw ECG signals using memory-efficient pre-allocation.
    """
    print(f"Loading {len(df)} ECG records at {sampling_rate}Hz...")
    path_column = 'filename_lr' if sampling_rate == 100 else 'filename_hr'

    # 1. Pre-allocate the empty array [N, Channels, Time]
    N = len(df)
    X = np.empty((N, 12, 1000), dtype=np.float32)

    for i, file_path in enumerate(df[path_column]):
        # signal shape is [1000, 12]
        signal, meta = wfdb.rdsamp(os.path.join(data_dir, file_path))

        # Transpose on the fly [12, 1000] and drop directly into memory
        X[i] = signal.T

    return X

def process_ecg(data_dir, output_dir):
    """
    Main pipeline for PTB-XL. Handles loading, metadata alignment,
    and patient-safe fold splitting as per Section 5.5.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\nStarting ECG Preprocessing Pipeline (PTB-XL)...")

    data_dir = get_actual_path(data_dir, "ptbxl_database.csv")

    # 1. Load the official metadata CSV
    meta_path = os.path.join(data_dir, 'ptbxl_database.csv')
    df_meta = pd.read_csv(meta_path, index_col='ecg_id')

    # 2. Select Sampling Rate
    # Justification for report: 100Hz drastically reduces storage and RAM
    # while preserving the core QRS complex features needed for SSL.
    SR = 100

    # 3. Patient-Safe Splitting (Section 5.5)
    # Fold 10 is the standard test set, 9 for validation, 1-8 for training
    df_test = df_meta[df_meta.strat_fold == 10]
    df_val = df_meta[df_meta.strat_fold == 9]
    df_train = df_meta[df_meta.strat_fold <= 8]

    print(f" -> Splits created: {len(df_train)} Train, {len(df_val)} Val, {len(df_test)} Test")

    # 4. Load the actual waveform arrays
    X_train = load_ecg_waveforms(df_train, data_dir, SR)
    X_val = load_ecg_waveforms(df_val, data_dir, SR)
    X_test = load_ecg_waveforms(df_test, data_dir, SR)

    print(" -> Generating ECG Metadata...")
    meta_train = generate_ecg_metadata(df_train, 'train', X_train.shape, SR)
    meta_val = generate_ecg_metadata(df_val, 'validation', X_val.shape, SR)
    meta_test = generate_ecg_metadata(df_test, 'test', X_test.shape, SR)

    # Combine into one master ECG manifest
    ecg_meta_df = pd.concat([meta_train, meta_val, meta_test], ignore_index=True)

    # Save the CSV
    meta_path = os.path.join(output_dir, 'ecg_metadata.csv')
    ecg_meta_df.to_csv(meta_path, index=False)
    print(f" -> Saved ECG Metadata: {meta_path}")

    # Save the Waveform Arrays [N, C, T]
    print(" -> Saving compressed ECG arrays (this might take a moment)...")
    np.savez_compressed(os.path.join(output_dir, 'ecg_train_data.npz'), X=X_train)
    np.savez_compressed(os.path.join(output_dir, 'ecg_val_data.npz'), X=X_val)
    np.savez_compressed(os.path.join(output_dir, 'ecg_test_data.npz'), X=X_test)

    print("\n==========================================")
    print("ECG PREPROCESSING SUCCESSFUL!")
    print("==========================================")

    return X_train, X_val, X_test, ecg_meta_df


def generate_ecg_metadata(df, split_name, X_shape, sampling_rate=100):
    """
    Formats the PTB-XL metadata into the unified manifest structure.
    """
    n_samples, n_channels, n_times = X_shape

    # PTB-XL has 100Hz and 500Hz paths
    path_col = 'filename_lr' if sampling_rate == 100 else 'filename_hr'

    metadata = {
        'sample_id': [f"ECG_{split_name}_{str(i).zfill(5)}" for i in range(n_samples)],
        'dataset_name': ['PTB-XL'] * n_samples,
        'modality': ['ECG'] * n_samples,
        'subject_id': df['patient_id'].values,
        'source_file': df[path_col].values,
        'split': [split_name] * n_samples,
        'label_or_event': df['scp_codes'].values,  # Contains the diagnostic labels
        'sampling_rate_hz': [sampling_rate] * n_samples,
        'n_channels': [n_channels] * n_samples,
        'n_samples': [n_times] * n_samples,
        'channel_schema': ['12-lead (I, II, III, aVR, aVL, aVF, V1-V6)'] * n_samples,
        'qc_flags': ['PASS'] * n_samples
    }

    return pd.DataFrame(metadata)
