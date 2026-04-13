import os
import numpy as np
import pandas as pd
import glob
from scipy import signal
from config.settings import (TARGET_SR, CHANNELS, LABEL_MAP_PAMAP2, LABEL_MAP_WISDM,
                             LABEL_MAP_MHEALTH, PRETRAIN_WINDOW_SEC, SUPERVISED_WINDOW_SEC,
                             SUPERVISED_OVERLAP, PRETRAIN_OVERLAP)
from sources.actual_path import get_actual_path

def load_pamap2(data_dir_pamap):
    """
    Loads PAMAP2 dat files, extracts only the wrist IMU columns, 
    and returns a combined DataFrame. Uses chunking to save RAM.
    """
    print("Loading PAMAP2...")
    CHANNELS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    all_data = []
    protocol_dir = os.path.join(data_dir_pamap, 'Protocol')

    # List all .dat files for the subjects
    file_list = glob.glob(os.path.join(protocol_dir, '*.dat'))

    for file_path in file_list:
        subject_id = os.path.basename(file_path).split('.')[0]

        # PAMAP2 has 54 columns. only these are needed:
        # Col 0: timestamp, Col 1: activity_id
        # Col 4, 5, 6: hand/wrist Accel (X, Y, Z)
        # Col 10, 11, 12: hand/wrist Gyro (X, Y, Z)
        usecols = [0, 1, 4, 5, 6, 10, 11, 12]

        # Read in chunks to prevent memory overload
        chunk_iter = pd.read_csv(file_path, sep='\\s+', header=None, usecols=usecols, chunksize=50000)
        for chunk in chunk_iter:
            chunk.columns = ['timestamp', 'native_label', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
            chunk['subject_id'] = subject_id
            chunk['dataset'] = 'PAMAP2'
            all_data.append(chunk)

    df = pd.concat(all_data, ignore_index=True)

    # Drop rows where IMU data is NaN (sensor dropped out)
    df = df.dropna(subset=CHANNELS)
    return df

# WISDM
def load_wisdm(data_dir_wisdm):
    """
    Loads WISDM watch accelerometer and gyroscope data.
    """
    print("Loading WISDM (Watch Sensors Only)...")

    # Paths to the specific watch folders
    watch_accel_dir = os.path.join(data_dir_wisdm, 'accel')
    watch_gyro_dir = os.path.join(data_dir_wisdm, 'gyro')

    def parse_wisdm_files(directory, sensor_type):
        all_data = []
        file_list = glob.glob(os.path.join(directory, '*.txt'))

        for file_path in file_list:
            # WISDM files have a trailing semicolon that breaks standard CSV parsing
            df = pd.read_csv(file_path, header=None, names=['subject_id', 'native_label', 'timestamp', 'x', 'y', 'z'])

            # Clean the trailing semicolon from the 'z' column and convert to float
            df['z'] = df['z'].astype(str).str.replace(';', '').astype(float)

            # Rename columns based on sensor type
            df = df.rename(columns={
                'x': f'{sensor_type}_x',
                'y': f'{sensor_type}_y',
                'z': f'{sensor_type}_z'
            })
            all_data.append(df)

        return pd.concat(all_data, ignore_index=True)

    # 1. Load the raw files
    print(" -> Parsing watch accelerometer files...")
    df_accel = parse_wisdm_files(watch_accel_dir, 'acc')

    print(" -> Parsing watch gyroscope files...")
    df_gyro = parse_wisdm_files(watch_gyro_dir, 'gyro')

    # 2. Merge the sensors
    print(" -> Aligning sensors by timestamp...")
    # Sort by timestamp before merging (required for merge_asof)
    df_accel = df_accel.sort_values('timestamp')
    df_gyro = df_gyro.sort_values('timestamp')

    # I merge accelerometer and gyroscope based on the closest timestamp
    # within a tight tolerance (e.g., 50ms, since 20Hz = 50ms per tick)
    # I group by subject and label so we don't accidentally merge different activities
    merged_df = pd.merge_asof(
        df_accel, df_gyro,
        on='timestamp',
        by=['subject_id', 'native_label'],
        direction='nearest',
        tolerance=50000000  # WISDM timestamps are in nanoseconds. 50ms = 50,000,000 ns
    )

    # Drop rows where there is no matching timestamp for both sensors
    merged_df = merged_df.dropna()
    merged_df['dataset'] = 'WISDM'
    return merged_df


# mhealth
def load_mhealth(data_dir_mhealth):
    """
    Loads mHealth log files, extracts only the right lower arm IMU,
    and returns a combined DataFrame.
    """
    print("Loading mHealth...")
    all_data = []
    CHANNELS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    # Path to the logs.
    file_list = glob.glob(os.path.join(data_dir_mhealth, 'MHEALTHDATASET', '*.log'))

    if not file_list:
        print(" -> ERROR: No .log files found. Check your folder path!")
        return None

    for file_path in file_list:
        # Extract the subject ID from the filename (e.g., 'mHealth_subject1.log' -> 'subject1')
        filename = os.path.basename(file_path)
        subject_id = filename.split('.')[0].split('_')[1]

        # only the arm sensors and the label are needed to match the 6-channel schema
        usecols = [14, 15, 16, 17, 18, 19, 23]

        # Read the space-separated file
        df = pd.read_csv(file_path, sep='\\s+', header=None, usecols=usecols)
        df.columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'native_label']

        df['subject_id'] = subject_id
        df['dataset'] = 'mHealth'

        all_data.append(df)

    merged_df = pd.concat(all_data, ignore_index=True)

    # Drop any rows where sensor data completely dropped out
    merged_df = merged_df.dropna(subset=CHANNELS)

    return merged_df


# HARMONISATION & ORCHESTRATION

def resample_dataset(df, original_sr, target_sr):
    """
    Resamples sensor data using Fourier method and aligns categorical labels
    using nearest-neighbor index mapping.
    """
    if original_sr == target_sr:
        print(f" -> Natively at {target_sr}Hz. Skipping resampling.")
        return df

    print(f" -> Resampling from {original_sr}Hz to {target_sr}Hz...")

    num_rows = int(len(df) * (target_sr / original_sr))

    # 1. Resample the continuous sensor data (The 6 CHANNELS)
    sensor_data = df[CHANNELS].values
    resampled_sensors = signal.resample(sensor_data, num_rows)
    df_resampled = pd.DataFrame(resampled_sensors, columns=CHANNELS)

    # 2. Sub-sample the categorical/metadata (Labels, Subject IDs, Dataset name)
    # from the original dataframe to pick the nearest labels.
    original_indices = np.linspace(0, len(df) - 1, num=num_rows)
    nearest_indices = np.round(original_indices).astype(int)

    # Safely map the metadata over to the new shrunken dataframe
    df_resampled['native_label'] = df['native_label'].iloc[nearest_indices].values
    df_resampled['subject_id'] = df['subject_id'].iloc[nearest_indices].values
    df_resampled['dataset'] = df['dataset'].iloc[nearest_indices].values

    return df_resampled

# Slicing data into fixed-size window
def create_windows(df, window_sec, overlap_ratio, sr=TARGET_SR):
    """
    Slices the continuous sensor data into fixed-size windows.
    Prevents cross-contamination by windowing per subject.
    """
    print(f" -> Creating {window_sec}s windows with {overlap_ratio * 100}% overlap...")

    # Calculate sizes based on the sampling rate (20 Hz)
    window_size = int(window_sec * sr)
    step_size = int(window_size * (1 - overlap_ratio))

    windows = []
    labels = []
    subject_ids = []
    dataset_names = []

    # Group by subject to prevent cross-contamination (Leakage Control)
    for subject, group in df.groupby('subject_id'):
        sensor_data = group[CHANNELS].values
        activity_labels = group['unified_label'].values

        d_name = group['dataset'].iloc[0]

        # Slide the window across this subject's data
        for start_idx in range(0, len(group) - window_size + 1, step_size):
            end_idx = start_idx + window_size

            # Extract the 6-channel sensor chunk
            window_chunk = sensor_data[start_idx:end_idx]

            # For the label, I take the most frequent activity occurring inside this specific window
            from scipy import stats
            window_label = stats.mode(activity_labels[start_idx:end_idx], keepdims=False)[0]

            windows.append(window_chunk)
            labels.append(window_label)
            subject_ids.append(subject)
            dataset_names.append(d_name)

    # Convert lists to efficient float32/int32 NumPy arrays
    x = np.array(windows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    groups = np.array(subject_ids)
    datasets = np.array(dataset_names)
    return x, y, groups, datasets

# Mapping activities into the shared 5-class schema
def apply_unified_schema(df):
    """
    Translates dataset-specific labels into the shared 5-class schema.
    Drops any rows that do not map to the shared schema (e.g., transient or unique activities).
    { 1 = Sitting
     2 = Standing
     3 = Walking
     4 = Running / Jogging (Merged)
     5 = Stairs (Merged)
     }
    """
    print(" -> Mapping labels to unified 5-class schema...")

    # Create an empty column for the new labels
    df['unified_label'] = np.nan

    # Apply PAMAP2 Mapping
    mask_pamap = df['dataset'] == 'PAMAP2'
    df.loc[mask_pamap, 'unified_label'] = df.loc[mask_pamap, 'native_label'].map(LABEL_MAP_PAMAP2)

    # Apply WISDM Mapping
    mask_wisdm = df['dataset'] == 'WISDM'
    df.loc[mask_wisdm, 'unified_label'] = df.loc[mask_wisdm, 'native_label'].map(LABEL_MAP_WISDM)

    # Apply mHealth Mapping
    mask_mhealth = df['dataset'] == 'mHealth'
    df.loc[mask_mhealth, 'unified_label'] = df.loc[mask_mhealth, 'native_label'].map(LABEL_MAP_MHEALTH)

    # Drop rows where the label couldn't be mapped (transient/unshared activities)
    initial_rows = len(df)
    df_clean = df.dropna(subset=['unified_label']).copy()
    final_rows = len(df_clean)

    # Convert label to integer now that NaNs are gone
    df_clean['unified_label'] = df_clean['unified_label'].astype(int)

    print(f" -> Cleaned Data: Dropped {initial_rows - final_rows} rows of unshared/transient activities.")
    return df_clean

# Saving outputs
def save_processed_data(X_pretrain, X_supervised, y_supervised, output_dir='data/processed'):
    """
    Saves the final windowed arrays to disk using compressed NumPy format.
    """
    # Create the processed directory if it doesn't exist yet
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- Saving Processed Data to {output_dir} ---")

    # Save Pretraining Data
    pretrain_file = os.path.join(output_dir, 'HAR_pretrain_data.npz')
    np.savez_compressed(pretrain_file, X=X_pretrain)
    print(f" -> Saved: {pretrain_file}")

    # Save Supervised Data
    supervised_file = os.path.join(output_dir, 'HAR_supervised_data.npz')
    np.savez_compressed(supervised_file, X=X_supervised, y=y_supervised)
    print(f" -> Saved: {supervised_file}")

# Metadata generator
def generate_metadata_csv(X, y, groups, datasets, split_name, output_dir='data/processed'):
    """
    Generates the required metadata CSV for the processed HAR windows.
    """
    print(f" -> Generating metadata for {split_name} data...")

    n_samples, window_length, n_channels = X.shape

    # Create the exact columns requested by the brief
    metadata = {
        'sample_id': [f"HAR_{split_name.upper() }_{str(i).zfill(5)}" for i in range(n_samples)],
        'dataset_name': datasets,
        'modality': ['HAR'] * n_samples,
        'subject_id': groups,
        'source_file': ['Windowed_From_Raw'] * n_samples,
        'split': [split_name] * n_samples,
        'label': y,
        'n_channels': [n_channels] * n_samples,
        'n_samples': [window_length] * n_samples,
        'channel_schema': ['acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z'] * n_samples,
        'qc_flags': ['PASS'] * n_samples
    }

    df_meta = pd.DataFrame(metadata)

    # If this is pretraining, labels aren't strictly meant to be used,
    # but keeping them in the CSV as a reference is fine.
    if split_name == 'pretrain':
        df_meta['label_or_event'] = np.nan

    return df_meta

# Apply preprocessing steps
def preprocess_har():
    print("Starting HAR Harmonisation Pipeline...\n")

    # 1. Load and Resample PAMAP2 (Looking for the 'Protocol' folder)
    pamap2_root = get_actual_path("data/raw/PAMAP2", "Protocol")
    df_pamap2 = load_pamap2(pamap2_root)
    df_pamap2 = resample_dataset(df_pamap2, original_sr=100, target_sr=TARGET_SR)

    # 2. Load and Resample mHealth (Looking for the 'MHEALTHDATASET' folder)
    mhealth_root = get_actual_path("data/raw/mHealth", "MHEALTHDATASET")
    df_mhealth = load_mhealth(mhealth_root)
    if df_mhealth is not None:
        df_mhealth = resample_dataset(df_mhealth, original_sr=50, target_sr=TARGET_SR)

    # 3. Load and Resample WISDM (Looking for the 'accel' folder inside the watch folder)
    wisdm_watch_dir = get_actual_path("data/raw/WISDM", "accel")
    df_wisdm = load_wisdm(wisdm_watch_dir)
    df_wisdm = resample_dataset(df_wisdm, original_sr=20, target_sr=TARGET_SR)

    # 4. Concatenate into one master dataframe
    dataframes_to_merge = [df for df in [df_pamap2, df_mhealth, df_wisdm] if df is not None]
    master_har_df = pd.concat(dataframes_to_merge, ignore_index=True)

    # 5. Clean and Map the Labels
    clean_har_df = apply_unified_schema(master_har_df)

    # 6. Create Pretraining Windows
    print("\n--- Generating Pretraining Data ---")
    X_pretrain, y_pretrain, groups_pretrain, ds_pretrain = create_windows(
        clean_har_df,
        window_sec=PRETRAIN_WINDOW_SEC,
        overlap_ratio=PRETRAIN_OVERLAP
    )

    # 7. Create Supervised Windows
    print("\n--- Generating Supervised Data ---")
    X_supervised, y_supervised, groups_supervised, ds_supervised = create_windows(
        clean_har_df,
        window_sec=SUPERVISED_WINDOW_SEC,
        overlap_ratio=SUPERVISED_OVERLAP
    )

    # Change [N, Time, Channels] -> [N, Channels, Time]
    X_pretrain = np.transpose(X_pretrain, (0, 2, 1))
    X_supervised = np.transpose(X_supervised, (0, 2, 1))

    # 8. Generate Metadata
    meta_pretrain = generate_metadata_csv(X_pretrain, y_pretrain, groups_pretrain, ds_pretrain, 'pretrain')
    meta_supervised = generate_metadata_csv(X_supervised, y_supervised, groups_supervised, ds_supervised, 'supervised')

    # Combine and save
    master_metadata = pd.concat([meta_pretrain, meta_supervised], ignore_index=True)
    meta_path = 'data/processed/har_metadata.csv'
    master_metadata.to_csv(meta_path, index=False)
    print(f" -> Saved Master Metadata CSV: {meta_path}")

    # SAVE THE ARRAYS!
    save_processed_data(X_pretrain, X_supervised, y_supervised)

    return X_pretrain, X_supervised, y_supervised

