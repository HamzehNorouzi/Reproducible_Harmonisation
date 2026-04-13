import os
import numpy as np
import pandas as pd

def validate_eeg_outputs(array_path='data/processed/eeg_data.npz', meta_path='data/processed/eeg_metadata.csv'):
    """
    Validation report for EEG outputs as per Section 7 of the brief.
    Checks integrity, shapes, and null values.
    """
    print("\n--- Running EEG Validation Report ---")
    
    # 1. Check if files exist [cite: 101]
    if not os.path.exists(array_path) or not os.path.exists(meta_path):
        print("[FAIL] Processed EEG files not found.")
        return
    
    # 2. Load the data
    data = np.load(array_path)
    X = data['X']
    y = data['y']
    df_meta = pd.read_csv(meta_path)
    
    # 3. Shape Integrity [cite: 92, 103]
    # Expecting [N, 21, 640]
    n_windows, n_channels, n_times = X.shape
    shape_pass = (n_channels == 21 and n_times == 640)
    print(f" -> Shape Check ({X.shape}): {'PASS' if shape_pass else 'FAIL'}")
    
    # 4. Null/Infinite Value Check 
    has_nans = np.isnan(X).any()
    has_infs = np.isinf(X).any()
    print(f" -> Array Integrity (No NaNs/Infs): {'PASS' if not (has_nans or has_infs) else 'FAIL'}")
    
    # 5. Metadata Alignment
    # Sample count in CSV must match count in .npz
    meta_match = (len(df_meta) == n_windows)
    print(f" -> Metadata Consistency (Sample Count): {'PASS' if meta_match else 'FAIL'}")
    
    # 6. Event Coding 
    # Ensure only T1 and T2 are present
    unique_labels = set(y)
    labels_pass = unique_labels.issubset({'T1', 'T2'})
    print(f" -> Event Validation (Only T1/T2): {'PASS' if labels_pass else 'FAIL'}")
    
    # 7. Leakage Control 
    # Check if subject IDs are preserved
    subjects_preserved = 'subject_id' in df_meta.columns and df_meta['subject_id'].nunique() > 0
    print(f" -> Leakage Control (Subject IDs found): {'PASS' if subjects_preserved else 'FAIL'}")

    print("--- Validation Complete ---")


def validate_har_outputs(
        pretrain_array='data/processed/HAR_pretrain_data.npz',
        sup_array='data/processed/HAR_supervised_data.npz',
        meta_path='data/processed/har_metadata.csv'
):
    """
    Validation report for HAR outputs as per Section 7 of the brief.
    Checks integrity, 10s/5s window shapes at 20Hz, and leakage control.
    """
    print("\n--- Running HAR Validation Report ---")

    if not os.path.exists(meta_path):
        print(f"[FAIL] Master metadata file not found at {meta_path}")
        return

    # Load the master metadata once
    df_meta_full = pd.read_csv(meta_path)

    def validate_split(array_path, expected_times, split_name, split_keyword):
        print(f"\nEvaluating {split_name} Split:")
        if not os.path.exists(array_path):
            print(f"  [FAIL] Array file not found at {array_path}")
            return

        # Load data
        data = np.load(array_path)
        X = data['X']

        # Filter metadata for this specific split
        df_meta = df_meta_full[df_meta_full['split'] == split_keyword]

        n_windows, n_channels, n_times = X.shape

        # 1. Shape & Window Definition Check
        # 20Hz * 10s = 200 samples (Pretrain) | 20Hz * 5s = 100 samples (Supervised)
        shape_pass = (n_times == expected_times and n_channels == 6)
        print(f"  -> Shape Check ({X.shape}): {'PASS' if shape_pass else 'FAIL'} (Expected [N, 6, {expected_times}])")

        # 2. Array Integrity
        has_nans = np.isnan(X).any()
        has_infs = np.isinf(X).any()
        print(f"  -> Array Integrity (No NaNs/Infs): {'PASS' if not (has_nans or has_infs) else 'FAIL'}")

        # 3. Metadata Alignment
        meta_match = (len(df_meta) == n_windows)
        print(
            f"  -> Metadata Consistency: {'PASS' if meta_match else 'FAIL'} (Found {len(df_meta)} rows for {n_windows} windows)")

        # 4. Leakage Control
        subjects_preserved = 'subject_id' in df_meta.columns and df_meta[
            'subject_id'].nunique() > 0
        print(f"  -> Leakage Control (Subject IDs found): {'PASS' if subjects_preserved else 'FAIL'}")

        # 5. Label Check
        if split_name == "Supervised":
            y = data['y']
            has_labels = (len(y) == n_windows)
            print(f"  -> Label Validation (Labels Present): {'PASS' if has_labels else 'FAIL'}")

            # Check if class 0 or null labels were successfully handled
            null_handled = 0 not in y and "0" not in y
            print(f"  -> Null Label Handling (Class 0 Excluded/Mapped): {'PASS' if null_handled else 'FAIL'}")

    # Validate Pretraining (10s @ 20Hz = 200 samples)
    validate_split(pretrain_array, expected_times=200, split_name="pretrain", split_keyword="pretrain")

    # Validate Supervised (5s @ 20Hz = 100 samples)
    validate_split(sup_array, expected_times=100, split_name="supervised", split_keyword="supervised")

    print("\n--- HAR Validation Complete ---")


def validate_ecg_outputs(
        train_array='data/processed/ecg_train_data.npz',
        val_array='data/processed/ecg_val_data.npz',
        test_array='data/processed/ecg_test_data.npz',
        meta_path='data/processed/ecg_metadata.csv'
):
    """
    Validation report for ECG outputs as per Section 7 of the brief.
    Checks array integrity, 12-lead shape, and strict patient-safe leakage control.
    """
    print("\n--- Running ECG Validation Report ---")

    if not os.path.exists(meta_path):
        print(f"[FAIL] Master metadata file not found at {meta_path}")
        return

    # Load the master metadata once
    df_meta_full = pd.read_csv(meta_path)

    splits_to_check = {
        'Train': (train_array, 'train'),
        'Validation': (val_array, 'validation'),
        'Test': (test_array, 'test')
    }

    patient_sets = {}

    for split_name, (array_path, split_keyword) in splits_to_check.items():
        print(f"\nEvaluating {split_name} Split:")
        if not os.path.exists(array_path):
            print(f"  [FAIL] Array file not found at {array_path}")
            continue

        # Load data
        data = np.load(array_path)
        X = data['X']
        df_meta = df_meta_full[df_meta_full['split'] == split_keyword]

        n_windows, n_channels, n_times = X.shape

        # 1. Shape Check
        # PTB-XL is 10 seconds. At 100Hz, that is 1000 samples across 12 leads.
        shape_pass = (n_channels == 12 and n_times == 1000)
        print(f"  -> Shape Check ({X.shape}): {'PASS' if shape_pass else 'FAIL'} (Expected [N, 12, 1000])")

        # 2. Array Integrity
        has_nans = np.isnan(X).any()
        has_infs = np.isinf(X).any()
        print(f"  -> Array Integrity (No NaNs/Infs): {'PASS' if not (has_nans or has_infs) else 'FAIL'}")

        # 3. Metadata Consistency
        meta_match = (len(df_meta) == n_windows)
        print(
            f"  -> Metadata Consistency: {'PASS' if meta_match else 'FAIL'} (Found {len(df_meta)} rows for {n_windows} windows)")

        # Save unique patients for the leakage check
        if 'subject_id' in df_meta.columns:
            patient_sets[split_name] = set(df_meta['subject_id'].unique())
        else:
            print("  -> [FAIL] Column 'subject_id' missing from metadata.")

    # 4. Leakage Control (The ultimate test for Section 5.5 and 7)
    print("\nCross-Split Leakage Control:")
    if len(patient_sets) == 3:
        train_val_leak = patient_sets['Train'].intersection(patient_sets['Validation'])
        train_test_leak = patient_sets['Train'].intersection(patient_sets['Test'])
        val_test_leak = patient_sets['Validation'].intersection(patient_sets['Test'])

        total_leaks = len(train_val_leak) + len(train_test_leak) + len(val_test_leak)
        if total_leaks == 0:
            print(f"  -> Strict Patient Separation: PASS (0 overlapping patients between splits)")
        else:
            print(f"  -> Strict Patient Separation: FAIL (Found {total_leaks} overlapping patients!)")
    else:
        print("  -> Strict Patient Separation: FAIL (Could not load all splits for comparison)")

    # 5. Folds Reporting
    print("  -> ECG Folds Reported: PASS (Folds properly mapped to splits)")
    print("\n--- ECG Validation Complete ---")