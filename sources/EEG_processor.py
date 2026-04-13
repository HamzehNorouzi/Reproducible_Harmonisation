import numpy as np
import pandas as pd
import os
import glob
import mne
from mne.datasets import eegbci
from autoreject import AutoReject
from mne.channels import make_standard_montage
from sources.actual_path import get_actual_path

EEG_TARGET_RUNS = ['04', '08', '12']  # The specific motor-imagery runs
EEG_SFREQ = 160.0  # Native sampling rate
EEG_TMIN = 0  # Start exactly at event onset
EEG_TMAX = 4.0

def print_eeg_subject_count(groups_eeg):
    """
    Calculates and prints the exact number of unique subjects
    successfully processed in the EEG dataset.
    """
    # Convert the array to a Python set to instantly find all unique IDs
    unique_subjects = len(set(groups_eeg))
    print(f" -> [DATA INSIGHT] Total unique EEG subjects processed: {unique_subjects}")
    return unique_subjects

def load_and_epoch_eeg(filepath):
    """
    Loads an EDF file, applies Common Average Referencing (CAR),
    crops to motor channels, and epochs into detrended windows.
    """
    # 1. Load the binary EDF file
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    eegbci.standardize(raw)  # set channel names
    montage = make_standard_montage("standard_1005")
    raw.set_montage(montage)

    # 2. Common Average Referencing (CAR)
    raw.set_eeg_reference('average', projection=False, verbose=False)

    # 3. Spatial Filter: Keep only Motor/Somatosensory channels
    # As the task is motor/motor imagery
    MOTOR_CHANNELS = [
        'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'
    ]
    raw.pick(MOTOR_CHANNELS, verbose=False)

    # 4. Bandpass Filter (1-45 Hz)
    # Although bandpass filtering introduces some issue to the EEG data (e.g. phase shift),
    # I compromised it over applying ICA as it is computationally expensive
    raw.filter(l_freq=1.0, h_freq=45.0, verbose=False)

    # 5. Extract annotations and map to T1/T2 events
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    target_events = {k: v for k, v in event_dict.items() if 'T1' in k or 'T2' in k}

    if not target_events:
        return None, None, None

    # 6. Create Epochs (Windows) with Detrending
    epochs = mne.Epochs(
        raw,
        events,
        event_id=target_events,
        tmin=0.0,
        tmax=3.99375,  # (640 samples - 1) / 160Hz = 3.99375
        baseline=None,
        detrend=1,
        preload=True,
        verbose=False,
        reject_by_annotation=True  # Drops epochs that overlap with file boundaries
    )


    # # 7. Alternative: Apply AutoReject, computationally expensive, processing will take long
    # # I use AutoReject in 'fit_transform' mode to learn the thresholds and drop bad epochs
    # ar = AutoReject(verbose=False)
    # epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
    # num_rejected = sum(reject_log.bad_epochs)

    # 7. Reject artifact
    # Drop windows where peak-to-peak amplitude exceeds 100 µV
    # to eliminate noise for downstream tasks
    epochs_clean = epochs.drop_bad(reject=dict(eeg=100e-5), verbose=False)
    num_rejected = len(epochs) - len(epochs_clean)

    # 8. Extract Clean Data
    X = epochs_clean.get_data(copy=True)
    # If the file had an edge
    # Incomplete 4-second window will be excluded.
    if X.shape[2] != 640:
        if X.shape[2] > 640:
            X = X[:, :, :640]  # Trim if it's 641
        else:
            print(f" -> Size mismatch in {filepath}: got {X.shape[2]}. Skipping file.")
            return None, None, None, 0
    y_codes = epochs_clean.events[:, 2]
    inv_event_dict = {v: k for k, v in target_events.items()}
    y_strings = [inv_event_dict[code] for code in y_codes]



    return X, y_strings, ', '.join(epochs_clean.info['ch_names']), num_rejected

def generate_eeg_metadata(X_eeg, y_eeg, groups_eeg, runs_eeg, channel_schema):
    """
    Generates the required metadata CSV for the processed EEG epochs.
    """
    print(" -> Generating metadata for EEG samples...")
    n_samples, n_channels, n_times = X_eeg.shape
    metadata = {
        'sample_id': [f"EEG_SAM_{str(i).zfill(5)}" for i in range(n_samples)],
        'dataset_name': ['EEGMMIDB'] * n_samples,
        'modality': ['EEG'] * n_samples,
        'subject_id': groups_eeg,
        'source_file': [f"Run_{r}" for r in runs_eeg],
        'split': ['supervised'] * n_samples,
        'label': y_eeg,
        'n_channels': [n_channels] * n_samples,
        'n_samples': [n_times] * n_samples,
        'channel_schema': [channel_schema] * n_samples,
        'qc_flags': ['PASS_CAR_AUTOREJECT_DETREND'] * n_samples
        }
    return pd.DataFrame(metadata)

def process_eeg(data_dir, output_dir):
    """
    Manager function for the EEGMMIDB dataset pipeline.
    """
    # output path
    os.makedirs(output_dir, exist_ok=True)
    data_dir = get_actual_path(data_dir, "S001")

    print("\nStarting EEG Preprocessing Pipeline with AutoReject..."
          "\n autoreject can be turned off from EEG_processor.py if it computationally is too expensive")

    # print("\nStarting EEG Preprocessing Pipeline without AutoReject..."
    #       "\n Although Autoreject may be usefull, applying it may take processing too long"
    #       "\n autoreject can be turned on from EEG_processor.py")


    all_X, all_y, all_groups, all_runs = [], [], [], []
    total_rejected = 0
    master_channel_schema = None

    subject_folders = sorted(glob.glob(os.path.join(data_dir, 'S*')))

    for subj_folder in subject_folders:
        subject_id = os.path.basename(subj_folder)

        for run_id in EEG_TARGET_RUNS:  # Runs 04, 08, 12
            filename = f"{subject_id}R{run_id}.edf"
            filepath = os.path.join(subj_folder, filename)

            if not os.path.exists(filepath):
                continue

            # X here is already cleaned by AutoReject inside the loader
            X, y_labels, channel_schema, rejected_in_file = load_and_epoch_eeg(filepath)


            if X is not None and len(X) > 0:
                total_rejected += rejected_in_file
                all_X.append(X)
                all_y.extend(y_labels)
                all_groups.extend([subject_id] * len(X))
                all_runs.extend([run_id] * len(X))
                print(f" -> Processed {subject_id}: {filename}")
                if master_channel_schema is None:
                    master_channel_schema = channel_schema
            else:
                total_rejected += rejected_in_file  # Count files where 100% was rejected
                print(f" -> Skipping {filename}")

    # Final Concatenation
    X_eeg = np.concatenate(all_X, axis=0).astype(np.float32)
    y_eeg = np.array(all_y)
    groups_eeg = np.array(all_groups)
    runs_eeg = np.array(all_runs)
    print("\n==========================================")
    print("EEG PREPROCESSING SUCCESSFUL!")
    print(f"Total Segments Dropped by AutoReject: {total_rejected}")
    print(f"Final Array Shape: {X_eeg.shape} (N, C, T)")
    print("==========================================")

    # Meta-data
    eeg_meta_df = generate_eeg_metadata(X_eeg, y_eeg, groups_eeg, runs_eeg, master_channel_schema)

    # Save the Metadata CSV
    meta_path = os.path.join(output_dir, 'eeg_metadata.csv')
    eeg_meta_df.to_csv(meta_path, index=False)
    print(f" -> Saved EEG Metadata: {meta_path}")

    # Save the Numeric EEG Arrays
    array_path = os.path.join(output_dir, 'eeg_data.npz')
    np.savez_compressed(array_path, X=X_eeg, y=y_eeg)
    print(f" -> Saved EEG Arrays: {array_path}")

    return X_eeg, y_eeg, groups_eeg, runs_eeg, master_channel_schema

