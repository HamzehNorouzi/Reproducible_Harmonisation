from sources.validation import validate_eeg_outputs, validate_ecg_outputs, validate_har_outputs
from sources.EEG_processor import process_eeg
from sources.ECG_processor import process_ecg
from sources.har_data_processor import preprocess_har
from sources.tracking_resources import track_resources
from sources.submission_pack import create_submission_pack
from sources.ML_manifest import generate_modality_manifest
from sources.EEG_processor import print_eeg_subject_count
from sources.output_log import OutputLogger
import sys

if __name__ == "__main__":

    # Initialize a dictionary to store the resource estimates
    resource_report = {}

    # # --- 1. EEG Modality ---
    print("\n[STEP 1] Processing EEG...")
    eeg_args = ("data/raw/EEGMMIDB", "data/processed")
    (X_eeg, y_eeg, groups_eeg, runs_eeg, master_channel_schema), eeg_time, eeg_ram = track_resources(process_eeg,
                                                                                                     *eeg_args)
    print(f"{print_eeg_subject_count(groups_eeg)} EEG subjects found!")
    resource_report['EEG'] = {'time': eeg_time, 'ram': eeg_ram}

    # --- 2. HAR Modality ---
    print("\n[STEP 2] Processing HAR...")
    (X_pretrain, X_supervised, y_supervised), har_time, har_ram = track_resources(preprocess_har)
    resource_report['HAR'] = {'time': har_time, 'ram': har_ram}

    # --- 3. ECG Modality ---
    print("\n[STEP 3] Processing ECG...")
    ecg_args = ("data/raw/PTB-XL", "data/processed")
    (X_train, X_val, X_test, ecg_meta_df), ecg_time, ecg_ram = track_resources(process_ecg, *ecg_args)
    resource_report['ECG'] = {'time': ecg_time, 'ram': ecg_ram}

    # --- 4. Final Reporting (Console + File) ---
    resource_text = "\n" + "=" * 30 + "\nRESOURCE ESTIMATE SUMMARY\n" + "=" * 30 + "\n"
    for modality, metrics in resource_report.items():
        resource_text += f"{modality}:\n"
        resource_text += f"  - Runtime: {metrics['time']:.2f} seconds\n"
        resource_text += f"  - Peak RAM: {metrics['ram']:.2f} MB\n"

    # Print to console and write to file
    print(resource_text)
    with open("reports/resource_estimate.txt", "w") as f:
        f.write(resource_text)
    print(" -> Saved Resource Estimate to: reports/resource_estimate.txt")

    # --- 5. Validation (Capture to File) ---
    print("\n[STEP 5] Running Validation Checks...")

    # Temporarily route all print() statements through the OutputLogger
    original_stdout = sys.stdout
    sys.stdout = OutputLogger("reports/validation_report.txt")

    validate_har_outputs()
    print("\nHAR output validated!")
    validate_eeg_outputs()
    print("\nEEG output validated!")
    validate_ecg_outputs()
    print("\nECG output validated!")

    # Restore normal printing behavior
    sys.stdout = original_stdout
    print("\n -> Saved Validation Report to: reports/validation_report.txt")

    # --- 6. Manifest & Packaging ---
    print("\nGenerating Manifest...")
    generate_modality_manifest()

    print("\nCreating submission pack...")
    create_submission_pack()