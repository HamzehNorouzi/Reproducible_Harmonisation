# Multimodal Preprocessing Pipeline for Self-Supervised Learning

This repository contains an end-to-end reproducible data engineering pipeline. It automatically downloads, harmonises, and preprocesses three distinct physiological modalities—Human Activity Recognition (HAR), Electroencephalography (EEG), and Electrocardiography (ECG)—into standardised 3D arrays (`[N, Channels, Time]`) optimised for downstream Self-Supervised Learning (SSL) architectures.

## Reproducing the Pipeline

This pipeline is fully containerised via a master shell wrapper. You do **not** need to manually download data, unzip files, or build virtual environments.

**Prerequisites:** 
* A Unix-like terminal (Linux, macOS, or Windows Git Bash)
* Python 3.8+
* standard CLI tools: `wget`, `unzip`, `time`

**Execution Commands:**
Open your terminal in the root directory of this project and run:

```bash
# 1. Make the shell script executable
chmod +x run.sh

# 2. Run the master pipeline
./run.sh

The shell script will create a virtual environmnt, install requirements, download data, and process them automatically.

Validation report and resource estimates are saved to reports/validation_report.txt

Global resource metrics, such as RAM usage, and run time, are saved to reports/global_resource_usage.txt
```

Project Structure

├── run.sh          # Master shell wrapper
├── requirements.txt         # Package dependencies
├── process_main.py          # Python orchestrator
├── README.md                # This document
│
├── config/
│   ├── __init__.py
│   └── settings.py          # Global static variables (Target SR, labels, etc.)
│
├── sources/                 # Modular Python processors
│   ├── __init__.py
│   ├── actual_path.py       # Dynamic path-finder for cross-OS compatibility
│   ├── ECG_processor.py     # PTB-XL processor (100Hz, patient-safe splits)
│   ├── EEG_processor.py     # PhysioNet EEG processor (Runs 4,8,12, CAR, AutoReject)
│   ├── har_data_processor.py# HAR Harmonisation (PAMAP2, mHealth, WISDM)
│   ├── validation.py        # Algorithmic integrity & leakage checks
│   ├── tracking_resources.py# RAM and Runtime profiling decorator
│   ├── ML_manifest.py       # Machine-readable output logging
│   └── submission_pack.py   # 100-sample extraction logic
│
├── data/
│   ├── raw/                 # Downloaded datasets and download_manifest.json
│   └── processed/           # Final .npz arrays, metadata CSVs, and pipeline_manifest.csv
│
├── reports/                 # Global shell resource tracking logs
└── submission_sample/       # Lightweight 100-sample outputs for grading


**Modality Processing Details**
1. HAR Harmonisation (PAMAP2, WISDM, mHealth)
Standardisation: All sets are downsampled to a unified 20 Hz and mapped to a shared 6-channel IMU schema (Wrist Accelerometer X/Y/Z, Gyroscope X/Y/Z).

Pretraining Windows: 10-second duration, 0% overlap.

Supervised Windows: 5-second duration, 50% overlap.

Labels: Mapped to a unified 6-class semantic schema, safely dropping transient/unmapped activities.

2. EEG (EEGMMIDB)
Filtering: Explicitly targets strictly Motor Imagery files (Runs 04, 08, 12).

Preprocessing: Applies Common Average Referencing (CAR), Bandpass filtering (1-45 Hz), and automated artefact rejection via autoreject.

Epoching: Precisely crops windows to 4.0 seconds relative to T1/T2 event onset markers.

3. ECG (PTB-XL)
Resource Optimisation: Leverages the 100Hz subset of the data to drastically reduce memory overhead.


**Final Outputs & Downstream Integration**
All processed outputs are saved in data/processed/ (and a 100-sample subset in submission_sample/).

Output Formatting
To strictly comply with downstream Deep Learning (PyTorch/TensorFlow) ingestion patterns, all signal arrays are natively transposed and saved as compressed float32 NumPy archives:

Shape: [N, Channels, Time] (e.g., [Batch, 12, 1000] for ECG)

Format: Compressed .npz

Validation
The pipeline features a robust internal validation suite (validation.py) that computationally proves the arrays contain no infinite/NaN values, match their expected tensor dimensions, and feature zero cross-split patient leakage.
