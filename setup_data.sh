#!/bin/bash
# setup_data.sh
# Orchestration script for multimodal dataset setup

# 1. Fail clearly if any command fails (Requirement 5.1)
set -e
set -o pipefail

echo "Download and folder setup"

# --- CONFIGURATION ---
DATE_TODAY=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# 2. Folder Setup
echo "Creating project directories..."
mkdir -p data/raw/PAMAP2 data/raw/WISDM data/raw/EEGMMIDB data/raw/PTB-XL data/raw/mHealth
mkdir -p data/interim
mkdir -p data/processed
mkdir -p configs
mkdir -p reports
mkdir -p submission_sample

echo "Directories created successfully."

# 3. Initialize Machine-Readable Manifest
MANIFEST_FILE="data/raw/download_manifest.json"
echo "[" > $MANIFEST_FILE

# download and log to manifest
download_dataset() {
    local name=$1
    local url=$2
    local dest_dir=$3
    local dl_type=${4:-"zip"} # Defaults to "zip" if no 4th argument is provided

    # --- NEW CHECK: Does the folder already contain files? ---
    # `ls -A` lists files. If the output is not empty, the data is already there.
    if [ "$(ls -A "$dest_dir" 2>/dev/null)" ]; then
        echo " -> [SKIP] $name data already exists in $dest_dir."
    else
        echo "Downloading $name..."

        if [ "$dl_type" == "recursive" ]; then
            # ---> THE FAST PHYSIONET METHOD <---
            if wget -r -N -c -np -q --show-progress "$url" -P "$dest_dir"; then
                echo "Successfully downloaded $name."
            else
                echo "ERROR: Failed to download $name from $url" >&2
                exit 1
            fi
        else
            # ---> THE STANDARD ZIP METHOD <---
            local filename=$(basename "$url")
            local filepath="$dest_dir/$filename"

            if wget -q --show-progress -c -O "$filepath" "$url"; then
                echo "Successfully downloaded $name."
                echo "Extracting $name..."
                unzip -q -n "$filepath" -d "$dest_dir"
            else
                echo "ERROR: Failed to download $name from $url" >&2
                exit 1
            fi
        fi
    fi

    # ---> LOG TO MANIFEST <---
    # This runs whether the file was freshly downloaded or skipped, 
    # ensuring your JSON manifest always accurately reflects what you have.
    cat <<EOF >> $MANIFEST_FILE
  {
    "dataset": "$name",
    "url": "$url",
    "download_date": "$DATE_TODAY",
    "status": "success",
    "local_path": "$dest_dir"
  },
EOF
}

# 4. Execute Downloads
download_dataset "EEGMMIDB" "https://physionet.org/files/eegmmidb/1.0.0/" "data/raw/EEGMMIDB" "recursive" # fast download
download_dataset "PTB-XL" "https://physionet.org/files/ptb-xl/1.0.3/" "data/raw/PTB-XL" "recursive" # fast download

# Standard UCI/Zip Downloads
download_dataset "PAMAP2" "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip" "data/raw/PAMAP2" "zip"
download_dataset "WISDM" "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip" "data/raw/WISDM" "zip"
download_dataset "mHealth" "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip" "data/raw/mHealth" "zip"

# Clean up JSON formatting (remove last comma and close array)
sed -i '$ s/,$//' $MANIFEST_FILE
echo "]" >> $MANIFEST_FILE

echo "========================================"
echo "Setup complete! Manifest saved to $MANIFEST_FILE."
echo "You can view the manifest by running: cat $MANIFEST_FILE"
