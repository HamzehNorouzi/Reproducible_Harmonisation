#!/bin/bash
# ==============================================================================
# Script Name: run_pipeline.sh
# Description: End-to-end orchestration for the Multimodal Preprocessing Pipeline
# Author: Hamzeh Norouzi
# Date: 13/04/2026
# ==============================================================================

set -e
set -o pipefail

# --- Signature Banner ---
print_signature() {
    # Using Cyan text for a sleek, terminal-hacker aesthetic
    echo -e "\033[0;36m" 
    echo "==============================================================="
    echo "   __  __       _ _   _                     _       _          "
    echo "  |  \/  |     | | | (_)                   | |     | |         "
    echo "  | \  / |_   _| | |_ _ _ __ ___   ___   __| | __ _| |         "
    echo "  | |\/| | | | | | __| | '_ \` _ \ / _ \ / _\` |/ _\` | |         "
    echo "  | |  | | |_| | | |_| | | | | | | (_) | (_| | (_| | |         "
    echo "  |_|  |_|\__,_|_|\__|_|_| |_| |_|\___/ \__,_|\__,_|_|         "
    echo "                                       Preprocessing Pipeline  "
    echo "==============================================================="
    echo "  Engineered by: Hamzeh Norouzi"
    echo "==============================================================="
    echo -e "\033[0m" # Reset terminal color back to normal
}

# Print the signature
print_signature

sleep 1 # Pause for 1 second so the user can read the banner
echo "==================================================="
echo "  Starting Multimodal Preprocessing Pipeline"
echo "==================================================="

# --- 1. Folder Setup ---
echo "[1/5] Setting up directories..."
mkdir -p data/raw/PAMAP2 data/raw/WISDM data/raw/EEGMMIDB data/raw/PTB-XL data/raw/mHealth
mkdir -p data/interim data/processed
mkdir -p reports submission_sample

# --- 2. Data Download (Optimized) ---
echo "[2/5] Downloading Datasets..."

# Helper function to download and log to manifest
download_dataset() {
    local name=$1
    local url=$2
    local dest_dir=$3
    local dl_type=${4:-"zip"}
    local MANIFEST_FILE="data/raw/download_manifest.json"

    # Initialize manifest if it doesn't exist
    if [ ! -f "$MANIFEST_FILE" ]; then
        echo "[" > "$MANIFEST_FILE"
    fi

    # Check if folder already contains files
    if [ "$(ls -A "$dest_dir" 2>/dev/null)" ]; then
        echo " -> [SKIP] $name data already exists in $dest_dir."
    else
        echo " -> Downloading $name..."
        if [ "$dl_type" == "recursive" ]; then
            if ! wget -r -N -c -np -q --show-progress "$url" -P "$dest_dir"; then
                echo "ERROR: Failed to download $name from $url" >&2
                exit 1
            fi
        else
            local filename=$(basename "$url")
            local filepath="$dest_dir/$filename"
            if wget -q --show-progress -c -O "$filepath" "$url"; then
                echo " -> Extracting $name..."
                unzip -q -n "$filepath" -d "$dest_dir"
            else
                echo "ERROR: Failed to download $name from $url" >&2
                exit 1
            fi
        fi
    fi

    # Log to manifest (Requires date command)
    local DATE_TODAY=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    cat <<EOF >> "$MANIFEST_FILE"
  {
    "dataset": "$name",
    "url": "$url",
    "download_date": "$DATE_TODAY",
    "status": "success",
    "local_path": "$dest_dir"
  },
EOF
}

# Execute Downloads
# Fast PhysioNet Downloads
download_dataset "EEGMMIDB" "https://physionet.org/files/eegmmidb/1.0.0/" "data/raw/EEGMMIDB" "recursive" # fast download
download_dataset "PTB-XL" "https://physionet.org/files/ptb-xl/1.0.3/" "data/raw/PTB-XL" "recursive" # fast download

# Standard UCI/Zip Downloads
download_dataset "PAMAP2" "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip" "data/raw/PAMAP2" "zip"
download_dataset "WISDM" "https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip" "data/raw/WISDM" "zip"
download_dataset "mHealth" "https://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip" "data/raw/mHealth" "zip"

echo " -> Checking for and extracting any nested zips (UCI format)..."
find data/raw -mindepth 2 -type f -name "*.zip" -execdir unzip -q -n {} \;

# Clean up JSON manifest formatting
sed -i '$ s/,$//' data/raw/download_manifest.json
echo "]" >> data/raw/download_manifest.json

# --- 3. Virtual Environment Setup ---
echo "[3/5] Setting up Python Virtual Environment..."
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo " -> Virtual environment not found. Creating one..."
    # Ensure we use python3 or python depending on system alias
    if command -v python3 &>/dev/null; then
        python3 -m venv $VENV_DIR
    else
        python -m venv $VENV_DIR
    fi

    # Cross-platform activation
    if [ -f "$VENV_DIR/Scripts/activate" ]; then
        source $VENV_DIR/Scripts/activate  # Windows (Git Bash)
    else
        source $VENV_DIR/bin/activate      # Mac/Linux
    fi

    echo " -> Upgrading pip and installing requirements..."
    pip install --upgrade pip -q
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt -q
        echo " -> Requirements installed successfully."
    else
        echo " -> WARNING: requirements.txt not found! Skipping installation."
    fi
else
    echo " -> Virtual environment already exists. Activating..."
    if [ -f "$VENV_DIR/Scripts/activate" ]; then
        source $VENV_DIR/Scripts/activate
    else
        source $VENV_DIR/bin/activate
    fi
fi

# --- 4. Execute Python Pipeline ---
echo "[4/5] Running Preprocessing, Validation, and Manifest Generation..."
echo "Tracking global RAM and CPU usage via /usr/bin/time..."

# Because the venv is activated, 'python' now perfectly points to your isolated environment
/usr/bin/time -v python process_main.py 2> reports/global_resource_usage.txt

echo " -> Python pipeline finished. Global resource metrics saved to reports/global_resource_usage.txt"

# --- 5. Clean Up & Finish ---
echo "[5/5] Wrapping up..."
# Deactivate the virtual environment so we don't mess up the user's terminal
deactivate

echo "==================================================="
echo "  PIPELINE COMPLETE! "
echo "  Check 'submission_sample/' and 'reports/'"
echo "==================================================="
