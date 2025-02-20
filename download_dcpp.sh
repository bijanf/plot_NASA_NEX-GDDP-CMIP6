#!/usr/bin/env bash
set -ex 
# Set these to match your environment:
REMOTE_USER="k202196"
REMOTE_HOST="levante.dkrz.de"
# Adjust or remove if you're using the default SSH port 22
LOCAL_DIR="/media/bijan/BIJI/nc-files/"  # Local directory where files should be downloaded

# Path to your file listing remote paths:
INPUT_FILE="data_dcpp.txt"

# Make sure the local directory exists:
mkdir -p "${LOCAL_DIR}"

# Loop through each line of data_dcpp.txt and download the file
while IFS= read -r REMOTE_PATH; do
  # Skip empty lines
  [[ -z "${REMOTE_PATH}" ]] && continue

  # Download using scp
  scp  \
      "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" \
      "${LOCAL_DIR}/"
done < "${INPUT_FILE}"
