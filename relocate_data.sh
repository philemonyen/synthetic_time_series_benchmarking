#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") <model> <dataset-path>"
  echo "Available models: sssd-ecg"
  echo "Example: $(basename "$0") sssd-ecg /path/to/Dataset"
  exit 1
}

if [[ $# -ne 2 ]]; then
  usage
fi

MODEL="$1"
DATASET_PATH="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_BASE_DIR="${SCRIPT_DIR}/model"

relocate_sssd_ecg_dataset() {
  local dataset_path="$1"
  local model_dir="${MODEL_BASE_DIR}/SSSD-ECG"
  local dest_dir="${model_dir}/src/sssd"
  local data_dir="${dataset_path}/data"
  local labels_dir="${dataset_path}/labels"
  local required_files=(
    "${data_dir}/ptbxl_train_data.npy"
    "${data_dir}/ptbxl_validation_data.npy"
    "${data_dir}/ptbxl_test_data.npy"
    "${labels_dir}/ptbxl_train_labels.npy"
    "${labels_dir}/ptbxl_validation_labels.npy"
    "${labels_dir}/ptbxl_test_labels.npy"
  )

  if [[ ! -d "${model_dir}" ]]; then
    echo "Error: SSSD-ECG model not found at ${model_dir}. Run ./load_model.sh sssd-ecg first." >&2
    exit 1
  fi

  if [[ ! -d "${data_dir}" || ! -d "${labels_dir}" ]]; then
    echo "Error: Expected dataset layout with data/ and labels/ subdirectories under ${dataset_path}" >&2
    exit 1
  fi

  for file_path in "${required_files[@]}"; do
    if [[ ! -f "${file_path}" ]]; then
      echo "Error: Missing required file ${file_path}" >&2
      exit 1
    fi
  done

  mkdir -p "${dest_dir}"
  cp "${data_dir}"/*.npy "${dest_dir}/"
  cp "${labels_dir}"/*.npy "${dest_dir}/"

  echo "Relocated PTB-XL dataset to ${dest_dir}"
}

case "${MODEL}" in
  sssd-ecg)
    relocate_sssd_ecg_dataset "${DATASET_PATH}"
    ;;
  *)
    echo "Unknown model: ${MODEL}" >&2
    usage
    ;;
esac
