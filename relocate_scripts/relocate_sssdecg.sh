#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_BASE_DIR="${PROJECT_DIR}/model"
DATASET_PATH="${PROJECT_DIR}/Dataset"
model_dir="${MODEL_BASE_DIR}/SSSD-ECG"
dest_dir="${model_dir}/src/sssd"
data_dir="${DATASET_PATH}/data"
labels_dir="${DATASET_PATH}/labels"
required_files=(
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
  echo "Error: Expected dataset layout with data/ and labels/ under ${DATASET_PATH}" >&2
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
