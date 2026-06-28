#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $(basename "$0") <model>"
  echo "Available models: sssd-ecg"
  exit 1
}

if [[ $# -ne 1 ]]; then
  usage
fi

MODEL="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_BASE_DIR="${SCRIPT_DIR}/model"

generate_requirements_from_src() {
  local model_dir="$1"
  local requirements_file="$2"

  python3 - "${model_dir}" "${requirements_file}" <<'PY'
import ast
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
requirements_file = Path(sys.argv[2])
src_dir = model_dir / "src"

import_to_package = {
    "torch": "torch",
    "numpy": "numpy",
    "scipy": "scipy",
    "tqdm": "tqdm",
    "yaml": "PyYAML",
    "einops": "einops",
    "opt_einsum": "opt-einsum",
    "pytorch_lightning": "pytorch-lightning",
    "pandas": "pandas",
    "wfdb": "wfdb",
    "resampy": "resampy",
    "h5py": "h5py",
    "skimage": "scikit-image",
    "torchvision": "torchvision",
    "torchsummary": "torchsummary",
}

packages: set[str] = set()

for py_file in src_dir.rglob("*.py"):
    try:
        tree = ast.parse(py_file.read_text())
    except SyntaxError:
        continue

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in import_to_package:
                    packages.add(import_to_package[root])
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in import_to_package:
                packages.add(import_to_package[root])

ordered = sorted(packages)
requirements_file.write_text("\n".join(ordered) + "\n")
print(f"Wrote {len(ordered)} packages to {requirements_file}")
PY
}

setup_sssd_ecg() {
  local model_dir="$1"
  local requirements_file="${model_dir}/requirements.txt"

  if [[ ! -d "${model_dir}/src" ]]; then
    echo "Error: SSSD-ECG source directory not found at ${model_dir}/src" >&2
    exit 1
  fi

  generate_requirements_from_src "${model_dir}" "${requirements_file}"
}

case "${MODEL}" in
  sssd-ecg)
    MODEL_DIR="${MODEL_BASE_DIR}/SSSD-ECG"
    MODEL_REPO="https://github.com/AI4HealthUOL/SSSD-ECG.git"
    ;;
  *)
    echo "Unknown model: ${MODEL}" >&2
    usage
    ;;
esac

mkdir -p "${MODEL_BASE_DIR}"

if [[ -d "${MODEL_DIR}/.git" ]]; then
  git -C "${MODEL_DIR}" pull --ff-only
elif [[ -e "${MODEL_DIR}" ]]; then
  echo "Error: ${MODEL_DIR} exists but is not a git repository" >&2
  exit 1
else
  git clone "${MODEL_REPO}" "${MODEL_DIR}"
fi

case "${MODEL}" in
  sssd-ecg)
    setup_sssd_ecg "${MODEL_DIR}"
    ;;
esac
