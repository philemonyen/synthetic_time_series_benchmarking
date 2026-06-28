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

download_figshare_ptbxl() {
  local dest_dir="$1"
  local figshare_share_url="$2"
  local figshare_article_id="$3"
  shift 3
  local required_data_files=("$@")

  mkdir -p "${dest_dir}"

  local missing=false
  for file_name in "${required_data_files[@]}"; do
    if [[ ! -f "${dest_dir}/${file_name}" ]]; then
      missing=true
      break
    fi
  done

  if [[ "${missing}" == "false" ]]; then
    echo "Preprocessed PTB-XL files already present in ${dest_dir}"
    return 0
  fi

  echo "Downloading preprocessed PTB-XL data from Figshare (${figshare_share_url})"
  python3 - "${dest_dir}" "${figshare_article_id}" "${figshare_share_url}" "${required_data_files[@]}" <<'PY'
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

dest_dir = Path(sys.argv[1])
article_id = sys.argv[2]
share_url = sys.argv[3]
required_files = sys.argv[4:]

def download_file(url: str, target: Path) -> None:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)

def copy_required_files(source_root: Path) -> None:
    for file_name in required_files:
        matches = list(source_root.rglob(file_name))
        if not matches:
            raise FileNotFoundError(f"Could not find {file_name} in downloaded Figshare archive")
        shutil.copy2(matches[0], dest_dir / file_name)

with tempfile.TemporaryDirectory() as tmp_dir:
    staging_dir = Path(tmp_dir)
    files_api = f"https://api.figshare.com/v2/articles/{article_id}/files"
    downloaded = False

    try:
        with urlopen(files_api) as response:
            files = json.load(response)
    except (HTTPError, URLError, json.JSONDecodeError):
        files = []

    if files:
        for file_info in files:
            file_name = file_info["name"]
            file_id = file_info["id"]
            download_url = file_info.get("download_url") or f"https://api.figshare.com/v2/file/download/{file_id}"
            target_path = staging_dir / file_name
            download_file(download_url, target_path)
            if target_path.suffix == ".zip":
                with zipfile.ZipFile(target_path) as archive:
                    archive.extractall(staging_dir)
        copy_required_files(staging_dir)
        downloaded = True
    else:
        archive_path = staging_dir / "ptbxl_preprocessed.zip"
        archive_urls = [
            f"https://ndownloader.figshare.com/articles/{article_id}/versions/latest",
            f"https://api.figshare.com/v2/articles/{article_id}/files",
        ]
        for archive_url in archive_urls:
            try:
                download_file(archive_url, archive_path)
                if zipfile.is_zipfile(archive_path):
                    with zipfile.ZipFile(archive_path) as archive:
                        archive.extractall(staging_dir)
                    copy_required_files(staging_dir)
                    downloaded = True
                    break
            except (HTTPError, URLError, zipfile.BadZipFile, FileNotFoundError):
                continue

    if not downloaded:
        raise RuntimeError(
            "Failed to download preprocessed PTB-XL data from Figshare. "
            f"Check {share_url} manually and place the required .npy files in {dest_dir}"
        )
PY
}

setup_sssd_ecg() {
  local model_dir="$1"
  local requirements_file="${model_dir}/requirements.txt"
  local sssd_src_dir="${model_dir}/src/sssd"

  if [[ ! -d "${model_dir}/src" ]]; then
    echo "Error: SSSD-ECG source directory not found at ${model_dir}/src" >&2
    exit 1
  fi

  generate_requirements_from_src "${model_dir}" "${requirements_file}"

  download_figshare_ptbxl \
    "${sssd_src_dir}" \
    "https://figshare.com/s/43df16e4a50e4dd0a0c5" \
    "21922947" \
    "ptbxl_train_data.npy" \
    "ptbxl_train_labels.npy" \
    "ptbxl_test_labels.npy"
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
