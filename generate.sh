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

generate_sssd_ecg() {
  local figshare_share_url="https://figshare.com/s/43df16e4a50e4dd0a0c5"
  local figshare_article_id="21922947"
  local required_data_files=(
    "ptbxl_train_data.npy"
    "ptbxl_train_labels.npy"
    "ptbxl_test_labels.npy"
  )
  local model_repo_dir
  local sssd_src_dir
  local default_config
  local training_date
  local trained_dir
  local synthesis_dir

  if [[ -d "${MODEL_BASE_DIR}/SSSD-ECG/SSSD-ECG" ]]; then
    model_repo_dir="${MODEL_BASE_DIR}/SSSD-ECG/SSSD-ECG"
  elif [[ -d "${MODEL_BASE_DIR}/SSSD-ECG" ]]; then
    model_repo_dir="${MODEL_BASE_DIR}/SSSD-ECG"
  else
    echo "Error: SSSD-ECG repository not found. Run ./load_model.sh sssd-ecg first." >&2
    exit 1
  fi

  sssd_src_dir="${model_repo_dir}/src/sssd"
  default_config="${sssd_src_dir}/config/config_SSSD_ECG.json"
  training_date="$(date +%Y-%m-%d)"
  trained_dir="${model_repo_dir}/trained/${training_date}"
  synthesis_dir="${SCRIPT_DIR}/synthesis/SSSD-ECG/${training_date}"

  if [[ ! -d "${sssd_src_dir}" ]]; then
    echo "Error: SSSD source directory not found at ${sssd_src_dir}" >&2
    exit 1
  fi

  if [[ ! -f "${default_config}" ]]; then
    echo "Error: SSSD-ECG config not found at ${default_config}" >&2
    exit 1
  fi

  download_figshare_ptbxl() {
    local dest_dir="$1"
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

  prepare_training_config() {
    local config_path="${trained_dir}/config.json"
    mkdir -p "${trained_dir}"
    cp "${default_config}" "${config_path}"

    python3 - "${config_path}" "${trained_dir}" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
trained_dir = Path(sys.argv[2]).resolve()

config = json.loads(config_path.read_text())
config["train_config"]["output_directory"] = str(trained_dir)
config["gen_config"]["output_directory"] = str(trained_dir)
config["gen_config"]["ckpt_path"] = f"{trained_dir}/"
config_path.write_text(json.dumps(config, indent=4) + "\n")
PY

    echo "${config_path}"
  }

  run_training() {
    local config_path="$1"
    echo "Training SSSD-ECG with ${config_path}"
    (
      cd "${sssd_src_dir}"
      python train.py -c "${config_path}"
    )
  }

  run_inference() {
    local config_path="$1"
    echo "Generating synthetic ECG with latest checkpoint from ${trained_dir}"
    (
      cd "${sssd_src_dir}"
      python3 - "${config_path}" <<'PY'
import json
import sys
from pathlib import Path

from utils.util import calc_diffusion_hyperparams
import inference as sssd_inference

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text())

sssd_inference.model_config = config["wavenet_config"]
sssd_inference.diffusion_config = config["diffusion_config"]
sssd_inference.diffusion_hyperparams = calc_diffusion_hyperparams(**config["diffusion_config"])

sssd_inference.generate(
    **config["gen_config"],
    ckpt_iter="max",
    num_samples=400,
)
PY
    )
  }

  collect_synthesis_outputs() {
    local config_path="$1"
    local ckpt_subdir
    ckpt_subdir="$(python3 - "${config_path}" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
model_config = config["wavenet_config"]
diffusion_config = config["diffusion_config"]
print(
    "ch{res}_T{T}_betaT{beta}".format(
        res=model_config["res_channels"],
        T=diffusion_config["T"],
        beta=diffusion_config["beta_T"],
    )
)
PY
)"

    local ckpt_dir="${trained_dir}/${ckpt_subdir}"
    mkdir -p "${synthesis_dir}"

    shopt -s nullglob
    local output_files=("${ckpt_dir}"/*_samples.npy "${ckpt_dir}"/*_labels.npy)
    shopt -u nullglob

    if [[ ${#output_files[@]} -eq 0 ]]; then
      echo "Error: No generated .npy files found in ${ckpt_dir}" >&2
      exit 1
    fi

    cp "${output_files[@]}" "${synthesis_dir}/"
    echo "Saved synthetic ECG outputs to ${synthesis_dir}"
  }

  local training_config
  download_figshare_ptbxl "${sssd_src_dir}"
  training_config="$(prepare_training_config)"
  run_training "${training_config}"
  run_inference "${training_config}"
  collect_synthesis_outputs "${training_config}"
}

case "${MODEL}" in
  sssd-ecg)
    generate_sssd_ecg
    ;;
  *)
    echo "Unknown model: ${MODEL}" >&2
    usage
    ;;
esac
