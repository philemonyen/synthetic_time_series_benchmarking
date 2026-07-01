# Synthetic Time-Series Data Benchmarking
This repo implements the unified synthetic time-series data benchmarking framework, covering dataset downloading, generation model installation, and synthesis evaluation. The current progress is denoted as below. 
### Available Datasets
✅ PTB-XL \
More datasets to come...
### Available Models
✅ SSSD-ECG \
More models to come...
### Evaluation - Developing...

# File Structure
```text
synthetic_time_series_benchmarking/
|--model/                   # Store the generation model source codes. Not tracked by git
   |--model1/
      |--config.json        # Model configuration
      |--requirements.txt   # Evironment dependencies
      |--src/               # model source code
|
|--preprocess/              # Data preprocessing methods
|--evaluation/              # Evaluation methods
|--synthesis/               # Generated data stored in .npy format. Not tracked by git
|--results/                 # Evaluation results. Not tracked by git. Not tracked by git
|
|--generate_scripts/        # Model-specific training and generation scripts
|   |--generate_sssdecg.sh
|--relocate_scripts/        # Model-specific dataset relocation scripts
|   |--relocate_sssdecg.sh
|
|--load_model.sh            # Download specified model to model/
|--evaluate.sh              # Evaluation script
|--sync.sh                  # Local-remote synchronization script
|--job.sh                   # The remote server job script
|
|--.gitignore
|--readme.md
```

# Execution Procedure

## Synthesis Generation
### Step 1: Download Dataset
Since remote server has no outward internet connection, dataset downloads must first be done locally. Dataset downloads will be done manually. 
#### SSSD-ECG
Pre-processed dataset can be downloaded from https://figshare.com/s/43df16e4a50e4dd0a0c5?file=38890965

Expected layout after extraction:
```text
Dataset/
├── data/
│   ├── ptbxl_train_data.npy
│   ├── ptbxl_validation_data.npy
│   └── ptbxl_test_data.npy
└── labels/
    ├── ptbxl_train_labels.npy
    ├── ptbxl_validation_labels.npy
    └── ptbxl_test_labels.npy
```

### Step 2: Download Model
Model downloading also needs to be done locally. To download a model, run
```
./load_model.sh <model>
```
If no arguments are provided, the script will print available models. 

### Step 3: Relocate Dataset for Model Usage
Place the extracted dataset under `Dataset/` at the project root, then run
```
./relocate_scripts/relocate_<model>>.sh
```

### Step 4: Sync Local Setup with Remote Server
To sync local setup with remote server, run
```
./sync.sh local-to-remote <destination>
```
`<destination>` is the full path (`userid@remote-server:path_to_dest`) to the target location on the remote server. 

### Step 5: Synthesis Generation on Remote Server
To train the model and generate synthetic data with it, SSH to the remote server and submit the job via
```
sbatch ./job.sh <model>
```
`job.sh` will handle job details and computing resource allocation, so make sure to double-check before submitting a job. For SSSD-ECG, it runs `./generate_scripts/generate_sssdecg.sh`. The generated synthesis will be stored in `synthesis/SSSD-ECG/{date}` where `date` is the execution timestamp. 

**Important: Relocate to the root directory (where `job.sh` is located) before job submission to ensure relative paths will work as expected**

### Step 6: Acquire Generated Synthesis from Remote Server
To acquire the generated synthetic data from remote server to local, run 
```
./sync.sh remote-to-local <path_to_synthesis_dir>
```
`<path_to_synthesis_dir>` is the full path (`userid@remote-server:path_to_synthesis_dir`) to the `synthesis/` directory
## Synthesis Evaluation - Developing...
