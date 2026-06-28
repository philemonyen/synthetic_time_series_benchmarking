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
|--load_model.sh            # Download specified model to model/
|--relocate_data.sh         # Relocate manual-download dataset to desired location
|--generate.sh              # Training & Generation script
|--evaluate.sh              # Evaluation script
|--sync.sh                  # Local-remote synchronization script
|--job.sh                   # THe remote server job script
|
|--.gitignore
|--readme.md
```

# Execution Procedure

## Synthesis Generation
### Step 1: Download Dataset
Since remot eserver has no outward internet connection, dataset downloads must first be done locally. Dataset donwloads will be done manually. 
#### SSSD-ECG
Pre-processed dataset can be downloaded from https://figshare.com/s/43df16e4a50e4dd0a0c5?file=38890965

### Step 2: Download Model
Model downloading also need to be done locally. To download a model, run
```
./load_model.sh <model>
```
If no arguments provided, the script will print available models. 

### Step 3: Relocate Dataset for Model Usage
To relocate the downloaded dataset to the desired location where models can access for training and inference, run
```
./relocate_data.sh <model> <path_to_dataset>
```
If no argument provided, the script will print relocation instructions for each available model.

### Step 4: Sync Local Setup with Remote Server
To sync local setup with remote server, run
```
./sync.sh local-to-remote <destination>
```
```<destination>``` is the full path (```userid@remote-server:path_to_dest```) to the target location in remote server. 

### Step 5: Synthesis Generation on Remote Server
To training the model and generate synthetic data with it, SSH to the remote server and submit the job via
```
sbatch ./job.sh <model>
```
```job.sh``` will handle job details and computing resource allocation, so make sure to double check before submitting a job. The generated synthesis will be stored in ```synthesis/{model}/{date}``` where ```model``` is the generation model and ```date``` is the execution timestamp. 

**Important: Relocate to the directory where ```job.sh``` is located before job submission**

### Step 6: Acquire Generated Synthesis from Remote Server
To acquire the generated synthetic data from remote server to local, run 
```
./sync.sh remote-to-local <path_to_synthesis_dir>
```
```path_to_synthesis_dir``` is the full path (```userid@remote-server:path_to_synthesis_dir```) to the ```synthesis/``` directory
## Synthesis Evaluation - Developing...