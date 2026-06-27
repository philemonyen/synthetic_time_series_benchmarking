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
|--data/                    # Store the open source datasets. Not tracked by git
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
|--load_data.sh             # Download specified dataset to data/
|--load_model.sh            # Download specified model to model/
|--generate.sh              # Training & Generation script
|--evaluate.sh              # Evaluation script
|--sync.sh                  # Local-remote synchronization script
|
|--.gitignore
|--readme.md
```

# Execution Procedure

## Synthesis Generation
### Step 1: Download Dataset & Generation Model 
Since remote server has no outward internet access, all downloads must be done locally first. \
To download a dataset, run
```
./load_dataset.sh <dataset>
```
To download a model, run
```
./load_model.sh <model>
```
If no arguments provided, the scrips will print available datasets or models. \
#### Current Progress
For SSSD-ECG generation, only run ```./load_model.sh sssd-ecg```. 

### Step 2: Sync Local Setup with Remote Server
To sync local setup with remote server, run
```
./sync.sh local-to-remote <destination>
```
```<destination>``` is the full path (```userid@remote-server:path_to_dest```) to the target location in remote server. 

### Step 3: Synthesis Generation on Remote Server
To training the model and generate synthetic data with it, SSH to the remote server and run
```
./generate.sh <model>
```
```generate.sh``` will handle python dependency collection, virtual environment creation, model training, and synthesis generation. The generated synthesis will be stored in ```synthesis/```

### Step 4: Acquire Generated Synthesis from Remote Server
To acquire the generated synthetic data from remote server to local, run 
```
./sync.sh remote-to-local <path_to_synthesis_dir>
```
```path_to_synthesis_dir``` is the full path (```userid@remote-server:path_to_synthesis_dir```) to the ```synthesis/``` directory
## Synthesis Evaluation - Developing...