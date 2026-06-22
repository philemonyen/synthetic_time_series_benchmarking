# Synthetic Time-Series Data Benchmarking
## File Structure
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
|--synthetic_data/          # Generated data stored in .npy format. Not tracked by git
|--results/                 # Evaluation results. Not tracked by git. Not tracked by git
|
|--load_data.sh             # Download specified dataset to data/
|--load_model.sh            # Download specified model to model/
|--train.sh                 # Training script
|--generate.sh              # Generation script
|--evaluate.sh              # Evaluation script
|--.gitignore
|--readme.md
```
