"""Locations of the preprocessed MIMIC-IV files (used by test_mimic_utility.py --real and validate_pipeline.py).
Edit DATA to the folder that holds them, locally or on the cluster."""
from pathlib import Path

DATA = Path("data/mimic")          # folder with the preprocessed files

PATHS = dict(train=str(DATA / "train.npy"), test=str(DATA / "test.npy"),
             stays_train=str(DATA / "stays_train.npy"), stays_test=str(DATA / "stays_test.npy"),
             subjects_train=str(DATA / "subjects_train.npy"), subjects_test=str(DATA / "subjects_test.npy"),
             annotations=str(DATA / "annotations.json"), channels=str(DATA / "channels.json"))
