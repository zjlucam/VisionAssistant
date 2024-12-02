import os

# Base directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__) + "/..")
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# File paths
IMG_DIR = os.path.join(DATA_DIR, "imagesblipresized")
FEATURE_DIR = os.path.join(DATA_DIR, "npyfilesforblip")
DATASET_PATH = os.path.join(DATA_DIR, "captions_dataset.xlsx")
MODEL_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "2dfeatureblip.pth")
