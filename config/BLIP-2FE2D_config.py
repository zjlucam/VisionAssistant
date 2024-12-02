import os

# Base directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__) + "/..")
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
PROCESSOR_DIR = os.path.join(BASE_DIR, "saved_processors")

# Specific file paths
DATASET_PATH = os.path.join(DATA_DIR, "captions_dataset.xlsx")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
TRAIN_DIR = os.path.join(IMAGE_DIR, "train")
VAL_DIR = os.path.join(IMAGE_DIR, "val")
TEST_DIR = os.path.join(IMAGE_DIR, "test")
