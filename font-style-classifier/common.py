import os
from os import path
from os import environ as env
import torch

# Get current script directory
FSC_DIR = path.dirname(path.abspath(__file__))


# ------------------------------------------------------------------------------------------------
# Variables
# ------------------------------------------------------------------------------------------------

FSC_DATA_DIR = path.join(FSC_DIR, "data")

FSC_GOOGLE_FONTS_DIR = path.join(FSC_DATA_DIR, "google_fonts")

FSC_DATASET_DIR = path.join(FSC_DATA_DIR, "dataset")

FSC_DATASET_CSV = path.join(FSC_DATASET_DIR, "dataset.csv")
FSC_TRAINING_SET_CSV = path.join(FSC_DATASET_DIR, "training_set.csv")
FSC_VALIDATION_SET_CSV = path.join(FSC_DATASET_DIR, "validation_set.csv")
FSC_TEST_SET_CSV = path.join(FSC_DATASET_DIR, "test_set.csv")

FSC_ENCODER_MODEL = env.get("FSC_ENCODER_MODEL", "V2Net")
FSC_ENCODER_CHECKPOINT_DIR = path.join(FSC_DATA_DIR, f"encoder-{FSC_ENCODER_MODEL}-checkpoints")
FSC_ENCODER_LATEST_CHECKPOINT = path.join(FSC_ENCODER_CHECKPOINT_DIR, "latest.pt")

FSC_DB_PATH = path.join(FSC_DATA_DIR, "db")
FSC_DB_URL = "http://localhost:6333"
FSC_DB_COLLECTION_NAME="font-style-classifier"


# ------------------------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------

# Create ./data directory if doesn't eixst
if not path.exists(FSC_DATA_DIR):
    os.mkdir(FSC_DATA_DIR)


if torch.cuda.is_available():
    print("[Common] CUDA is available, setting default device to: \"cuda\"")
    torch.set_default_device('cuda')


