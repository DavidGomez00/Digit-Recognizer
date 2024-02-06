import torch

# Data settings
CSV_PATH="/home/master/GitHub/Digit-Recognizer/data/train.csv"
BATCH_SIZE=256
NUM_WORKERS=7
VAL_SIZE=0.2
SPLIT_SEED=42

# Train settings
LEARNING_RATE=0.01
ACCELERATOR="gpu" if torch.cuda.is_available() else "cpu"
DEVICES = [0]
MIN_EPOCHS = 1
MAX_EPOCHS = 100
PRECISION = "bf16-mixed"