import os

# Dataset Directories
base_data_dir = './processed_videos'
train_dir = os.path.join(base_data_dir, 'train')
val_dir = os.path.join(base_data_dir, 'val')

# Classes
classes = ['Dissolved', 'Undissolved', 'Gel', 'Swelling', 'Dispersion']

# Frame and Video Parameters
FRAME_SIZE = (224, 224)
FRAMES_PER_VIDEO = 15

# Training Parameters
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001
SEED = 17

# Learning Rate Scheduler Parameters
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.1
MIN_LR = 1e-6

# Early Stopping Parameters
EARLY_STOPPING_PATIENCE = 10

# Checkpoints
checkpoint_path = './checkpoints/base_3dcnn.keras'
weights_checkpoint_path = './checkpoints/base_3dcnn.weights.h5'
