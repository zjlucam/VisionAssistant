import os

# Dataset Directories
base_data_dir = './processed_videos'
train_dir = os.path.join(base_data_dir, 'train')
val_dir = os.path.join(base_data_dir, 'val')

# Classes
classes = ['Dissolved', 'Undissolved', 'Gel', 'Swelling', 'Dispersion']

# Frame and Video Parameters
FRAME_SIZE = (224, 224)  # Frame dimensions
FRAMES_PER_VIDEO = 15    # Number of frames per video clip

# Training Parameters
BATCH_SIZE = 16
EPOCHS = 150
LEARNING_RATE = 0.001

# Checkpoints
model_checkpoint_path = './checkpoints/base_3dcnn.keras'
weights_checkpoint_path = './checkpoints/base_3dcnn.weights.h5'
