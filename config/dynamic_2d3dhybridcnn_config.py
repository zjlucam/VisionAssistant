import os

# Dataset Directories
base_data_dir = './processed_videos'
train_dir = os.path.join(base_data_dir, 'train')
val_dir = os.path.join(base_data_dir, 'val')
test_dir = os.path.join(base_data_dir, 'test')

# Classes
classes = ['Dissolved', 'Undissolved', 'Gel', 'Swelling', 'Dispersion']

# Model Parameters
FRAME_SIZE = (224, 224)      # Dimensions of each video frame
FRAMES_PER_VIDEO = 15        # Number of frames per video
BATCH_SIZE = 16              # Batch size
EPOCHS = 150                 # Number of epochs
LEARNING_RATE = 0.001        # Learning rate

# Checkpoints
checkpoint_path = './checkpoints/dynamic_2d3dhybridcnn.keras'
weights_checkpoint_path = './checkpoints/dynamic_2d3dhybridcnn.weights.h5'

# Parallel Processing Parameters
MAX_WORKERS = 8  # Maximum threads for data processing
