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
EPOCHS = 150
LEARNING_RATE = 0.001
SEED = 17  

# Checkpoints
checkpoint_path = './checkpoints/base_3dcnn.keras' 
weights_checkpoint_path = './checkpoints/base_3dcnn.weights.h5'
