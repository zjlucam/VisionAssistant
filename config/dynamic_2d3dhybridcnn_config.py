import os

# Dataset Directories
base_data_dir = './processed_videos'
train_dir = os.path.join(base_data_dir, 'train')
val_dir = os.path.join(base_data_dir, 'val')
test_dir = os.path.join(base_data_dir, 'test')

# Classes
classes = ['Dissolved', 'Undissolved', 'Gel', 'Swelling', 'Dispersion']

# Model Parameters
FRAME_SIZE = (224, 224)
FRAMES_PER_VIDEO = 15   
BATCH_SIZE = 16             
EPOCHS = 150                 
LEARNING_RATE = 0.001        

# Checkpoints
checkpoint_path = './checkpoints/dynamic_2d3dhybridcnn.keras'
weights_checkpoint_path = './checkpoints/dynamic_2d3dhybridcnn.weights.h5'

# Parallel Processing Parameters
MAX_WORKERS = 8  

# Reproducibility
SEED = 17
