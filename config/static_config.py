import os

# User-defined or default dataset directory
dataset_base_dir = os.getenv('DATASET_DIR', './dataset')

# Subdirectories for train, validation, and test splits
train_directory = os.path.join(dataset_base_dir, 'train')
val_directory = os.path.join(dataset_base_dir, 'val')
test_directory = os.path.join(dataset_base_dir, 'test')

# Parameters
seed = 17
batch_size = 32
target_size = (224, 224)
class_names = ['dispersion', 'undissolved', 'dissolved', 'gel', 'swelling']
class_indices = {name: i for i, name in enumerate(class_names)}

# Checkpoint path
checkpoint_path = './checkpoints/base2dcnnmodel.keras'
