import random
import numpy as np
import tensorflow as tf
from dynamic_inference.3dcnnbase.train import main as train_main
from dynamic_inference.3dcnnbase.data_processing import save_video_frames_parallel
from config.dynamic_3dcnnbase_config import base_data_dir, classes, SEED

def preprocess_videos():
    save_video_frames_parallel(
        data_dir=base_data_dir,
        save_dir=base_data_dir,
        classes=classes,
        frame_size=(224, 224),
        num_frames=15,
        max_workers=4
    )

if __name__ == "__main__":
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    preprocess_videos()
    train_main()
