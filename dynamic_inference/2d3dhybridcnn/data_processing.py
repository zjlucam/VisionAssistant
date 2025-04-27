import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def load_class_files(class_dir, label):
    videos = []
    labels = []

    for file_name in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file_name)
        frames = np.load(file_path)  # Load pre-saved frames as numpy array
        videos.append(frames)
        labels.append(label)

    return videos, labels
