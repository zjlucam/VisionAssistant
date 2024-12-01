import os
import numpy as np
from sklearn.model_selection import train_test_split

def load_preprocessed_frames(data_dir, classes):
    videos, labels = [], []
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            videos.append(np.load(file_path))
            labels.append(label)
    return np.array(videos), np.array(labels)

def memory_data_generator(videos, labels, batch_size, num_classes):
    while True:
        indices = np.random.permutation(len(videos))
        for start in range(0, len(videos), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            yield videos[batch_indices], tf.keras.utils.to_categorical(labels[batch_indices], num_classes)
