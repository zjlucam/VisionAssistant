import os
import numpy as np
from sklearn.model_selection import train_test_split
from dynamic_inference.2d3dhybridcnn.data_processing import load_class_files

def load_and_split_frames(data_dir, classes, test_size=0.2, random_state=42, max_workers=4):
    videos = []
    labels = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(load_class_files, os.path.join(data_dir, class_name), label)
            for label, class_name in enumerate(classes)
        ]
        for future in futures:
            class_videos, class_labels = future.result()
            videos.extend(class_videos)
            labels.extend(class_labels)

    videos, labels = np.array(videos), np.array(labels)

    train_videos, temp_videos, train_labels, temp_labels = train_test_split(
        videos, labels, test_size=0.3, random_state=random_state, stratify=labels
    )
    val_videos, test_videos, val_labels, test_labels = train_test_split(
        temp_videos, temp_labels, test_size=0.5, random_state=random_state, stratify=temp_labels
    )

    return train_videos, train_labels, val_videos, val_labels, test_videos, test_labels

def memory_data_generator(videos, labels, batch_size, num_classes):
    num_samples = videos.shape[0]
    indices = np.arange(num_samples)
    while True:
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            yield videos[batch_indices], tf.keras.utils.to_categorical(labels[batch_indices], num_classes)
