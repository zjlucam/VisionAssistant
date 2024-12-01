import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def process_video(video_path, save_path, frame_size=(224, 224), num_frames=15):
    if os.path.exists(save_path):
        return  # Skip if already processed

    frames = []
    cap = cv2.VideoCapture(video_path)
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)
        frames.append(frame)
    cap.release()

    # Pad or truncate to correct frame count
    frames = frames[:num_frames] + [frames[-1]] * (num_frames - len(frames))

    np.save(save_path, np.array(frames))

def save_video_frames_parallel(data_dir, save_dir, classes, frame_size=(224, 224), num_frames=15, max_workers=4):
    os.makedirs(save_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for class_name in classes:
            class_dir = os.path.join(data_dir, class_name)
            save_class_dir = os.path.join(save_dir, class_name)
            os.makedirs(save_class_dir, exist_ok=True)

            for video_name in os.listdir(class_dir):
                video_path = os.path.join(class_dir, video_name)
                save_path = os.path.join(save_class_dir, f'{os.path.splitext(video_name)[0]}.npy')
                executor.submit(process_video, video_path, save_path, frame_size, num_frames)
