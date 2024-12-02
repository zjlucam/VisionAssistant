from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import numpy as np
from tqdm import tqdm

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataframe, processor, img_dir, feature_dir, device, max_length=77):
        self.processor = processor
        self.img_dir = img_dir
        self.feature_dir = feature_dir
        self.max_length = max_length
        self.device = device

        def process_row(row):
            image_path = os.path.join(self.img_dir, f"{row['image_path']}.jpg")
            feature_path = os.path.join(self.feature_dir, f"{row['image_path']}.npy")

            if os.path.exists(image_path) and os.path.exists(feature_path):
                return image_path, feature_path, row['caption']
            return None

        self.data = [process_row(row) for row in tqdm(dataframe.to_dict(orient="records")) if process_row(row)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, feature_path, caption = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        features = torch.tensor(np.load(feature_path), dtype=torch.float32).to(self.device)
        inputs = self.processor(images=image, text=caption, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["cnn_features"] = features
        return inputs


def custom_collate_with_cnn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    collated = {k: torch.stack([d[k] for d in batch]) for k in batch[0] if k != "cnn_features"}
    collated["cnn_features"] = torch.stack([d["cnn_features"] for d in batch])
    return collated
