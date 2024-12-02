import os
from PIL import Image
from torch.utils.data import Dataset
from joblib import Parallel, delayed

class ImageCaptioningDataset(Dataset):
    def __init__(self, dataframe, processor, image_dir, max_length=77, target_size=(224, 224)):
        self.processor = processor
        self.max_length = max_length
        self.target_size = target_size
        self.image_dir = image_dir

        # Preload images and captions
        def process_row(row):
            image_filename = f"{row['image_path']}.jpg"
            image_path = os.path.join(self.image_dir, image_filename)
            try:
                image = Image.open(image_path).convert('RGB').resize(self.target_size)
                return image, row['caption']
            except FileNotFoundError:
                print(f"Image not found: {image_path}")
                return None

        self.image_caption_pairs = Parallel(n_jobs=-1)(
            delayed(process_row)(row) for row in dataframe.to_dict(orient="records")
        )
        self.image_caption_pairs = [pair for pair in self.image_caption_pairs if pair is not None]

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self, idx):
        image, caption = self.image_caption_pairs[idx]
        inputs = self.processor(
            images=image, text=caption, return_tensors="pt", padding="max_length",
            truncation=True, max_length=self.max_length
        )
        return {k: v.squeeze(0) for k, v in inputs.items()}


def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return {k: torch.stack([d[k] for d in batch]) for k in batch[0]}
