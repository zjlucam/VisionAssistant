import torch
from torch.utils.data import DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from blip22dfe_config import *
from datasets.image_captioning_dataset import ImageCaptioningDataset, custom_collate_with_cnn
from models.blip2_with_cnn import Blip2WithCNNFeatures
from training.train_loop import train_epoch
from training.validate_loop import validate_epoch
from evaluation.test_evaluation import evaluate_model, calculate_bleu

# Load processor and base model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
base_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

# Initialize BLIP model with CNN
cnn_feature_dim = 86528
model = Blip2WithCNNFeatures(base_model, cnn_feature_dim)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load dataset
df = pd.read_excel(DATASET_PATH)
train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.1765, random_state=42)
train_dataset = ImageCaptioningDataset(train_df, processor, IMG_DIR, FEATURE_DIR, device)
val_dataset = ImageCaptioningDataset(val_df, processor, IMG_DIR, FEATURE_DIR, device)

# Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate_with_cnn)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_with_cnn)

# Optimizer
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = train_epoch(model, train_dataloader, optimizer, device)
    print(f"Training Loss: {train_loss:.4f}")
    val_loss = validate_epoch(model, val_dataloader, device)
    print(f"Validation Loss: {val_loss:.4f}")
