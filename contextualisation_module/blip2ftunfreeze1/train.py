from VisionAssistant.config.blip2ftunfreeze1_config import *
from VisionAssistant.models.dataset import ImageCaptioningDataset, custom_collate
from VisionAssistant.models.model_utils import load_model_and_processor
from VisionAssistant.models.training import train_epoch, validate_epoch

# Load model and processor
model, processor = load_model_and_processor(saved_model_path, saved_processor_path)

# Prepare datasets and dataloaders
train_dataset = ImageCaptioningDataset(train_df, processor, TRAIN_DIR)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train the model
for epoch in range(10):
    train_loss = train_epoch(model, train_dataloader, optimizer, device)
    print(f"Epoch {epoch + 1} Train Loss: {train_loss:.4f}")
