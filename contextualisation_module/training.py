import torch
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        if batch is None:
            continue
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            if batch is None:
                continue
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()

    return total_loss / len(dataloader)
