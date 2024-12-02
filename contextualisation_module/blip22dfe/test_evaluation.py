from transformers import Blip2Processor
import torch

def evaluate_model(model, dataloader, processor, device):
    model.eval()
    hypotheses = []
    references = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cnn_features = batch["cnn_features"].to(device)
            outputs = model(pixel_values, input_ids, attention_mask, cnn_features)
            generated_ids = outputs.logits.argmax(dim=-1)
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            true_texts = processor.batch_decode(input_ids, skip_special_tokens=True)
            hypotheses.extend(generated_texts)
            references.extend(true_texts)
    return references, hypotheses

# Example metric calculation
from nltk.translate.bleu_score import corpus_bleu
def calculate_bleu(references, hypotheses):
    return corpus_bleu([[ref.split()] for ref in references], [hyp.split() for hyp in hypotheses])
