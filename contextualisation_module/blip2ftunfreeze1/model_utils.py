from transformers import Blip2Processor, Blip2ForConditionalGeneration

def save_model_and_processor(model, processor, model_dir, processor_dir):
    model.save_pretrained(model_dir)
    processor.save_pretrained(processor_dir)

def load_model_and_processor(model_dir, processor_dir):
    processor = Blip2Processor.from_pretrained(processor_dir)
    model = Blip2ForConditionalGeneration.from_pretrained(model_dir)
    return model, processor
