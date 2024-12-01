import os
from config.static_2dcnnbase_config.py import *
from static_inference.data_loader import get_data_generators
from static_inference.model import build_model
from static_inference.train import train_model
from static_inference.evaluate import evaluate_model

def main():
    # Check if dataset directory exists
    if not os.path.exists(dataset_base_dir):
        raise FileNotFoundError(
            f"The dataset directory '{dataset_base_dir}' does not exist. "
            "Please download the dataset and update the path in config/static_config.py or set the DATASET_DIR environment variable."
        )

    # Load data
    train_generator, val_generator, test_generator = get_data_generators(
        train_directory, val_directory, test_directory, target_size, batch_size, seed
    )

    # Build model
    model = build_model(input_shape=(224, 224, 3), num_classes=len(class_names))

    # Train model
    train_model(
        model,
        train_generator,
        val_generator,
        checkpoint_path=checkpoint_path,
        epochs=50,
    )

    # Evaluate model
    evaluate_model(model, test_generator)

if __name__ == "__main__":
    main()
