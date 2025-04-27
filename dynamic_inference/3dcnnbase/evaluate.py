import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from dynamic_inference.3dcnnbase.data_loader import memory_data_generator, load_preprocessed_frames
from config.dynamic_3dcnnbase_config import *

def plot_confusion_matrix(generator, model, classes):
    # Initialize lists for true and predicted labels
    y_true = []
    y_pred = []

    # Loop over the generator
    for x_batch, y_batch in generator:
        # Append true labels
        y_true.extend(np.argmax(y_batch, axis=1))  # Convert one-hot labels to integer

        # Predict the batch
        y_pred_batch = model.predict(x_batch, verbose=0)
        y_pred.extend(np.argmax(y_pred_batch, axis=1))

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title('Confusion Matrix')
    plt.show()

def evaluate_model():
    # Load preprocessed validation data
    print("Loading validation data...")
    val_videos, val_labels = load_preprocessed_frames(val_dir, classes)

    # Load the trained model
    print("Loading trained model...")
    model = load_model(checkpoint_path)

    # Evaluate on validation data
    print("Evaluating the model...")
    val_generator = memory_data_generator(val_videos, val_labels, BATCH_SIZE, len(classes))
    val_steps = len(val_videos) // BATCH_SIZE
    val_loss, val_accuracy = model.evaluate(val_generator, steps=val_steps, verbose=1)

    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")

    # Generate confusion matrix
    print("Generating confusion matrix...")
    plot_confusion_matrix(val_generator, model, classes)

if __name__ == "__main__":
    evaluate_model()
