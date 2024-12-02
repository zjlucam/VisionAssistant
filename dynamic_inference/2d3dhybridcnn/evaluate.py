from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from dynamic_inference.2d3dhybridcnn.data_loader import memory_data_generator, load_and_split_frames
from config.dynamic_2d3dhybridcnn_config import *

def evaluate_model():
    # Load data
    _, _, _, _, test_videos, test_labels = load_and_split_frames(data_dir, classes)
    test_generator = memory_data_generator(test_videos, test_labels, BATCH_SIZE, len(classes))

    # Load model
    model = load_model(checkpoint_path)

    # Evaluate on test data
    test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_videos) // BATCH_SIZE, verbose=1)
    print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

    # Confusion matrix
    y_pred = np.argmax(model.predict(test_videos, verbose=1), axis=1)
    cm = confusion_matrix(test_labels, y_pred, labels=range(len(classes)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
