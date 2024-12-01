from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
import numpy as np

def evaluate_model(model, test_generator):
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    test_predictions = model.predict(test_generator)
    test_predicted_classes = np.argmax(test_predictions, axis=1)
    test_true_classes = test_generator.classes

    conf_matrix = confusion_matrix(test_true_classes, test_predicted_classes)
    print("Confusion Matrix:")
    print(conf_matrix)

    f1 = f1_score(test_true_classes, test_predicted_classes, average='weighted')
    recall = recall_score(test_true_classes, test_predicted_classes, average='weighted')
    precision = precision_score(test_true_classes, test_predicted_classes, average='weighted')

    print(f'F1 Score: {f1:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Precision: {precision:.4f}')
