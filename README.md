# VisionAssistant
Parameter Efficient Multi-Model Vision Assistant for Polymer Solvation Behaviour Inference

x
x
x
x
x
x
xx
x

## Benchmarking Against Pretrained 2D CNNs

To evaluate the performance of the custom 2D CNN model, we benchmarked it against popular pretrained architectures: **DenseNet**, **VGG16**, **InceptionV3**, and **ResNet**. These models were fine-tuned on the same dataset under identical conditions.

### Benchmark Models
1. **DenseNet**:
   - **Trainable Parameters**: ~7.2 million
   - **Pretrained Weights**: [DenseNet Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/applications/DenseNet121)
   - **Paper**: [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)

2. **VGG16**:
   - **Trainable Parameters**: ~14.8 million
   - **Pretrained Weights**: [VGG Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16)
   - **Paper**: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

3. **InceptionV3**:
   - **Trainable Parameters**: ~22.4 million
   - **Pretrained Weights**: [InceptionV3 Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3)
   - **Paper**: [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)

4. **ResNet**:
   - **Trainable Parameters**: ~23.6 million
   - **Pretrained Weights**: [ResNet Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50)
   - **Paper**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

### Fine-Tuning Methodology
- **Dataset**: The same dataset used for training the custom 2D CNN was used for fine-tuning these models.
- **Pretrained Weights**: All models were initialized with ImageNet pretrained weights.
- **Task-Specific Layers**:
  - The fully connected layers of each model were replaced with a task-specific classification head for the five solvation behavior classes: `dispersion`, `undissolved`, `dissolved`, `gel`, and `swelling`.
- **Hyperparameters**:
  - Optimizer: Adam
  - Learning Rate: 0.001
  - Batch Size: 32
  - Epochs: 50
- **Frameworks Used**:
  - TensorFlow/Keras: [Installation Guide](https://www.tensorflow.org/install)

### Results
| Model          | Trainable Parameters (M) | Validation Accuracy (%) | Test Accuracy (%) |
|----------------|--------------------------|-------------------------|-------------------|
| **Base 2D CNN** | ~2.8                    | 70.1                    | 64.9              |
| **DenseNet**    | ~7.2                    | 82.3                    | 80.6              |
| **VGG16**       | ~14.8                   | 97.5                    | 95.9              |
| **InceptionV3** | ~22.4                   | 97.1                    | 95.7              |
| **ResNet**      | ~23.6                   | 97.4                    | 96.5              |

> **Note**: Fine-tuning scripts for these benchmark models are not included in this repository as they are widely available and well-documented. Refer to the provided links for details on implementing and fine-tuning these models.
