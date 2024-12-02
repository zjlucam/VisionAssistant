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

## Dynamic Inference Module: 3D CNN and Hybrid 2D/3D CNN

The Dynamic Inference Module applies video-based deep learning models to predict solvation states in polymer-solvent systems. This module includes:
1. **3D CNN Base Model**: A lightweight spatiotemporal model with ~4.8M trainable parameters.
2. **Hybrid 2D/3D CNN Model**: Combines 2D CNNs for spatial features and 3D CNNs for temporal features (~4.8M parameters).
3. **Benchmark Models**:
   - **R3D**: ResNet-3D with 33.2M trainable parameters.
   - **C3D**: Convolutional 3D with 78.1M trainable parameters.

### Results
| Model           | Trainable Parameters (M) | Validation Accuracy (%) | Test Accuracy (%) |
|-----------------|--------------------------|-------------------------|-------------------|
| **3D CNN Base** | ~4.8                    | XX.X                   | XX.X              |
| **Hybrid 2D/3D**| ~4.8                    | XX.X                   | XX.X              |
| **R3D**         | ~33.2                   | XX.X                   | XX.X              |
| **C3D**         | ~78.1                   | XX.X                   | XX.X              |

> **Note**: Benchmark models (R3D and C3D) were fine-tuned using standard libraries. For implementation, refer to:
> - [R3D Documentation](https://pytorch.org/vision/stable/models.html#video-classification)
> - [C3D GitHub](https://github.com/DavideA/c3d-pytorch)

### Steps to Run
1. **Preprocess Videos**: Convert raw videos into normalized frames:
   ```bash
   python dynamic_inference/data_processing.py

# Contextualisation Module: BLIP-22DFE

The Contextualisation Module integrates the BLIP-2 model with 2D CNN features to enhance the interpretation of polymer-solvent solvation behaviors.

## Instructions to Download and Use BLIP-2

### Step 1: Install Dependencies
Ensure you have the required Python packages installed. Use the following command:
```bash
pip install torch torchvision transformers numpy pandas tqdm nltk rouge-score

from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Load BLIP-2 processor and model
processor = Blip2Processor.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    revision="51572668da0eb669e01a189dc22abe6088589a24"
)

model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    revision="51572668da0eb669e01a189dc22abe6088589a24"
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("BLIP-2 Model and Processor successfully loaded.")
