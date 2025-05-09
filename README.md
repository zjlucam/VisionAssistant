# 📷 Vision Assistant Model Repository

This repository contains the codebase (framework) for dynamic and static inference using various deep learning models, including BLIP-2, 2D CNNs, 3D CNNs, and hybrid 2D-3D CNN architectures.

The dataset (images, videos, image-text label pairs) can be found here: https://zenodo.org/records/14319147

## 📁 Repository Structure
```
📦repo
 ┣ 📂config
 ┃ ┣ 📜blip22dfe_config.py
 ┃ ┣ 📜blip2ftunfreeze1_config.py
 ┃ ┣ 📜dynamic_2d3dhybridcnn_config.py
 ┃ ┣ 📜dynamic_3dcnnbase_config.py
 ┃ ┗ 📜static_2dcnnbase_config.py
 ┣ 📂contextualisation_module
 ┃ ┣ 📂blip22dfe
 ┃ ┃ ┣ 📜blip2_with_cnn.py
 ┃ ┃ ┣ 📜image_captioning_dataset.py
 ┃ ┃ ┣ 📜main.py
 ┃ ┃ ┣ 📜test_evaluation.py
 ┃ ┃ ┣ 📜train_loop.py
 ┃ ┃ ┗ 📜validate_loop.py
 ┃ ┗ 📂blip2ftunfreeze1
 ┃ ┃ ┣ 📜dataset.py
 ┃ ┃ ┣ 📜evaluation.py
 ┃ ┃ ┣ 📜model_utils.py
 ┃ ┃ ┣ 📜train.py
 ┃ ┃ ┣ 📜training.py
 ┃ ┃ ┗ 📜visualisation.py
 ┣ 📂dynamic_inference
 ┃ ┣ 📂2d3dhybridcnn
 ┃ ┃ ┣ 📜data_loader.py
 ┃ ┃ ┣ 📜data_processing.py
 ┃ ┃ ┣ 📜evaluate.py
 ┃ ┃ ┣ 📜model.py
 ┃ ┃ ┗ 📜train.py
 ┃ ┗ 📂3dcnnbase
 ┃ ┃ ┣ 📜data_loader.py
 ┃ ┃ ┣ 📜data_processing.py
 ┃ ┃ ┣ 📜evaluate.py
 ┃ ┃ ┣ 📜main.py
 ┃ ┃ ┣ 📜model.py
 ┃ ┃ ┗ 📜train.py
 ┣ 📂static_inference
 ┃ ┣ 📜data_loader.py
 ┃ ┣ 📜evaluate.py
 ┃ ┣ 📜main.py
 ┃ ┣ 📜model.py
 ┃ ┗ 📜train.py
 ┣ 📜data_loader.py
 ┣ 📜evaluate.py
 ┣ 📜main.py
 ┣ 📜model.py
 ┣ 📜train.py
 ┣ 📜requirements.txt
 ┗ 📜README.md
```

# VisionAssistant
Parameter Efficient Multi-Model Vision Assistant for Polymer Solvation Behaviour Inference

- **Static Inference Module**: Trained with a **2D CNN** on static images to classify solvation states.  
- **Dynamic Inference Module**: Utilises a **3D CNN and Hybrid 2D/3D CNN** for video-based solvation state prediction.  
- **Contextualisation Module**: Integrates **BLIPv2 with 2D CNN features** to provide interpretable descriptions of solvation behaviours.  

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
- **Pretrained Weights**: All models were initialised with ImageNet pretrained weights.
- **Task-Specific Layers**:
  - The fully connected layers of each model were replaced with a task-specific classification head for the five solvation behavior classes: `dispersion`, `undissolved`, `dissolved`, `gel`, and `swelling`.
- **Hyperparameters**:
  - Optimiser: Adam
  - Learning Rate: 0.001
  - Learning Rate Scheduler: ReduceLROnPlateau (monitors validation loss and reduces the learning rate when it plateaus)
    - Factor: 0.1 (reduces LR by a factor of 10)
    - Patience: 5 epochs (waits 5 epochs with no improvement before reducing LR)
    - Minimum LR: 1e-6
  - Batch Size: 32
  - Epochs: 50
  - Early Stopping: Enabled
    - Patience: 10 epochs (training stops if validation loss doesn’t improve for 10 consecutive epochs)
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
| **3D CNN Base** | ~4.8                     | 73.8                    | 71.2              |
| **Hybrid 2D/3D**| ~4.8                     | 94.1                    | 93.7              |
| **R3D**         | ~33.2                    | 93.2                    | 93.7              |
| **C3D**         | ~78.1                    | 95.1                    | 94.9              |

> **Note**: Benchmark models (R3D and C3D) were fine-tuned using standard libraries. For implementation, refer to:
> - [R3D Documentation](https://pytorch.org/vision/stable/models.html#video-classification)
> - [C3D GitHub](https://github.com/DavideA/c3d-pytorch)

### Note: To reduce GPU RAM requirements: Preprocess videos by converting raw videos into normalised frames:
   ```bash
   python dynamic_inference/data_processing.py
```

# Contextualisation Module: Vision-Language Integration with 2D CNN Features

The Contextualisation Module integrates the BLIP-2 model with 2D CNN features to enhance the interpretation of polymer-solvent solvation behaviors.

The Contextualisation Module enhances the interpretability of polymer-solvent solvation behaviors by integrating BLIPv2 with extracted features from the 2D CNN model. This approach improves classification accuracy while maintaining a low number of trainable parameters.

Concatenation Strategy for Efficient Contextualisation
Instead of relying on BLIPv2 alone, the module concatenates features from a pre-trained 2D CNN with BLIPv2's vision-language representation.
This allows for better contextual accuracy in describing solvation states while keeping the number of trainable parameters low compared to fully fine-tuning BLIPv2.

## Instructions to Download and Use BLIP-2 

## Install Dependencies

Ensure you have the required Python packages installed. Run the following command:

```bash
pip install torch torchvision transformers numpy pandas tqdm nltk rouge-score

from transformers import Blip2Processor, Blip2ForConditionalGeneration

# For the purposes of the study, a certain hash state of the BLIP-2 model was used to avoid a tokenisation error.

# Load BLIP-2 processor and model with a fixed hash state
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
```

## Benchmarking Results for the Contextualisation Module

The table below summarises the performance of BLIPv2 configurations, highlighting the impact of different fine-tuning strategies and the concatenation of 2D CNN features on BLEU-4 and ROUGE scores.

| **Model Configuration** | **Trainable Parameters** | **BLEU-4 ↑** | **BLEU-4 Variance ↓** | **ROUGE ↑** | **ROUGE Variance ↓** |
|-------------------------|-------------------------|--------------|----------------------|-------------|----------------------|
| **BLIPv2 (Unfreeze FC1, FC2, and Final Norm Layer)** | **52M** | 0.564 | 0.07 | 0.83 | 0.057 |
| **BLIPv2 + 2D CNN Feature Concatenation (Same Unfrozen Layers)** | **52M** | **0.855** | **0.04** | **0.937** | **0.03** |
| **BLIPv2 (Unfreeze N-1 Layers)** | **80M** | 0.849 | 0.05 | 0.931 | 0.04 |
| **BLIPv2 (Unfreeze N-2 Layers)** | **160M** | **0.868** | **0.03** | **0.943** | **0.02** |

### Key Insights
- **2D CNN Feature Concatenation Improves Performance:**
  - At **52M trainable parameters**, **concatenating 2D CNN features significantly boosts BLEU-4 (+0.291) and ROUGE (+0.107) scores**, outperforming even the **80M and 160M** configurations.
- **Minimal Trade-off Between Model Complexity and Performance:**
  - **80M and 160M parameter models** slightly improve BLEU-4 and ROUGE but at a **higher computational cost**.
  - The **52M model with concatenation** achieves near **160M-level performance while maintaining efficiency**.

### Conclusion
The **BLIPv2 + 2D CNN Feature Concatenation (52M)** model **offers the best balance** between **efficiency and performance**, demonstrating that domain-specific feature integration can outperform brute-force parameter scaling.
