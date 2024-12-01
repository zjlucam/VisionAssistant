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
