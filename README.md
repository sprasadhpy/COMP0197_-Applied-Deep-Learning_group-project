# COMP0197_-Applied-Deep-Learning_group-project
Self-supervised learning - group project

Instructions to run are in instructions.txt

# Main Architecture - VQ-VAE


This code is a PyTorch implementation of the Vector Quantized Variational Autoencoder (VQ-VAE) model. It is ported from the [official implementation](https://github.com/deepmind/sonnet/blob/v2/examples/vqvae_example.ipynb) provided by DeepMind in their Sonnet library.

### Model Architecture

The VQ-VAE consists of three main components: Encoder, Vector Quantizer, and Decoder.

#### Encoder

The Encoder module is responsible for encoding input images into a lower-dimensional feature space. It contains convolutional layers followed by residual stacks.

#### Vector Quantizer

The Vector Quantizer module quantizes the continuous latent space into a discrete space represented by a fixed set of embeddings. It involves comparing the input embeddings with the learned dictionary embeddings and assigning the closest dictionary embedding to each input.

#### Decoder

The Decoder module reconstructs the input image from the quantized latent space representation. It consists of convolutional layers followed by residual stacks and upsampling layers.

### Usage

To use the VQ-VAE model, instantiate the `VQVAE` class with the desired parameters. You can then train the model using your dataset and evaluate its performance.

### Files

- `vqvae.py`: Contains the implementation of the VQ-VAE model.
- `train.py`: Includes code for training the VQ-VAE model.
- `evaluate.py`: Provides functions for evaluating the performance of the trained model.

### Requirements

- Python >= 3.6
- PyTorch >= 1.7
- tqdm
- matplotlib

### Citation

If you find this code useful in your research, please consider citing the original DeepMind paper:

@article{vqvae,
title = {Neural Discrete Representation Learning},
author = {van den Oord, A{"a}ron and Vinyals, Oriol and Kavukcuoglu, Koray},
journal = {Advances in Neural Information Processing Systems},
year = {2017}
}

# Results - VQ-VAE Model Evaluation on Oxford-IIIT Pet Dataset

## Evaluation Metrics

| Metric      | Result  |
|-------------|---------|
| Dice Score  | 0.8517  |
| Accuracy    | 0.8986  |
| IoU Score   | 0.7520  |

## Inference

The VQ-VAE model exhibits strong performance on the Oxford-IIIT Pet Dataset, with high accuracy and robustness in image segmentation as indicated by the Dice and IoU scores.

--------------------------------------


# Benchmark Architecture - LinkNet

LinkNet is a convolutional neural network architecture designed for semantic segmentation tasks. It utilizes a ResNet model as the encoder and decoder blocks for feature extraction and upscaling. This README provides an overview of the LinkNet implementation in PyTorch.

## Requirements

- Python 3.x
- PyTorch
- torchvision

## Usage

To use LinkNet in your project, follow these steps:

1. **Import the necessary modules:**
    ```python
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    from torchmetrics.functional import dice
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from linknet import link_net
    import os
    ```

2. **Define the decoder block:**
    ```python
    class decoder_block(nn.Module):
        def __init__(self, in_size, out_size):
            ...
    ```

3. **Define the LinkNet model:**
    ```python
    class link_net(nn.Module):
        def __init__(self, classes, encoder='resnet34'):
            ...
    ```

4. **Training and Evaluation:**
    ```python
    # Train the model
    trained_model = train_segmentation_model(train_loader, val_loader, device, num_epochs=100, lr=1e-5)

    # Evaluate the model
    dice_score, iou_score = evaluate_model(trained_model, test_loader, device)
    ```

## Implementation Details

- **Decoder Block:** Implements a decoder block for upscaling and feature processing.
- **LinkNet Model:** Utilizes a ResNet encoder followed by decoder blocks with skip connections for better feature localization.
- **Pretrained Weights:** The ResNet encoder is initialized with pretrained weights.

# Input and Output Feature Maps

The table below illustrates the dimensions of input and output feature maps at different stages of the encoder and decoder blocks.

| Block | Encoder (m x n) | Decoder (m x n) |
|-------|-----------------|-----------------|
| 1.    | 64 x 64         | 64 x 64         |
| 2.    | 64 x 128        | 128 x 64        |
| 3.    | 128 x 256       | 256 x 128       |
| 4.    | 256 x 512       | 512 x 256       |

These dimensions provide insight into the transformation of feature maps as they pass through the encoder and decoder blocks of the model.



## Example

```python
# Import necessary modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional import dice
from torch.optim.lr_scheduler import ReduceLROnPlateau
from linknet import link_net
import os

# Define the LinkNet model
class link_net(nn.Module):
    def __init__(self, classes, encoder='resnet34'):
        ...
        # Initialize the ResNet encoder with pretrained weights
        res = resnet.resnet34(pretrained=True)
        ...
    
    def forward(self, x):
        ...
        return x

# Train the model
trained_model = train_segmentation_model(train_loader, val_loader, device, num_epochs=100, lr=1e-5)

# Evaluate the model
dice_score, iou_score = evaluate_model(trained_model, test_loader, device)

# Model Performance on Validation and Test Sets

The table below presents the performance metrics of the model on both the validation and test sets.

| Metric                               | Validation Set | Test Set |
|--------------------------------------|----------------|----------|
| Loss                                 | 0.288          | N/A      |
| Dice Score                           | 0.858          | 0.857    |
| Intersection over Union (IoU)        | 0.770          | 0.768    |

## Interpretation

- **Loss:** The average loss on the validation set is 0.288.
- **Dice Score:** The Dice score, a measure of overlap between predicted and ground truth masks, is 0.858 on the validation set and 0.857 on the test set.
- **Intersection over Union (IoU):** The IoU score, which measures the overlap of predicted and ground truth masks normalized by their union, is 0.770 on the validation set and 0.768 on the test set.
```
# Experiments

#### 1)Class Imbalance Experiment

The Default mode that includes pre-training and fine-tuning shows robust scores across the board with the lowest training loss, high Train and Test Dice scores, and the highest Test Accuracy, indicating a well-optimized model to predict. Other training modes such as including training with half the dataset and also, fully supervised without pretraining show higher losses and slightly lower scores, suggesting some impact on the model's ability to generalize. However, our experiment on the Class Imbalance mode, designed to tackle biased datasets, performs admirably well. Though with a slight compromise in Test IOU and Accuracy, which is a common challenge in such dataset scenarios. Overall, the consistency in high Dice and IOU scores across all training modes reflects the model's strong segmentation ability.

## VAE Results: Training Performance

| Mode                                               | Training Loss | Train Dice Score |
|----------------------------------------------------|---------------|------------------|
| Default (Pre-training & Fine Tuning)               | 0.61          | 0.945            |
| Pre-train with half dataset Finetuning default     | 0.6182        | 0.937            |
| Fully Supervised (No Pretraining)                  | 0.6587        | 0.8931           |
| Pre-training default Fine Tuning (Half Dataset)    | 0.612         | 0.9571           |
| Class Imbalance (Cats: 'Maine', 'Birman', 'Bombay'; Dogs: All) | 0.63   | 0.92577          |

## VAE Results: Validation and Testing Performance

| Mode                                               | Val Dice Score | Test Dice Score |
|----------------------------------------------------|----------------|-----------------|
| Default (Pre-training & Fine Tuning)               | 0.849          | 0.851           |
| Pre-train with half dataset Finetuning default     | 0.826          | 0.829           |
| Fully Supervised (No Pretraining)                  | 0.8538         | 0.851           |
| Pre-training default Fine Tuning (Half Dataset)    | 0.8471         | 0.8484          |
| Class Imbalance (Cats: 'Maine', 'Birman', 'Bombay'; Dogs: All) | 0.8575   | 0.8433          |

## VAE Results: Test Set Performance

| Mode                                               | Test IOU Score | Test Accuracy |
|----------------------------------------------------|----------------|---------------|
| Default (Pre-training & Fine Tuning)               | 0.752          | 0.853         |
| Pre-train with half dataset Finetuning default     | 0.718          | 0.831         |
| Fully Supervised (No Pretraining)                  | 0.750          | 0.855         |
| Pre-training default Fine Tuning (Half Dataset)    | 0.746          | 0.850         |
| Class Imbalance (Cats: 'Maine', 'Birman', 'Bombay'; Dogs: All) | 0.740   | 0.846         |



