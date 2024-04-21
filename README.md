# COMP0197_-Applied-Deep-Learning_group-project
Self-supervised learning - group project

Instructions to run are in instructions.txt

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


