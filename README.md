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
