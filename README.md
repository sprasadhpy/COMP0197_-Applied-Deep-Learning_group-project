# COMP0197_-Applied-Deep-Learning_group-project
Self-supervised learning - group project

Instructions to run are in instructions.txt

# LinkNet :Benchmark Architecture

LinkNet is a convolutional neural network architecture designed for image segmentation tasks. It utilizes a ResNet model as the encoder and decoder blocks for feature extraction and upscaling.

## Requirements

- Python 3.x
- PyTorch
- torchvision

## Usage

To use LinkNet in your project, follow these steps:

1. **Import the necessary modules:**
    ```python
    import torch.nn as nn
    from torchvision.models import resnet
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

4. **Instantiate the LinkNet model:**
    ```python
    model = link_net(classes=NUM_CLASSES)
    ```

5. **Perform forward pass:**
    ```python
    output = model(input_tensor)
    ```

## Implementation Details

- **Decoder Block:** Implements a decoder block for upscaling and feature processing.
- **LinkNet Model:** Utilizes a ResNet encoder followed by decoder blocks with skip connections for better feature localization.
- **Pretrained Weights:** The ResNet encoder is initialized with pretrained weights.

## Example

```python
# Import necessary modules
import torch.nn as nn
from torchvision.models import resnet

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

# Instantiate the LinkNet model
model = link_net(classes=NUM_CLASSES)

# Perform forward pass
output = model(input_tensor)
