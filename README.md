# COMP0197 - Applied-Deep-Learning_Group-project

# Team
| Name                        | Degree                                         | Email                                  |
|-----------------------------|------------------------------------------------|----------------------------------------|
| Abhineet kumar              | MSc Computational Statistics and Machine Learning | ucabku0@ucl.ac.uk                   |
| Akhil Prakash Shenvi Sangaonkar | MSc Computational Statistics and Machine Learning | ucabap7@ucl.ac.uk                   |
| Bushra Aldhanhani           | MSc Machine Learning                          | Bushra.Aldhanhani.23@ucl.ac.uk        |
| Idan Glassberg              | MSc Machine Learning                          | ucabig0@ucl.ac.uk                     |
| Marianna Rybnikova          | MSc Computational Statistics and Machine Learning | marianna.rybnikova.23@ucl.ac.uk     |
| Shyaam Prasadh              | MSc Machine Learning                          | ucabsr5@ucl.ac.uk                     |
| Sreekar Cango               | MSc Robotics and Computation                  | ucabs44@ucl.ac.uk                     |
| Sruthi Susan Kuriakose      | MSc Machine Learning                          | ucaburi@ucl.ac.uk                     |


# Self-supervised learning - Group project

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

| Metric     |   Pretrain VQ-VAE Full Finetuned |   SL VQ-VAE |   Pretrain VQ-VAE Half Finetune |   Pretrain VQ-VAE Quarter Finetune |   Half Pretrain VQ-VAE Fully Finetune |
|:-----------|---------------------------------:|------------:|--------------------------------:|-----------------------------------:|--------------------------------------:|
| Dice Score |                            0.864 |       0.853 |                           0.845 |                              0.829 |                                 0.855 |
| Accuracy   |                            0.865 |       0.857 |                           0.846 |                              0.830 |                                 0.857 |
| IoU Score  |                            0.769 |       0.752 |                           0.742 |                              0.719 |                                 0.755 |

#### Model performance Insights


1. Pretrain VQ-VAE Full Finetuned Model columns exhibits the highest scores across all three metrics and it implies that full finetuning after pretraining leads to substantial improvements in model performance metrics. This configuration sets the benchmark- showing a slight decrease in Dice Score (-0.12%), and a more noticeable decrease in IoU Score (-11.10%).
   
2. SL VQ-VAE Model exhibits high accuracy, suggesting that the SL VQ-VAE setup is robust, albeit with a slightly lower capability in segmenting images as reflected in the IoU Score (-12.25%). However, demonstrates a small decline in Dice Score (-0.47%) compared to the full finetuned model.
   
3. Pretrain VQ-VAE Half Finetune Model also indicate a decrease in performance metrics compared to full finetuning, with a -2.20% change in Dice Score and -12.29% in IoU Score.  Highlights the importance of finetuning in achieving optimal model performance.
 
4. Pretrain VQ-VAE Quarter Finetune Model shows a further decline in all metrics, with Dice Score dropping by -4.05% and IoU Score by -13.37%.These results reinforce that finetuning is directly correlated with the model's accuracy and segmentation quality.
   
5.  Half Pretrain VQ-VAE Fully Finetune Model demonstrates that full finetuning can substantially compensate, achieving metrics close to those of the fully pretrained and finetuned model especially in Dice Score (-0.23%) and Accuracy, and with a smaller decrease in IoU Score (-11.90%).

--------------------------------------

# Pretraining Data 

## Collection

In this project, pre-training data was collected from different datasets which are the CIFAR-10, Stanford dataset, Kaggle set and Microsoft Cats&Dogs dataset. The C-10 dataset consists of 60,000 32x32 colour images in 10 different classes, with 6,000 images per class. The classes of interest were the dogs class and the cats class. The Stanford Dogs Dataset contains 20,580 images of dogs, which represent 120 different breeds. On the other hand, the Kaggle dogs and cats set includes about 1,500 images of cats and 1,500 images of dogs, while the  Microsoft Cats&Dogs dataset includes 12,500 images of cats and 12,500 images of dogs. Except for CIFAR-10, the other datasets do not have a consistent image size and it can vary within the same class. The data collected for dogs is from the CIFAR-10 and Stanford dataset and consists of 25580 images. However, the cats' dataset which is collected from CIFAR-10,  Kaggle set and Microsoft Cats&Dogs dataset only consists of 18499 images. To ensure equality, the dogs' dataset was reduced to be the same size as the cats' dataset.

## Preprocessing 

After compiling data from various publicly available datasets and consolidating them into our final dataset, the images underwent normalization to a range of 0 to 1. To augment the dataset and improve model robustness, we initially considered several transformations:

- Rotation: Randomly rotating images by 0, 90, 180, and 270 degrees.
- Color Jittering: Adjusting brightness, contrast, saturation, and hue with specified values (brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1).
- Greyscale Function: Converting images to greyscale with a predefined probability *p*.
- Masking: Masking a predefined percentage of each image through random area selection.
  
However, for the final pretraining stage, only the masking transformation (images were randomly assigned to masking ranging from 5% to 25%) was employed to enhance the dataset.

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
------------------------------------------------------------------------------------------------------

# Benchmark Architecture - SegNet

SegNet is a deep convolutional encoder-decoder architecture. It is designed to take an image as input and produce a pixel-wise label map as output.The encoder captures high-level features by applying a series of convolutional and pooling layers. The key innovation in SegNet is its decoder network, which uses the spatial pooling indices generated during the max-pooling in the encoder phase to upsample and produce a  segmentation map. This is a form of up-sampling that is both memory-efficient and helps to preserve the fine-grained details in the output.

This model can be used as a fully supervised model, and also as a pretrained model. The pretraining has been done using the Masked Autoencoder approach on our pretraining dataset.

All relevant python scripts are encapsulated in ```SegNet/SegNetMSE.py``` . It can be run using ```python SegNet/SegNetMSE.py```

## Evaluation Metrics - Results of Segnet (pretrained and finetuned) on Oxford-IIIT Pet Dataset

| Metric      | Result  |
|-------------|---------|
| Dice Score  | 0.511  |
| Accuracy    | 0.875  |
| IoU Score   | 0.343  |
---------------------------------------------------------------------------------------------------
# Experiments

### 1) Class Imbalance Experiment & 2) Pretrain with half dataset 

The Default mode that includes pre-training and fine-tuning shows robust scores across the board with the lowest training loss, high Train and Test Dice scores, and the highest Test Accuracy, indicating a well-optimized model to predict. Other training modes such as including training with half the dataset and also, fully supervised without pretraining show higher losses and slightly lower scores, suggesting some impact on the model's ability to generalize. However, our experiment on the Class Imbalance mode, designed to tackle biased datasets, performs admirably well. Though with a slight compromise in Test IOU and Accuracy, which is a common challenge in such dataset scenarios. Overall, the consistency in high Dice and IOU scores across all training modes reflects the model's strong segmentation ability.

## VQ-VAE Results: Training Performance

| Mode                                               | Training Loss | Train Dice Score |
|----------------------------------------------------|---------------|------------------|
| Default (Pre-training & Fine Tuning)               | 0.61          | 0.945            |
| Pre-train with half dataset Finetuning default     | 0.6182        | 0.937            |
| Fully Supervised (No Pretraining)                  | 0.6587        | 0.8931           |
| Pre-training default Fine Tuning (Half Dataset)    | 0.612         | 0.9571           |
| Class Imbalance (Cats: 'Maine', 'Birman', 'Bombay'; Dogs: All) | 0.63   | 0.92577          |

## VQ-VAE Results: Validation and Testing Performance

| Mode                                               | Val Dice Score | Test Dice Score |
|----------------------------------------------------|----------------|-----------------|
| Default (Pre-training & Fine Tuning)               | 0.849          | 0.851           |
| Pre-train with half dataset Finetuning default     | 0.826          | 0.829           |
| Fully Supervised (No Pretraining)                  | 0.8538         | 0.851           |
| Pre-training default Fine Tuning (Half Dataset)    | 0.8471         | 0.8484          |
| Class Imbalance (Cats: 'Maine', 'Birman', 'Bombay'; Dogs: All) | 0.8575   | 0.8433          |

## VQ-VAE Results: Test Set Performance

| Mode                                               | Test IOU Score | Test Accuracy |
|----------------------------------------------------|----------------|---------------|
| Default (Pre-training & Fine Tuning)               | 0.752          | 0.853         |
| Pre-train with half dataset Finetuning default     | 0.718          | 0.831         |
| Fully Supervised (No Pretraining)                  | 0.750          | 0.855         |
| Pre-training default Fine Tuning (Half Dataset)    | 0.746          | 0.850         |
| Class Imbalance (Cats: 'Maine', 'Birman', 'Bombay'; Dogs: All) | 0.740   | 0.846         |


### 3) Image Segmentation Blurring Experiment

In the image segmentation task, we conducted a blurring experiment to analyze the impact of blurring levels on model performance. Blurring levels ranged from 0.0% to 100.0%, and metrics such as Dice Score, IoU Score, and Accuracy were evaluated for each level.

## Blurring Experiment Results

The table below presents the metrics obtained for different blurring levels:

| Blurring Level | Dice Score | IoU Score | Accuracy |
|----------------|------------|-----------|----------|
| 0.0%           | 0.851      | 0.751     | 0.853    |
| 25.0%          | 0.749      | 0.618     | 0.750    |
| 50.0%          | 0.683      | 0.531     | 0.683    |
| 75.0%          | 0.649      | 0.494     | 0.649    |
| 100.0%         | 0.621      | 0.466     | 0.621    |

These results provide insights into how blurring affects the segmentation model's performance, with decreasing scores observed as blurring levels increase.


## 4) VQ-VAE Performance Metrics Comparison -SSL Vs.SL

This table compares the performance of the Vector Quantised-Variational AutoEncoder (VQ-VAE) in Self-Supervised Learning (SSL) and Supervised Learning (SL) settings, rounded to three decimal places.

| Metric    | SSL VQ-VAE  | Supervised Learning VQ-VAE |
|-----------|-------------|----------------------------|
| Dice Score| 0.864       | 0.853                      |
| Accuracy  | 0.865       | 0.857                      |
| IoU Score | 0.769       | 0.752                      |

Interpretation: The VQ-VAE model shows a marginally better performance in SSL across all metrics- suggesting an advantage in this learning context.


