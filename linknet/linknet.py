# Specifies the encoding to be used in this Python file, allowing for characters beyond ASCII
# -*- coding: utf-8 -*-

# Import necessary PyTorch modules for neural network creation and manipulation

import torch.nn as nn

from torchvision.models import resnet

# Define a decoder block, which will be used in the LinkNet architecture for upscaling and processing features
class decoder_block(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        # First convolution to reduce the channel size
        self.conv1 = nn.Conv2d(in_size, in_size//4, 1)
        self.norm1 = nn.BatchNorm2d(in_size//4)  # Batch normalization for stabilization
        self.relu1 = nn.ReLU(inplace=True)  # ReLU activation function for non-linearity

        # Transposed convolution for upscaling
        self.deconv2 = nn.ConvTranspose2d(in_size//4, in_size//4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_size//4)  # Another batch normalization layer
        self.relu2 = nn.ReLU(inplace=True)  # Another ReLU for non-linearity

        # Final convolution to match the desired output size
        self.conv3 = nn.Conv2d(in_size//4, out_size, 1)
        self.norm3 = nn.BatchNorm2d(out_size)  # Final batch normalization layer
        self.relu3 = nn.ReLU(inplace=True)  # Final ReLU activation

    # Define how the data flows through this block
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


# Define the LinkNet model, which uses a pretrained ResNet as the encoder
class link_net(nn.Module):
    def __init__(self, classes, encoder='resnet34'):
        super().__init__()

        # Initialize the ResNet encoder with pretrained weights
        res = resnet.resnet34(pretrained=True)
        
        # Use the initial layers of ResNet for encoding
        self.conv = res.conv1
        self.bn = res.bn1
        self.relu = res.relu
        self.maxpool = res.maxpool
        self.encoder1 = res.layer1
        self.encoder2 = res.layer2
        self.encoder3 = res.layer3
        self.encoder4 = res.layer4

        # Define the decoder blocks, progressively decreasing in channel size
        self.decoder4 = decoder_block(512, 256)
        self.decoder3 = decoder_block(256, 128)
        self.decoder2 = decoder_block(128, 64)
        self.decoder1 = decoder_block(64, 64)

        # Final layers for upscaling and classification
        self.finaldeconv = nn.ConvTranspose2d(64, 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, classes, 2, padding=1)

    # Define the forward pass, describing how the input tensor flows through the model
    def forward(self, x):
        # Encoder path
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder path with skip connections, adding features from the encoder path for better localization
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        # Final classification layers to produce the segmentation map
        x = self.finaldeconv(d1)
        x = self.finalrelu1(x)
        x = self.finalconv2(x)
        x = self.finalrelu2(x)
        x = self.finalconv3(x)
        return x