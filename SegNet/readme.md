
The pytorch models can be downloaded from 
https://liveuclac-my.sharepoint.com/:f:/r/personal/ucabap7_ucl_ac_uk/Documents/CW2%20Segnet%20Models?csf=1&web=1&e=54IEqO
Required package tqdm and matplotlib


To run the file first download the data from DataCollection.py.
pretrain_dataset_path should be the folder containing the downloaded data from  DataCollection.py
Then run the SegNet.py file


The saved pytorch models can be found here:
https://liveuclac-my.sharepoint.com/:f:/g/personal/ucabap7_ucl_ac_uk/EgddPL4y9IpKmct5_MG8h7IBIi0VEEUWoZeoCfvMaz4Fqg?e=UJkOMp

 **Required packages:**
    ```
    conda install matplotlib
    pip install tqdm
    ```
1. **Import the necessary modules:**
    ```python
    import torch
    import datetime
    import torchvision
    import torch.nn as nn
    import os
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    import requests
    import tarfile
    from torchvision.datasets import ImageFolder
    from torch.utils.data import ConcatDataset
    import zipfile
    from PIL import Image, ImageEnhance
    import torchvision.transforms.functional as TF
    from torchvision import datasets, transforms
    import torch.optim as optim
    from tqdm import tqdm
    import random
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(4)
    import torchvision.transforms as T
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    t2img = T.ToPILImage()
    img2t = T.ToTensor()
    ```

2. **Define the Model:**
    ```python
    class DC2(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        y = self.seq(x)
        y_shape = y.shape
        y, index = self.max_pool (y)
        return y, index, y_shape

    class DC3(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
        )
        self.max_pool  = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, x):
        y = self.seq(x)
        y_shape = y.shape
        y, index = self.max_pool(y)
        return y, index, y_shape

    class UC2(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
        )
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, index, output_size):
        y = self.max_unpool(x, index, output_size=output_size)
        y = self.seq(y)
        return y

    class UC3(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=input_channel, out_channels=input_channel, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(),
        )
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2)

    def forward(self, x, index, output_size):
        y = self.max_unpool(x, index, output_size=output_size)
        y = self.seq(y)
        return y


    class Segnet(torch.nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.out_channels = 3
        self.batch_norm_input = nn.BatchNorm2d(3)
        self.dc_1 = DC2(3, 64, kernel_size=kernel_size)
        self.dc_2 = DC2(64, 128, kernel_size=kernel_size)
        self.dc_3 = DC3(128, 256, kernel_size=kernel_size)
        self.dc_4 = DC3(256, 512, kernel_size=kernel_size)


        self.uc_4 = UC3(512, 256, kernel_size=kernel_size)
        self.uc_3 = UC3(256, 128, kernel_size=kernel_size)
        self.uc_2 = UC2(128, 64, kernel_size=kernel_size)
        self.uc_1 = UC2(64, 3, kernel_size=kernel_size)

    def forward(self, batch: torch.Tensor):
        x = self.batch_norm_input(batch)

        # SegNet Encoder
        x, max_pool_1_index, s1 = self.dc_1(x)
        x, max_pool_2_index, s2 = self.dc_2(x)
        x, max_pool_3_index, s3 = self.dc_3(x)
        x, max_pool_4_index, s4 = self.dc_4(x)


        # SegNet Decoder

        x = self.uc_4(x, max_pool_4_index, output_size=s4)
        x = self.uc_3(x, max_pool_3_index, output_size=s3)
        x = self.uc_2(x, max_pool_2_index, output_size=s2)
        x = self.uc_1(x, max_pool_1_index, output_size=s1)

        return x
    ```
```
device = "cuda" if torch.cuda.is_available() else "cpu"
```
4. **Training and Evaluation:**
    ```python
    Define the model , optimiser and loss
    model = Segnet().to(device)
    criterion1 =  torch.nn.MSELoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    # pre train the model
    mse , dlpt, iou = pretrain(model, criterion1, optimizer, pre_train_dataloader, 20)
    # train the model
    tl , dl,iou = train(model, criterion1, optimizer, train_dataloader, 50)

    # Evaluate the model

    test(model,  test_dataloader, epochs=1)
    ```

# Input and Output Feature Maps of segNet



| Block | Encoder (m x n) | Decoder (m x n) |
|-------|-----------------|-----------------|
| 1.    | 3 x 64          | 64 x 3          |
| 2.    | 64 x 128        | 128 x 64        |
| 3.    | 128 x 256       | 256 x 128       |
| 4.    | 256 x 512       | 512 x 256       |



