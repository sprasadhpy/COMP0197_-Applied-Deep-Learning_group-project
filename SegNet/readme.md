
The pytorch models can be downloaded from https://liveuclac-my.sharepoint.com/:f:/r/personal/ucabap7_ucl_ac_uk/Documents/CW2%20Segnet%20Models?csf=1&web=1&e=54IEqO
Required package tqdm and matplotlib
pip install tqdm
conda install matplotlib
To run the file first download the data from DataCollection.py.
pretrain_dataset_path should be the folder containing the downloaded data from  DataCollection.py
Then run the SegNet.py file



The saved pytorch models can be found here:
https://liveuclac-my.sharepoint.com/:f:/g/personal/ucabap7_ucl_ac_uk/EgddPL4y9IpKmct5_MG8h7IBIi0VEEUWoZeoCfvMaz4Fqg?e=UJkOMp
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


