# -*- coding: utf-8 -*-
"""TosruthiFinal.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cErvRZPMFyKspC1lKxThr7OBvvSxRNkj
"""

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

#  Assuming folder_path is already set to the current working directory
#with zipfile.ZipFile(f"{folder_path}/pets_images_final.zip", 'r') as zip_ref:
#   zip_ref.extractall("/sample_data")
#pretrain_data_path = '/pets_images'

# List everything in /tmp
tmp_contents = os.listdir(pretrain_data_path)
print(tmp_contents[:10])

num_workers = os.cpu_count()
num_workers

transform = transforms.Compose([
    transforms.Resize((120, 120)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # normalize to [-1, 1]
    transforms.Lambda(lambda x: x / 2 + 0.5),  # normalize to [0, 1]
])

def greyscale(image):
    grayscale_transform = transforms.RandomGrayscale(p=0.2)
    imgs = []
    for i in range(image.shape[0]):
        im=transforms.ToPILImage()(image[i])
        greyscale_image = grayscale_transform(im)
        imgs.append(transform(greyscale_image).unsqueeze(0))

    return torch.cat(imgs,0)

def distort_color(image):
    color_jitter = transforms.ColorJitter(
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.1
    )
    imgs=[]
    for i in range(image.shape[0]):
        im=transforms.ToPILImage()(image[i])
        distorted_image = color_jitter(im)
        imgs.append(transform(distorted_image).unsqueeze(0))
    return torch.cat(imgs,0)

def mask_image(image, low_mask_percentage,high_mask_percentage):
    for i in range(image.shape[0]):
        mask_percentage=random.randint(low_mask_percentage,high_mask_percentage)
        rows, cols = image.shape[2:]
        rows_to_mask = int(rows * mask_percentage / 100)
        cols_to_mask = int(cols * mask_percentage / 100)

        top_left_row = random.randint(0, rows - rows_to_mask)
        top_left_col = random.randint(0, cols - cols_to_mask)

        image[i,:, top_left_row:top_left_row + rows_to_mask, top_left_col:top_left_col + cols_to_mask] = 0.5

    return image

distort=transforms.Compose([
    transforms.RandomChoice([lambda x: mask_image(x,5,25),greyscale,distort_color]),
])

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

class CustomImageDataset(Dataset):
    """A custom dataset class for loading preprocessed and saved images."""

    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (string): Path to the directory with preprocessed images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, img_name) for img_name in os.listdir(image_dir)
                            if os.path.isfile(os.path.join(image_dir, img_name))]

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to be fetched

        Returns:
            A tuple (image, label), where 'image' is the transformed image tensor,
            and 'label' could be a dummy value if not applicable.
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Convert to RGB for consistency

        if self.transform:
            image = self.transform(image)

        label = 0
        return image

pre_training_data= CustomImageDataset(pretrain_data_path,transform=transform)
pre_train_dataloader = DataLoader(pre_training_data, batch_size=64, shuffle=True)
len(pre_train_dataloader)

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

    def forward(self ,batch: torch.Tensor):
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
        y =x/ x.max()
        return y

model = Segnet().to(device)
criterion1 =  torch.nn.MSELoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

def calculate_supervised_dice_score(predictions, ground_truth):
    """
    Calculate the supervised Dice score for labeled data.

    Args:
        predictions (torch.Tensor): Predictions for labeled data, with shape (batch_size, num_classes, height, width).
        ground_truth (torch.Tensor): Ground truth masks for labeled data, with shape (batch_size, num_classes, height, width).

    Returns:
        torch.Tensor: Supervised Dice score.
    """
    # Convert raw model outputs into probabilities within the range [0, 1] to ensure alignment with the ground truth masks
    predictions = torch.sigmoid(predictions)

    # Smoothing factor to prevent division by zero
    smooth = 1e-5

    # Compute the intersection and union
    intersection = torch.sum(ground_truth * predictions, dim=(1, 2, 3))
    #cardinality = torch.sum(ground_truth, dim=(1, 2, 3)) + torch.sum(predictions, dim=(1, 2, 3))
    cardinality = torch.sum(ground_truth + predictions, dim=(1, 2, 3))

    # Calculate the Dice score
    dice_score = 2 * (intersection + smooth) / (cardinality+ smooth)
    return dice_score.sum()

def calculate_supervised_iou_score(predictions, ground_truth):
    """
    Calculate the supervised IOU score for labeled data.

    Args:
        predictions (torch.Tensor): Predictions for labeled data, with shape (batch_size, num_classes, height, width).
        ground_truth (torch.Tensor): Ground truth masks for labeled data, with shape (batch_size, num_classes, height, width).

    Returns:
        torch.Tensor: Supervised IOU score.
    """
    # Convert raw model outputs into probabilities within the range [0, 1] to ensure alignment with the ground truth masks
    predictions = torch.sigmoid(predictions)

    # Smoothing factor to prevent division by zero
    smooth = 1e-5

    # Compute the intersection and union
    intersection = torch.sum(ground_truth * predictions, dim=(1, 2, 3))
    union = torch.sum(ground_truth, dim=(1, 2, 3)) + torch.sum(predictions, dim=(1, 2, 3)) -intersection

    # Calculate the IOU score
    iou_score = (intersection + smooth) / (union + smooth)

    return iou_score.sum()

def pretrain(net, criterion, optimizer, trainloader, epochs):
    trainloss=[]
    net.train().to(device)
    dl=[]
    iouscore=[]
    for epoch in range(epochs):
        print('Epoch', epoch+1)
        running_loss = 0.0
        dice=0.0
        total=0.0
        iou=0.0
        for i, data in tqdm(enumerate(trainloader, 0)):
            optimizer.zero_grad()
            inputs = data.to(device)
            inputs_distorted=distort(inputs).float().to(device)
            outputs = net(inputs_distorted)
            loss = criterion(outputs, inputs)
            loss.backward(retain_graph=False)
            dice+=calculate_supervised_dice_score(outputs,inputs).item()
            iou+=calculate_supervised_iou_score(outputs,inputs).item()
            optimizer.step()

            running_loss += loss.detach().item()
            total += inputs.size(0)
        trainloss.append(running_loss/total)
        iouscore.append(iou/total)
        dl.append(dice/total)
        print('[%d, %2d]   loss: %.5f Dice Score: %.5f IOU Score: %.5f'  %
                    (epoch + 1,epochs, trainloss[epoch],dl[epoch] ,iouscore[epoch]))
    return trainloss , dl , iouscore

mse , dlpt, iou = pretrain(model, criterion1, optimizer, pre_train_dataloader, 20)

model_path = os.path.join(folder_path, f'segnet_pretrained_MSE.pt')
torch.save(model.state_dict(), model_path)

print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

transform = transforms.Compose([
    transforms.Resize((120, 120)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # normalize to [-1, 1]
    transforms.Lambda(lambda x: x / 2 + 0.5),  # normalize to [0, 1]
])
target_transform=transforms.Compose([
    transforms.Resize((120, 120),interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.nn.functional.one_hot((x*255).long() - 1,3).squeeze(0).permute(2,0,1))

])



training_data = torchvision.datasets.OxfordIIITPet(root='./data/oxford-pets', split='trainval',transform=transform,target_types="segmentation",target_transform=target_transform,download=True)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True,num_workers=num_workers)




testing_data = torchvision.datasets.OxfordIIITPet(root='./data/oxford-pets', split='test',transform=transform,target_types="segmentation",target_transform=target_transform,download=True)
test_dataloader = DataLoader(testing_data, batch_size=8, shuffle=False,num_workers=num_workers)

(train_pets_inputs, train_pets_targets) = next(iter(train_dataloader))

train_pets_inputs.shape, train_pets_targets.shape

pets_input_grid = torchvision.utils.make_grid(train_pets_inputs, nrow=8)
t2img(pets_input_grid)

pets_targets_grid = torchvision.utils.make_grid(train_pets_targets /1. , nrow=8)
t2img(pets_targets_grid)



def train(net, criterion, optimizer, trainloader, epochs):
    trainloss=[]
    dl=[]
    iouscore=[]
    net.train().to(device)
    for epoch in range(epochs):

        print('Epoch', epoch+1)
        running_loss = 0.0
        dice=0.0
        total=0.0
        iou=0.0
        acc=0.0
        for  batch_idx, data in tqdm(enumerate(trainloader, 0)):
            optimizer.zero_grad()
            inputs, labels = data
            inputs, labels = inputs.float().to(device), labels.float().to(device)
            outputs = net(inputs)





            loss = criterion(outputs, labels)

            loss.backward(retain_graph=False)


            dice+=calculate_supervised_dice_score(outputs,labels).item()
            iou+=calculate_supervised_iou_score(outputs,labels).item()
            acc+=torch.mean((torch.argmax(outputs,1)==torch.argmax(labels,1)).float(),dim=(1,2)).sum().item()
            optimizer.step()
            running_loss += loss.detach().item()
            total += inputs.size(0)
        dl.append(dice/total)
        iouscore.append(iou/total)
        print("accuracy:", acc / total)
        trainloss.append(running_loss/total)
        print('[%d, %2d]   loss: %.5f Dice Score: %.5f IOU Score: %.5f'  %
                    (epoch + 1,epochs, trainloss[epoch],dl[epoch] ,iouscore[epoch]))

    return trainloss , dl , iouscore

tl , dl,iou = train(model, criterion1, optimizer, train_dataloader, 50)

model_path = os.path.join(folder_path, f'segnet_trained_MSE.pt')
torch.save(model.state_dict(), model_path)

def test(model,  test_data, epochs=20):
    testacc=[]
    ds=[]
    iouscore=[]
    model.eval().to(device)
    for epoch in range(epochs):
        print('Epoch',epoch+1)
        dice=0.0
        total=0.0
        acc=0.0
        iou=0.0
        with torch.no_grad():
            for batch_idx, data in tqdm(enumerate(test_data, 0)):
                inputs , labels = data
                #inputs,labels = inputs.to(device), labels.to(device)
                #print(labels.shape,'label')
                #print(inputs.shape,'input')
                #predictions = model(inputs)
                #print(predictions.shape,'pred')
                #print(predictions.shape,labels.shape)
                #pred = nn.Softmax(dim=1)(predictions)

                inputs,labels = inputs.to(device), labels.to(device)
                #print(labels.shape,'label')
                #print(inputs.shape,'input')

                predictions = model(inputs)



                #pred = nn.Softmax(dim=1)(predictions)

                #pred_labels = pred.argmax(dim=1)
                #print(pred_labels.shape)
                #pred_labels = pred_labels.unsqueeze(1)
                #print(pred_labels.shape)

                #pred_mask = pred_labels.to(torch.float)
                acc+=torch.mean((torch.argmax(predictions,1)==torch.argmax(labels,1)).float(),dim=(1,2)).sum().item()
                dice +=calculate_supervised_dice_score(predictions,labels).item()
                #acc += IoUMetric(pred, labels).detach().item()
                iou+=calculate_supervised_iou_score(predictions,labels).item()
                #pred_labels = predictions.argmax(dim=1)
                #print(pred_labels.shape)
                #pred_labels = pred_labels.unsqueeze(1)
                #print(pred_labels.shape)
                #pred_mask = pred_labels.to(torch.float)
                #print(pred_mask.shape)
                #dice+=calculate_supervised_dice_score(predictions,labels).item()
                total += inputs.size(0)



        ds.append(dice/total)
        iouscore.append(iou/total)
        testacc.append(acc/total)
        print('Testing : [%d, %2d]   Dice Score: %.5f IOU Score: %.5f Accuracy : %.5f ' %
                    (epoch + 1,epochs,   ds[epoch] ,iouscore[epoch],testacc[epoch]))
        if (epoch +1 == 1):
          fig = plt.figure(figsize=(10, 12))
          fig.suptitle('Segmentations', fontsize=12)

          fig.add_subplot(3, 1, 1)
          plt.imshow(t2img(torchvision.utils.make_grid(inputs, nrow=8)))
          plt.axis('off')
          plt.title("Images")

          fig.add_subplot(3, 1, 2)
          plt.imshow(t2img(torchvision.utils.make_grid(labels.float()/1.0 , nrow=8)))
          plt.axis('off')
          plt.title("Ground Truth Labels")

          fig.add_subplot(3, 1, 3)
          plt.imshow(t2img(torchvision.utils.make_grid(predictions, nrow=8)))
          plt.axis('off')
          plt.title("Predicted Labels")
    return

test(model,  test_dataloader, epochs=1)
