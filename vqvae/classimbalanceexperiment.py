import torch
import numpy as np
from vqvae import *
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from class_imbalance import get_imbalance_dataset
from torch.utils.data import SubsetRandomSampler
torch.cuda.empty_cache()

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
    
    # Compute the intersection and cardinality
    intersection = torch.sum(ground_truth * predictions, dim=(1, 2, 3))
    cardinality = torch.sum(ground_truth, dim=(1, 2, 3)) + torch.sum(predictions, dim=(1, 2, 3))
    
    # Calculate the Dice score
    dice_score = 2 * (intersection + smooth) / (cardinality + smooth)
    return dice_score.sum()




device="cuda"
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

use_pretrained=True # Set this to False to train the model from scratch(Fully supervised)
load_half_data=False # Set this to True to load half of the data
use_imbalance_data=False # Set this to True to use the imbalance dataset


training_data = torchvision.datasets.OxfordIIITPet(root='./data/oxford-pets',transform=transform,target_types="segmentation",target_transform=target_transform, download=True)
if use_imbalance_data:
    training_data=get_imbalance_dataset()

total_samples = len(training_data)
train_samples = int(total_samples * 0.8)
val_samples = total_samples - train_samples

training_data, val_data = torch.utils.data.random_split(training_data, [train_samples, val_samples])
# Calculate the total number of samples and the half point in the training dataset
total_samples = len(training_data)
print('Total number of samples:', total_samples)
half_point = total_samples // 2
if load_half_data:
    # If load_half_data is True, use only half of the training dataset
    training_data = torch.utils.data.Subset(training_data, range(half_point))


train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)

net=VQVAE(in_channels=3,
        num_hiddens=256,
        num_downsampling_layers=2,
        num_residual_layers=4,
        num_residual_hiddens=256,
        embedding_dim=64,
        num_embeddings=512,
        use_ema="use_ema",
        decay=0.99,
        epsilon=1e-5,)
net.to(device)

if use_pretrained:
    net.load_state_dict(torch.load("pretrain_vqvae.pt"))

criterion = nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(),lr=2e-4) # use lr=2e-4 for supervised learning
for epoch in range(50):
    epoch_loss=0
    dice=0
    for i, data in enumerate(train_dataloader, 0):
        imgs,segs=data
        imgs=imgs.float().to(device)
        segs=segs.float().to(device)
        out=net(imgs)
        loss = criterion(torch.sigmoid(out["x_recon"]), segs) + 0.25 * out["commitment_loss"]
        dice+=calculate_supervised_dice_score(out["x_recon"],segs).item()
        epoch_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    val_dice=0
    with torch.no_grad():
        for j, data in enumerate(val_dataloader, 0):
            imgs,segs=data
            imgs=imgs.float().to(device)
            segs=segs.float().to(device)
            out=net(imgs)
            val_dice+=calculate_supervised_dice_score(out["x_recon"],segs).item()
    print(f"Epoch {epoch}, loss: {epoch_loss/i}, dice score: {dice/len(training_data)}, val dice score: {val_dice/len(val_data)}")
    torch.save(net.state_dict(), "seg_vqvae.pt")
print("done")

