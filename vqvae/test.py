import torch
import numpy as np
from vqvae import *
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
torch.cuda.empty_cache()
device="cuda"

def calculate_supervised_dice_score(predictions, ground_truth):
    """
    Calculate the supervised Dice loss for labeled data.

    Args:
        predictions (torch.Tensor): Predictions for labeled data, with shape (batch_size, num_classes, height, width).
        ground_truth (torch.Tensor): Ground truth masks for labeled data, with shape (batch_size, num_classes, height, width).

    Returns:
        torch.Tensor: Supervised Dice loss.
    """
    # Convert raw model outputs into probabilities within the range [0, 1] to ensure alignment with the ground truth masks
    predictions = torch.sigmoid(predictions)
    
    # Smoothing factor to prevent division by zero
    smooth = 1e-5
    
    # Compute the intersection and union
    intersection = torch.sum(ground_truth * predictions, dim=(1, 2, 3))
    union = torch.sum(ground_truth, dim=(1, 2, 3)) + torch.sum(predictions, dim=(1, 2, 3))
    
    # Calculate the Dice score
    dice_score = 2 * (intersection + smooth) / (union + smooth)
    return dice_score.sum()

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

testing_data = torchvision.datasets.OxfordIIITPet(root='./data/oxford-pets', split='test',transform=transform,target_types="segmentation",target_transform=target_transform,download=True)
test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=True)

with torch.no_grad():
    net = VQVAE(in_channels=3,
                num_hiddens=256,
                num_downsampling_layers=2,
                num_residual_layers=4,
                num_residual_hiddens=256,
                embedding_dim=64,
                num_embeddings=512,
                use_ema=True,
                decay=0.99,
                epsilon=1e-5, )
    net.to(device)
    net.load_state_dict(torch.load("seg_vqvae.pt"))
    dice=0
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            imgs,segs=data
            imgs = imgs.float().to(device)
            segs = segs.float().to(device)
            out = net(imgs)
            dice+=calculate_supervised_dice_score(out["x_recon"],segs).item()
    print(dice/len(testing_data))

