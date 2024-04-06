import torch
import numpy as np
from vqvae import *
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import random

device="cuda"
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

training_data=ImageFolder("imgs",transform=transform)
train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)

net=VQVAE(in_channels=3,
        num_hiddens=256,
        num_downsampling_layers=2,
        num_residual_layers=4,
        num_residual_hiddens=256,
        embedding_dim=64,
        num_embeddings=512,
        use_ema=True,
        decay=0.99,
        epsilon=1e-5,)
net.to(device)

criterion = nn.MSELoss()
optimizer=torch.optim.Adam(net.parameters(),lr=1e-4)
for epoch in range(100):
    epoch_loss=0
    for i, data in enumerate(train_dataloader, 0):
        imgs,_=data
        imgs=imgs.float().to(device)
        inputs=distort(imgs).float().to(device)
        out=net(inputs)
        loss=criterion(out["x_recon"],imgs)+0.25*out["commitment_loss"]
        epoch_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, loss:",epoch_loss/i)
    torch.save(net.state_dict(), "pretrain_vqvae.pt")
print("done")

