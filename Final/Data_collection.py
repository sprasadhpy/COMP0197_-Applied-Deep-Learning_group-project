


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

import random

torch.manual_seed(4)

# define transformations that will be applied to all the datasets
transform = transforms.Compose([
    transforms.Resize((120, 120)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # normalize to [-1, 1]
    transforms.Lambda(lambda x: x / 2 + 0.5)  # normalize to [0, 1]
])

# function to filter out images with a certain class
def filter_classes(dataset, class_lbl):
    indices = []
    classes = dataset.classes
    for i, (_, label) in enumerate(dataset):
      if label in [classes.index(class_lbl)]:
        indices.append(i)
    return indices



trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# filter out images with cats and dogs only
cat_indices = filter_classes(trainset, 'cat')
dog_indices = filter_classes(trainset, "dog")

# Create a subset containing only images with cats and dogs
cat_subset = torch.utils.data.Subset(trainset, cat_indices)
dog_subset = torch.utils.data.Subset(trainset, dog_indices)
print("The CIFAR10 dataset downloaded successfully.")

#os.remove('./data/cifar-10-python.tar.gz')


_URL = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'
tar_file_path = './data/stanford-dogs.tar'

response = requests.get(_URL)
with open(tar_file_path, 'wb') as f:
    f.write(response.content)

extract_dir = './data/stanford-dogs'
with tarfile.open(tar_file_path, 'r') as tar:
    tar.extractall(extract_dir)

os.remove(tar_file_path)

print("The Stanford Dogs dataset downloaded and extracted successfully.")

stanford_dogs_dataset = ImageFolder(root='./data/stanford-dogs/Images', transform=transform)
#classes = stanford_dogs_dataset.classes
stanford_dogs_dataset.classes = 'dog'

for i in range(len(stanford_dogs_dataset)):
    _, label = stanford_dogs_dataset.samples[i]
    stanford_dogs_dataset.samples[i] = stanford_dogs_dataset.samples[i][0], 'dog'

"""### Concatenating dogs datasets"""

dogs = ConcatDataset([stanford_dogs_dataset, dog_subset])
random_indices = torch.randperm(len(dogs))
cats = torch.utils.data.Subset(dogs, random_indices)
print(len(dogs))

# check dimensionality

dataloader = DataLoader(dogs, batch_size=1, shuffle=False)
dataset_size = len(dogs)
print("Size of the concatenated dataset:", dataset_size)

for image, _ in dataloader:
    sample_image_dimensions = image.shape[1:]  # Ignore batch dimension
    break

print("Dimensions of a sample image:", sample_image_dimensions)



_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_file_path = './data/cats_and_dogs_filtered.zip'

response = requests.get(_URL)
with open(zip_file_path, 'wb') as f:
    f.write(response.content)
extract_dir = './data'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

os.remove(zip_file_path)

print("Cats Kaggle dataset downloaded and extracted successfully.")

cats_dogs_train = ImageFolder(root='./data/cats_and_dogs_filtered/train', transform=transform)

cat_indices = filter_classes(cats_dogs_train, 'cats')

cat_subset_2 = torch.utils.data.Subset(cats_dogs_train, cat_indices)
print(len(cat_subset_2))

"""### Extracting "Cats" class from Microsoft Cats&Dogs dataset"""

_URL = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip'
zip_file_path = './data/kagglecatsanddogs.zip'

response = requests.get(_URL)
with open(zip_file_path, 'wb') as f:
    f.write(response.content)
extract_dir = './data'
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

os.remove(zip_file_path)

print("Microsoft cats and dogs dataset downloaded and extracted successfully.")

cats_dogs_train_3 = ImageFolder(root='./data/PetImages', transform=transform)
classes = cats_dogs_train_3.classes

cat_indices = []
for idx in range(len(cats_dogs_train_3)):
    try:
        img, label = cats_dogs_train_3[idx]
        if label == classes.index('Cat'):
            cat_indices.append(idx)
    except Exception as e:
        print(f"Skipped corrupted file at index {idx}: {e}")


# create a subset containing only cat images
cat_subset_3 = torch.utils.data.Subset(cats_dogs_train_3, cat_indices)
print(len(cat_subset_3))

"""### Concatenating cats datasets"""

cats = ConcatDataset([cat_subset, cat_subset_2, cat_subset_3])
random_indices = torch.randperm(len(cats))
cats = torch.utils.data.Subset(cats, random_indices)
print(len(cats))

# check dimensionality

dataloader = DataLoader(cats, batch_size=1, shuffle=False)
dataset_size = len(cats)
print("Size of the concatenated dataset:", dataset_size)
for image, _ in dataloader:
    sample_image_dimensions = image.shape[1:]  # Ignore batch dimension
    break

print("Dimensions of a sample image:", sample_image_dimensions)

"""### Concatenating cats and dogs datasets and ensure equal sizes"""

dogs_reduced = torch.utils.data.Subset(dogs, torch.randperm(len(cats)))
pets = ConcatDataset([cats, dogs_reduced])
pets = torch.utils.data.Subset(pets,torch.randperm(len(pets)))

print(len(pets))

"""### Saving data for pretraining in a separate folder"""

output_dir = "./pets_images"
os.makedirs(output_dir, exist_ok=True)

for i, (image, label) in enumerate(pets):
    image_pil = transforms.ToPILImage()(image)
    filename = f"image_{i}.jpg"
    image_path = os.path.join(output_dir, filename)
    image_pil.save(image_path)

print("Images saved successfully.")
