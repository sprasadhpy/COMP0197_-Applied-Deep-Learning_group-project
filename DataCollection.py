import torch
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

#### Extracting "cats" and "dogs" classes from CIFAR10 ####
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# filter out images with cats and dogs only
cat_indices = filter_classes(trainset, 'cat')
dog_indices = filter_classes(trainset, "dog")

# Create a subset containing only images with cats and dogs
cat_subset = torch.utils.data.Subset(trainset, cat_indices)
dog_subset = torch.utils.data.Subset(trainset, dog_indices)
print("The CIFAR10 dataset downloaded successfully.")

#### Extracting "dogs" classes from Stanford dataset ####
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


#### Concatenating dogs datasets ####
dogs = ConcatDataset([stanford_dogs_dataset, dog_subset])
random_indices = torch.randperm(len(dogs))
cats = torch.utils.data.Subset(dogs, random_indices)

#### Extracting "Cats" class from kaggle set ####
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


#### Extracting "Cats" class from Microsoft Cats&Dogs dataset ####
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


#### Concatenating cats datasets ####
cats = ConcatDataset([cat_subset, cat_subset_2, cat_subset_3])
random_indices = torch.randperm(len(cats))
cats = torch.utils.data.Subset(cats, random_indices)

#### Concatenating cats and dogs datasets and ensure equal sizes ####
dogs_reduced = torch.utils.data.Subset(dogs, torch.randperm(len(cats)))
pets = ConcatDataset([cats, dogs_reduced])
pets = torch.utils.data.Subset(pets,torch.randperm(len(pets)))


#### Saving data for pretraining in a separate folder ####
output_dir = "./data/pretraining_data"
os.makedirs(output_dir, exist_ok=True)

for i, (image, label) in enumerate(pets):
    image_pil = transforms.ToPILImage()(image)
    filename = f"image_{i}.jpg"
    image_path = os.path.join(output_dir, filename)
    image_pil.save(image_path)

print("Pretraining data have been collected successfully.")


#### Implement Transformation Functions ####

# function to randomly rotate an image and return the rotation degree
def rotate_image(image):
    degrees = [0, 90, 180, 270]
    deg = random.choice(degrees)
    rotated_image = image.rotate(deg, expand=True)
    return rotated_image, deg

# function to apply color distortion to an image and return the distorted image
def distort_color(image):
    color_jitter = transforms.ColorJitter(
        brightness=0.5,
        contrast=0.5,
        saturation=0.5,
        hue=0.1
    )
    distorted_image = color_jitter(image)
    return distorted_image

# function to apply random greyscale transformation to an image
def greyscale(image, proportion):
    grayscale_transform = transforms.RandomGrayscale(p=proportion)
    greyscale_image = grayscale_transform(image)
    return greyscale_image

# function to apply masking transformation to a random part of the image
def mask_image(image, mask_percentage):

    rows, cols = image.shape[1:]
    rows_to_mask = int(rows * mask_percentage / 100)
    cols_to_mask = int(cols * mask_percentage / 100)

    top_left_row = random.randint(0, rows - rows_to_mask)
    top_left_col = random.randint(0, cols - cols_to_mask)

    image[:, top_left_row:top_left_row + rows_to_mask, top_left_col:top_left_col + cols_to_mask] = 0.5
    return image

# run if you want to apply masking
'''
output_dir = "./data/masked_pets_images"
os.makedirs(output_dir, exist_ok=True)

mask_proportion = 25 # in %

for i, (image, label) in enumerate(pets):
    image_pil = transforms.ToPILImage()(image)
    masked_image = mask_image(image, mask_proportion)
    masked_image = transforms.ToPILImage()(masked_image)
    # save the masked image
    filename = f"image_{i}_masked.jpg"
    image_path = os.path.join(output_dir, filename)
    masked_image.save(image_path)

print("Images saved successfully with masking.")
'''


# uncomment if you want to apply combination of transformations
'''
output_dir = "./data/transformed_pets_images"
os.makedirs(output_dir, exist_ok=True)
greyscale_proportion = 0.2 # from 0 to 1
mask_proportion = 25 # in %

for i, (image, label) in enumerate(pets):
    image_pil = transforms.ToPILImage()(image)
    im = greyscale(image_pil, greyscale_proportion)
    im = distort_color(im)
    im, rotation = rotate_image(im)

    # save the distorted image
    filename = f"image_{i}_transformed.jpg"
    image_path = os.path.join(output_dir, filename)
    im.save(image_path)

print("Transformed images saved successfully.")
'''

