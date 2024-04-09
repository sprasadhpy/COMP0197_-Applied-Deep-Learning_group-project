
import os 
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from urllib.request import urlretrieve
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


class OxfordPetDataset(Dataset):
    def __init__(self, images_filenames, images_directory, masks_directory, transform=None, transform_mask=None):
        self.images_filenames = images_filenames
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.transform = transform
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames.loc[idx] + '.jpg' 
        image = Image.open(os.path.join(self.images_directory, image_filename)).convert('RGB')
        mask = Image.open(
            os.path.join(self.masks_directory, image_filename.replace(".jpg", ".png")))
        #mask = preprocess_mask(mask)
        if self.transform is not None:
            transformed = self.transform(image)
            transformed_m = self.transform_mask(mask)
            image = transformed
            mask = transformed_m
        return image, mask


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        print("Dataset already exists on the disk. Skipping download.")
        return

    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=os.path.basename(filepath)) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    shutil.unpack_archive(filepath, extract_dir)


def merge_trainval_test(filepath):
    """
        #   Image CLASS-ID SPECIES BREED ID
        #   ID: 1:37 Class ids
        #   SPECIES: 1:Cat 2:Dog
        #   BREED ID: 1-25:Cat 1:12:Dog
        #   All images with 1st letter as captial are cat images
        #   images with small first letter are dog images
    """
    merge_dir = os.path.dirname(os.path.abspath(f'{filepath}/annotations/data.txt'))
    #if os.path.exists(merge_dir):
    #    print("Merged data is already exists on the disk. Skipping creating new data file.")
    #    return
    df = pd.read_csv(f"{filepath}/annotations/trainval.txt", sep=" ", 
                     names=["Image", "ID", "SPECIES", "BREED ID"])
    df2 = pd.read_csv(f"{filepath}/annotations/test.txt", sep=" ",
                      names=["Image", "ID", "SPECIES", "BREED ID"])
    frame = [df, df2]
    df = pd.concat(frame)
    df.reset_index(drop=True)
    df.to_csv(f'{filepath}/annotations/data.txt', index=None, sep=' ')
    print("Merged data is created.")

def get_data_loader(BATCH_SIZE):
    dataset_directory = os.path.join("./dataset")
    if not os.path.exists(dataset_directory):
        os.mkdir(dataset_directory)
    filepath = os.path.join(dataset_directory, "images.tar.gz")
    download_url(
        url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz", filepath=filepath,
    )
    extract_archive(filepath)
    filepath = os.path.join(dataset_directory, "annotations.tar.gz")
    download_url(
        url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz", filepath=filepath,
    )
    extract_archive(filepath)
    filepath = os.path.join(dataset_directory)
    merge_trainval_test(filepath)
    dataset = pd.read_csv(f"{filepath}/annotations/data.txt", sep=" ")

    image_ids = []
    labels = []
    with open(f"{filepath}/annotations/trainval.txt") as file:
        for line in file:
            image_id, label, *_ = line.strip().split()
            image_ids.append(image_id)
            labels.append(int(label)-1)

    classes = [
        " ".join(part.title() for part in raw_cls.split("_"))
        for raw_cls, _ in sorted(
            {(image_id.rsplit("_", 1)[0], label) for image_id, label in zip(image_ids, labels)},
            key=lambda image_id_and_label: image_id_and_label[1],
        )
        ]

    idx_to_class = dict(zip(range(len(classes)), classes))

    dataset['nID'] = dataset['ID'] - 1

    decode_map = idx_to_class
    
    def decode_label(label):
        return decode_map[int(label)]

    dataset["class"] = dataset["nID"].apply(lambda x: decode_label(x))

    y = dataset['class']
    x = dataset['Image']

    trainval, x_test, y_trainval, y_test = train_test_split(x, y,
                                                        stratify=y, 
                                                        test_size=0.2,
                                                        random_state=42)

    x_train, x_val, y_train, y_val = train_test_split(  trainval, y_trainval,
                                                        stratify=y_trainval, 
                                                        test_size=0.3,
                                                        random_state=42)

    train_transform = transforms.Compose([transforms.Resize((256, 256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    target_transform = transforms.Compose([transforms.PILToTensor(),     
                                transforms.Resize((256, 256)),     
                                transforms.Lambda(lambda x: (x-1).squeeze().type(torch.LongTensor)) ])

    root_directory = os.path.join(dataset_directory)
    images_directory = os.path.join(root_directory, "images")
    masks_directory = os.path.join(root_directory, "annotations", "trimaps")

    train_images_filenames = x_train.reset_index(drop=True)
    val_images_filenames = x_val.reset_index(drop=True)
    test_images_filenames = x_test.reset_index(drop=True)

    train_dataset = OxfordPetDataset(train_images_filenames,
                                 images_directory, 
                                 masks_directory, 
                                 transform=train_transform, 
                                 transform_mask=target_transform)



    val_dataset = OxfordPetDataset(val_images_filenames,
                                images_directory,
                                masks_directory,
                                transform=train_transform,
                                transform_mask=target_transform)

    test_dataset = OxfordPetDataset(test_images_filenames,
                                images_directory,
                                masks_directory,
                                transform=train_transform,
                                transform_mask=target_transform)


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader,  val_loader, test_loader

    