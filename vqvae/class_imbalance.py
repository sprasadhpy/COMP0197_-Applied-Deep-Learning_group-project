import os
import shutil
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
import glob
import torchvision.transforms as transforms
import torchvision
import torch

class PetsDataset(Dataset):
    def __init__(
        self,
        df,
        label_to_id,
        image_path=None,
        image_transforms=None,
    ):
        self.df = df
        self.image_path = Path(image_path) if image_path is not None else None
        self.image_transforms = image_transforms
        self.label_to_id = label_to_id
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        image_path = data.image

        if self.image_path is not None:
            image_path = self.image_path / image_path

        raw_image = Image.open(image_path)
        image = raw_image.convert("RGB")

        if self.image_transforms is not None:
            image = self.image_transforms(image)

        raw_label = data.label
        label = self.label_to_id[raw_label]

        return image, label

def delete_images(folder_path, Class_type, keep_class):
    for class_type in Class_type:
        if class_type not in keep_class:
            # Get all images in the directory
            # print(f'{folder_path}/{class_type}*')
            images = glob.glob(f'{folder_path}/{class_type}*')
            # print(images)
            for image in images:
                # Get the image name without the extension
                image_name = os.path.basename(image).split('.')[0]
                # If the image name starts with the class type, delete it
                if image_name.startswith(class_type):
                    os.remove(image)
def process_file(file_path, temp_file_path, Class_type, keep_class):
    # Temporary file to store the lines we want to keep
    with open(temp_file_path, 'w') as temp_file:
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    continue

                # Get the label at the start of the line
                label = line.split()[0]

                label = label.split('_')[0]

                # If the label is in Class_type and not in keep_class, skip this line
                if label in Class_type and label not in keep_class:
                    continue
                # Otherwise, write the line to the temporary file
                # print(label)
                temp_file.write(line)

    # Replace the original file with the temporary file
    os.replace(temp_file_path, file_path)

def get_imbalance_dataset():
    # Delete the existing imbalanced dataset if it exists and create a new one
    if os.path.exists('./data/oxford-pets-imbalanced'):
        shutil.rmtree('./data/oxford-pets-imbalanced')
    # Make a copy of the original dataset
    shutil.copytree('./data/oxford-pets', './data/oxford-pets-imbalanced')

    # Create a new dataset with imbalanced classes
    # Get the list of all image classes in the dataset
    classes = os.listdir('./data/oxford-pets-imbalanced/oxford-iiit-pet/images')

    labels = set()
    for label in classes:
        obj = label.split('_')[0]
        labels.add(obj)

    Dogs = []
    Cats = []

    for label in labels:
        if label[0].isupper():
            Cats.append(label)
        else:
            Dogs.append(label)

    print('Cat classes:', Cats)
    print('Dog classes:', Dogs)
    # Define the two cat classes you want to keep
    keep_cats = ['Maine', 'Birman', 'Bombay']



    delete_images('./data/oxford-pets-imbalanced/oxford-iiit-pet/images', Cats, keep_cats)

    #delete xmls
    delete_images('./data/oxford-pets-imbalanced/oxford-iiit-pet/annotations/xmls', Cats, keep_cats)

    #delete trimaps
    delete_images('./data/oxford-pets-imbalanced/oxford-iiit-pet/annotations/trimaps', Cats, keep_cats)

    temp_path = './data/oxford-pets-imbalanced/oxford-iiit-pet/annotations/temp.txt'
    process_file('./data/oxford-pets-imbalanced/oxford-iiit-pet/annotations/list.txt', temp_path, Cats, keep_cats)

    # process trainval.txt
    temp_path = './data/oxford-pets-imbalanced/oxford-iiit-pet/annotations/temp.txt'
    process_file('./data/oxford-pets-imbalanced/oxford-iiit-pet/annotations/trainval.txt', temp_path, Cats, keep_cats)

    # process test.txt
    temp_path = './data/oxford-pets-imbalanced/oxford-iiit-pet/annotations/temp.txt'
    process_file('./data/oxford-pets-imbalanced/oxford-iiit-pet/annotations/test.txt', temp_path, Cats, keep_cats)

    # load the dataset
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

    dataset = torchvision.datasets.OxfordIIITPet(
        root='./data/oxford-pets-imbalanced',
        transform=transform,
        target_types="segmentation",
        target_transform=target_transform,
        download=False
    )



    return dataset


# create_imbalance_dataset()