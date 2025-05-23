import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import itertools
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
from PIL import Image

class DDIDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None, skin_threshold=35, file_list=None):
        """
        Args:
            data_dir (str): file directory with all the DDI images.
            csv_file (str): path to the ddi_metadata.csv file with annotations.
            transform (callable): transform applied on a sample.
            skin_threshold (int): threshold to decide light or dark skin tone.
        """
        df = pd.read_csv(csv_file)
        if file_list is not None:
            df = df[df["DDI_file"].isin(file_list)]

        df = df[df["skin_tone"] > skin_threshold].reset_index(drop=True) # drops all skin tones that are below threshold
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        self.skin_threshold = skin_threshold

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row["DDI_file"])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img
    
class HAMDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None, file_list=None):
        """
        Args:
          data_dir (str): path to folder containing HAM10000 images
          csv_file (str): path to HAM10000_metadata.csv
          transform (callable): torchvision transforms to apply
          file_list (list[str]): list of filenames to include
        """
        self.data_dir  = data_dir
        self.transform = transform

        df = pd.read_csv(csv_file)
        df["image_id"] = df["image_id"].astype(str)
        self.meta = df.set_index("image_id")

        files = [
            f for f in os.listdir(data_dir)
            if os.path.splitext(f)[1].lower() == ".jpg"
        ]

        if file_list is not None:
            files = [f for f in files if f in file_list]
        self.image_files = sorted(files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        image_id, _ = os.path.splitext(fname)

        row = self.meta.loc[image_id]
        dx = row["dx"].strip().lower()
        
        img = Image.open(os.path.join(self.data_dir, fname)).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, image_id
    
class SCINDataset(Dataset): #SCIN dataset --> filtered such that only darkskin images (monkscale USA 7-10) are included
    def __init__(self, data_dir, transform=None, file_list=None):
        """
        Args:
            data_dir (str): Path to SCIN image folder.
            transform (callable, optional): Image transformations.
        """
        self.data_dir = data_dir
        self.transform = transform
        raw_files = file_list if file_list is not None else os.listdir(data_dir)
        self.file_list = [
            f for f in raw_files
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.data_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img

    
# data splitting
# train: 60%, val: 20%, test: 20%
def split_data(data_dir, train_ratio=0.6, val_ratio=0.2):
    all_files = [
        f for f in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, f))
        and f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]  
    num_files = len(all_files)
    
    train_size = int(num_files * train_ratio)
    val_size = int(num_files * val_ratio)
    
    train_files = all_files[:train_size]
    val_files = all_files[train_size:train_size + val_size]
    test_files = all_files[train_size + val_size:]
    
    return train_files, val_files, test_files

# data loading with transformations initialized above
def load_data_ddi(data_dir, csv_file, file_list, transform, batch_size, skin_threshold=35):
    ds = DDIDataset(data_dir, csv_file, transform=transform, skin_threshold=skin_threshold, file_list=file_list)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

def load_data_ham(data_dir, csv_file, file_list, transform, batch_size):
    ds = HAMDataset(data_dir, csv_file, transform=transform, file_list=file_list)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

def load_data_scin(data_dir, file_list, transform, batch_size):
    ds = SCINDataset(data_dir, transform=transform, file_list=file_list)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

def display_image(image_tensor):
    # display tensor image
    image = transforms.ToPILImage()(image_tensor)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def get_dataloaders(ddi_data_dir, ham_data_dir, scin_data_dir, batch_size=32, num_workers=4, seed=42): #ADD IN SCIN DATASET
    # define transformations
    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomGrayscale(0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std =(0.229, 0.224, 0.225)),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std =(0.229, 0.224, 0.225)),
    ])

    ddi_label_file = os.path.join(ddi_data_dir, "ddi_metadata.csv")
    ham_label_file = os.path.join(ham_data_dir, "HAM10000_metadata.csv")
    
    # split the data into train, val, and test sets
    ddi_image_dir = os.path.join(ddi_data_dir, "images")
    ham_image_dir = os.path.join(ham_data_dir, "images")
    scin_image_dir = os.path.join(scin_data_dir, "images")
    train_files_ddi, val_files_ddi, test_files_ddi = split_data(ddi_image_dir)
    train_files_ham, val_files_ham, test_files_ham = split_data(ham_image_dir)
    train_files_scin, val_files_scin, test_files_scin = split_data(scin_image_dir)

    # create data loaders
    ddi_loader_train, ddi_loader_val, ddi_loader_test = [
        load_data_ddi(
            ddi_image_dir,
            ddi_label_file,
            files,
            transform,
            batch_size
        )
        for files, transform in [
            (train_files_ddi, train_transform),
            (val_files_ddi,   eval_transform),
            (test_files_ddi,  eval_transform),
        ]
    ]

    ham_loader_train, ham_loader_val, ham_loader_test = [
        load_data_ham(
            ham_image_dir,
            ham_label_file,
            files,
            transform,
            batch_size
        )
        for files, transform in [
            (train_files_ham, train_transform),
            (val_files_ham,   eval_transform),
            (test_files_ham,  eval_transform),
        ]
    ]

    scin_loader_train, scin_loader_val, scin_loader_test = [
        load_data_scin(
            scin_image_dir,
            files,
            transform,
            batch_size
        )
        for files, transform in [
            (train_files_scin, train_transform),
            (val_files_scin,   eval_transform),
            (test_files_scin,  eval_transform),
        ]
    ]

    return ddi_loader_train, ddi_loader_val, ddi_loader_test, ham_loader_train, ham_loader_val, ham_loader_test, scin_loader_train, scin_loader_val, scin_loader_test

# ================================================================================
# main function to load data
# ================================================================================

if __name__ == "__main__":
    ddi_train_loader, ddi_val_loader, ddi_test_loader, ham_train_loader, ham_val_loader, ham_test_loader, scin_train_loader, scin_val_loader, scin_test_loader = get_dataloaders(ddi_data_dir, ham_data_dir, scin_data_dir, batch_size)
    print('Data loading complete!')

    '''
    # display some images
    for images, skin_labels, disease_labels in ddi_loader_train:
        for i in range(4):
            display_image(images[i])
            print(f"Skin Tone: {skin_labels[i]}, Malignant: {disease_labels[i]}")
        break
    '''