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
    def __init__(self, data_dir, csv_file, transform=None, skin_threshold=35):
        """
        Args:
            data_dir (str): file directory with all the DDI images.
            csv_file (str): path to the ddi_metadata.csv file with annotations.
            transform (callable): transform applied on a sample.
            skin_threshold (int): threshold to decide light or dark skin tone.
        """
        self.data_dir = data_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.skin_threshold = skin_threshold

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row["DDI_file"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        skin_label = "light" if row["skin_tone"] < self.skin_threshold else "dark"
        malignant = row["malignant"]
        
        return {"image": image, 
                "skin_tone": skin_label, 
                "malignant": malignant}
    
    class HAMDataset(Dataset):
        """
        Args:
            data_dir (str): path to folder containing HAM10000 images.
            csv_file (str): path to HAM10000_metadata.csv.
            transform (callable, optional): torchvision transforms to apply.
        """
        def __init__(self, data_dir, csv_file, transform=None):
            self.data_dir = data_dir
            self.transform = transform
            
            df = pd.read_csv(csv_file)
            df["image_id"] = df["image_id"].astype(str)
            self.meta = df.set_index("image_id")
            
            self.image_files = [
                fname for fname in os.listdir(data_dir)
                if os.path.splitext(fname)[1].lower() in {".jpg", ".jpeg", ".png"}
            ]
        
        def __len__(self):
            return len(self.image_files)
        
        def __getitem__(self, idx):
            fname = self.image_files[idx]
            image_id, _ = os.path.splitext(fname)
            
            row = self.meta.loc[image_id]
            lesion_id = row["lesion_id"]
            dx = row["dx"].strip().lower()
            malignant = (dx == "melanoma")
            
            img_path = os.path.join(self.data_dir, fname)
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            
            return {
                "image": image,
                "lesion_id": lesion_id,
                "malignant": malignant
            }

# load ddi and HAM10000 data
ddi_data_dir = "../data/ddi_cropped"
os.listdir(ddi_data_dir)
print(len(os.listdir(ddi_data_dir)))

ham_data_dir = "../data/HAM10000"
os.listdir(ham_data_dir)
print(len(os.listdir(ham_data_dir)))

# label the data according to their information
ddi_label_file = os.path.join(ddi_data_dir, "ddi_metadata.csv")
ham_label_file = os.path.join(ham_data_dir, "HAM10000_metadata.csv")
df_ddi = pd.read_csv(ddi_label_file)
df_ham = pd.read_csv(ham_label_file)
print("ddi description file:\n", df_ddi.head())
print("___________________________________")
print("ham description file:\n", df_ham.head())

# data splitting
# train: 60%, val: 20%, test: 20%
def split_data(data_dir, train_ratio=0.6, val_ratio=0.2):
    all_files = os.listdir(data_dir)
    num_files = len(all_files)
    
    train_size = int(num_files * train_ratio)
    val_size = int(num_files * val_ratio)
    
    train_files = all_files[:train_size]
    val_files = all_files[train_size:train_size + val_size]
    test_files = all_files[train_size + val_size:]
    
    return train_files, val_files, test_files

train_files_ddi, val_files_ddi, test_files_ddi = split_data(ddi_data_dir)
train_files_ham, val_files_ham, test_files_ham = split_data(ham_data_dir)

print("number of train images for ddi:", len(train_files_ddi))
print("number of val images for ddi:", len(val_files_ddi))
print("number of test images for ddi:", len(test_files_ddi))
print("------------------------------")
print("number of train images for ham:", len(train_files_ham))
print("number of val images for ham:", len(val_files_ham))
print("number of test images for ham:", len(test_files_ham))

def display_image(image):
    # Convert the tensor to a PIL image
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def load_data(batch_size):
    transform = transforms.Compose([
        transforms.Resize((224,224)), # Resizes (32,32) to (224,224)
        # transforms.RandomCrop((220,220)), # Takes a random (32,32) crop
        # transforms.ColorJitter(brightness=0.5), # Change brightness of image
        transforms.RandomRotation(degrees=45), # Perhaps a random rotation from -45 to 45 degrees
        # transforms.RandomHorizontalFlip(p=0.5), # Flips the image horizontally with probability 0.5
        # transforms.RandomVerticalFlip(p=0.05), # Flips image vertically with probability 0.05
        transforms.RandomGrayscale(p=0.2), # Converts to grayscale with probability 0.2
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizes between -1 and 1
    ])
    # Load the dataset
    ddi_dataset = datasets.ImageFolder(ddi_data_dir, transform=transform)
    ham_dataset = datasets.ImageFolder(ham_data_dir, transform=transform)
    # Create data loaders
    train_loader = DataLoader(ddi_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ham_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ham_dataset, batch_size=batch_size, shuffle=True)
    # Display a sample image from the dataset
    sample_image, _ = next(iter(train_loader))
    display_image(sample_image[0])  # Display the first image in the batch
    return train_loader, val_loader, test_loader