import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import os

class MelanomaDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 1])
        image = read_image(img_name)
        label = self.dataframe.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        return image, label
