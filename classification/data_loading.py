import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupShuffleSplit
from torchvision import transforms
from PIL import Image

class ClassificationDataset(Dataset):
    def __init__(self, data_dir, df, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row["image_path"])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(row["label"])
        return img, label

def load_ham_metadata(csv_path):
    df = pd.read_csv(csv_path)
    df['image_path'] = df['image_id'] + '.jpg'
    df['label'] = (df['dx'] == 'mel').astype(int)
    df = df.rename(columns={'lesion_id':'id'})
    df = df.drop(columns=['image_id', 'dx', 'dx_type', 'age', 'sex', 'localization'])
    return df

def load_ddi_metadata(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    df['image_path'] = df['DDI_file']
    df['label'] = df['malignant'].astype(int)
    df = df.rename(columns={'DDI_ID':'id'})
    df = df.drop(columns=['DDI_file', 'skin_tone', 'malignant', 'disease'])
    return df

def load_generated_metadata(csv_path, images_path):
    df = pd.read_csv(csv_path)
    generated_images = set(os.listdir(images_path))
    original_images = {img.split("_dark")[0] for img in generated_images if "_dark" in img}
    df = df[df['image_id'].isin(original_images)].reset_index(drop=True)
    df['image_path'] = df['image_id'] + '.jpg'
    df['label'] = (df['dx'] == 'mel').astype(int)
    df = df.rename(columns={'lesion_id':'id'})
    df = df.drop(columns=['image_id', 'dx', 'dx_type', 'age', 'sex', 'localization'])
    return df

def get_transforms():
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

    return train_transform, eval_transform

def group_split(df, group_col, test_size, val_size, random_state):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(gss.split(df, groups=df[group_col]))
    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(gss2.split(trainval_df, groups=trainval_df[group_col]))
    train_df = trainval_df.iloc[train_idx].reset_index(drop=True)
    val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, test_df

def split_data(df, group_col='id', label_col='label', test_size=0.2, val_size=0.125, random_state=42):
    df_benign = df[df[label_col] == 0]
    df_malignant = df[df[label_col] == 1]

    train_0, val_0, test_0 = group_split(df_benign, group_col, test_size, val_size, random_state)
    train_1, val_1, test_1 = group_split(df_malignant, group_col, test_size, val_size, random_state)

    train_df = pd.concat([train_0, train_1]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = pd.concat([val_0, val_1]).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.concat([test_0, test_1]).sample(frac=1, random_state=42).reset_index(drop=True)

    return train_df, val_df, test_df

def get_dataloaders(train_df, val_df, test_df, root_dir, batch_size=32):
    train_transform, eval_transform = get_transforms()

    train_dataset = ClassificationDataset(root_dir, train_df, transform=train_transform)
    val_dataset = ClassificationDataset(root_dir, val_df, transform=eval_transform)
    test_dataset = ClassificationDataset(root_dir, test_df, transform=eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_pos_ratio(df):
    num_pos = (df['label'] == 1).sum()
    num_neg = (df['label'] == 0).sum()
    pos_ratio = num_neg/num_pos
    return pos_ratio

def main():
    ham_dir = "../data/HAM10000"
    ddi_dir = "../data/ddi_cropped"
    generated_dir = "../data/darHAM"

    ham_df = load_ham_metadata(ham_dir+'/HAM10000_metadata.csv')
    df_DDI = load_ddi_metadata(ddi_dir+'/ddi_metadata.csv')
    generated_df = load_generated_metadata(generated_dir+'/HAM10000_metadata.csv')

    print('Data loading complete!')

if __name__ == "__main__":
    main()