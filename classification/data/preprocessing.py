import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from torchvision import transforms

def load_metadata(csv_path):
    df = pd.read_csv(csv_path)
    df['image_path'] = df['image_id'] + '.jpg'
    df['label'] = (df['dx'] == 'mel').astype(int)
    df = df.drop(columns=['image_id', 'dx', 'dx_type', 'age', 'sex', 'localization'])
    return df

def split_data(df, group_col='lesion_id', test_size=0.2, val_size=0.125, random_state=42):
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(gss1.split(df, groups=df[group_col]))
    trainval_df = df.iloc[trainval_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(gss2.split(trainval_df, groups=trainval_df[group_col]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, test_df

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_dataloaders(train_df, val_df, test_df, root_dir, batch_size=32):
    from .dataset import MelanomaDataset
    train_dataset = MelanomaDataset(train_df, root_dir, transform=get_transforms())
    val_dataset = MelanomaDataset(val_df, root_dir, transform=get_transforms())
    test_dataset = MelanomaDataset(test_df, root_dir, transform=get_transforms())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def get_class_ratio(df):
    num_pos = (df['label'] == 1).sum()
    num_neg = (df['label'] == 0).sum()
    ratio = num_neg/num_pos
    return ratio
