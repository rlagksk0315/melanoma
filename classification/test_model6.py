import torch
import os
from data_loading import load_ham_metadata, load_ddi_metadata, load_generated_metadata, split_data, get_dataloaders_4, get_pos_ratio
from models import get_efficientnet
from main_1 import plot_loss
from train import train_3
from evaluate import evaluate
import argparse
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#TODO: change the load_ham_metadata function to the right name
df_HAM = load_ham_metadata('../data/HAM10000/HAM10000_metadata.csv')
df_DDI = load_ddi_metadata('../data/ddi_cropped/ddi_metadata.csv')
df_AUG = load_generated_metadata('../data/HAM10000/HAM10000_metadata.csv', '../data/darkHAM_unet')   # TODO: change the path to the right name 
train_ham_df, val_ham_df, test_ham_df = split_data(df_HAM)
test_ddi_df, val_ddi_df, train_ddi_df = split_data(df_DDI)
df_AUG['image_id'] = df_AUG['image_path'].apply(lambda x: x.split('_fake_Y')[0]) + '.jpg'
train_aug_df = df_AUG[(df_AUG['image_id']).isin(train_ham_df['image_path'])][['id','image_path','label']]
pos_ratio = get_pos_ratio(pd.concat([train_ham_df, train_ddi_df, train_aug_df]))

#TODO: change the get_dataloaders_1 function to the right name
(train_loader,
val_ham_loader, test_ham_loader,
val_ddi_loader, test_ddi_loader) = get_dataloaders_4(train_ham_df, val_ham_df, test_ham_df,
                                                    train_ddi_df, val_ddi_df, test_ddi_df,
                                                    train_aug_df,
                                                    '../data/HAM10000/images',
                                                    '../data/ddi_cropped/images',
                                                    '../data/darkHAM_unet',
                                                    batch_size=32)

model = get_efficientnet(num_classes=1)

best_loss_ckpt = os.path.join('/projects/pancreas-cancer-hpc/hana-eus/melanoma/results/hana/main_6_classification', 'model_best_loss.pth')
state = torch.load(best_loss_ckpt, map_location=device)
model.load_state_dict(state['model_state_dict'])
model.to(device)
print("Best-loss DDI: ", evaluate(model, test_ddi_loader, '/projects/pancreas-cancer-hpc/hana-eus/melanoma/results/hana/main_6_classification', "loss_model_6_ddi", device))
print("Best-loss HAM: ", evaluate(model, test_ham_loader, '/projects/pancreas-cancer-hpc/hana-eus/melanoma/results/hana/main_6_classification', "loss_model_6_ddi", device))

# evaluate checkpoint with best validation accuracy
best_acc_ckpt = os.path.join('/projects/pancreas-cancer-hpc/hana-eus/melanoma/results/hana/main_6_classification', 'model_best_acc.pth')
state = torch.load(best_acc_ckpt, map_location=device)
model.load_state_dict(state['model_state_dict'])
model.to(device)
print("Best-acc DDI: ", evaluate(model, test_ddi_loader, '/projects/pancreas-cancer-hpc/hana-eus/melanoma/results/hana/main_6_classification', "acc_model_6_ddi", device))
print("Best-acc HAM: ", evaluate(model, test_ham_loader, '/projects/pancreas-cancer-hpc/hana-eus/melanoma/results/hana/main_6_classification', "acc_model_6_ham", device))