import torch
import os
from data_loading import load_ham_metadata, load_ddi_metadata, load_generated_metadata, split_data, get_dataloaders_4, get_pos_ratio
from models import get_efficientnet
from main_1 import plot_loss
from train import train_3
from evaluate import evaluate
import argparse
import pandas as pd

"""
Training for malignant/benign lesion classification on HAM+DDI+augmented dataset.
"""

parser = argparse.ArgumentParser(description='Train malignant/benign classifier')
parser.add_argument('--results_path', type=str, default='results/(your_name)/model_5', help='Path to save the results')
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=1e-3)

args = parser.parse_args()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #TODO: change the load_ham_metadata function to the right name
    df_HAM = load_ham_metadata('../data/HAM10000/HAM10000_metadata.csv')
    df_DDI = load_ddi_metadata('../data/ddi_cropped/ddi_metadata.csv')
    df_AUG = load_generated_metadata('../data/HAM10000/HAM10000_metadata.csv', '../data/darkHAM2')   # TODO: change the path to the right name 
    train_ham_df, val_ham_df, test_ham_df = split_data(df_HAM)
    train_ddi_df, val_ddi_df, test_ddi_df = split_data(df_DDI)
    train_aug_df, val_aug_df, test_aug_df = split_data(df_AUG)
    pos_ratio = get_pos_ratio(pd.concat([train_ham_df, train_ddi_df, train_aug_df]))

    #TODO: change the get_dataloaders_1 function to the right name
    (train_loader,
     val_ham_loader, test_ham_loader,
     val_ddi_loader, test_ddi_loader) = get_dataloaders_4(train_ham_df, val_ham_df, test_ham_df,
                                                          train_ddi_df, val_ddi_df, test_ddi_df,
                                                          train_aug_df, val_aug_df, test_aug_df,
                                                          '../data/HAM10000/images',
                                                          '../data/ddi_cropped/images',
                                                          '../data/darkHAM2',
                                                          batch_size=32)
    model = get_efficientnet(num_classes=1, pretrained=True)
    train_metrics_ham, train_metrics_ddi = train_3(model, train_loader, val_ham_loader, val_ddi_loader, pos_ratio, device, epochs=args.num_epochs, lr=args.learning_rate)

    #TODO: add appropriate results_path when running the code
    plot_loss(train_metrics_ham, args.results_path, "model_5_ham")
    plot_loss(train_metrics_ddi, args.results_path, "model_5_ddi")

    # Save the model 
    #TODO: change the model path to the right name
    best_loss_ckpt = os.path.join(args.results_path, 'model_best_loss.pth')
    state = torch.load(best_loss_ckpt, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    print("Best-loss DDI: ", evaluate(model, test_ddi_loader, args.results_path, "loss_model_5_ddi", device))
    print("Best-loss HAM: ", evaluate(model, test_ham_loader, args.results_path, "loss_model_5_ddi", device))

    # evaluate checkpoint with best validation accuracy
    best_acc_ckpt = os.path.join(args.results_path, 'model_best_acc.pth')
    state = torch.load(best_acc_ckpt, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    print("Best-acc DDI: ", evaluate(model, test_ddi_loader, args.results_path, "acc_model_5_ddi", device))
    print("Best-acc HAM: ", evaluate(model, test_ham_loader, args.results_path, "acc_model_5_ham", device))

if __name__ == "__main__":
    main()
