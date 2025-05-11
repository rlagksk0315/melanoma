import torch
import os
from data_loading import load_ham_metadata, load_ddi_metadata, load_generated_metadata, split_data, get_dataloaders_8, get_pos_ratio
from models import get_efficientnet
from plot import plot_loss
from train import train
from evaluate import evaluate
import argparse
import pandas as pd

"""
Training for malignant/benign lesion classification on HAM+DDI+augmented dataset.
"""

parser = argparse.ArgumentParser(description='Train malignant/benign classifier')
parser.add_argument('--results_path', type=str, default='results/isalis/model_8', help='Path to save the results')
parser.add_argument('--num_epochs', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=1e-3)

args = parser.parse_args()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df_DDI = load_ddi_metadata('../data/ddi_cropped/ddi_metadata.csv')
    df_AUG = load_generated_metadata('../data/HAM10000/HAM10000_metadata.csv', '../data/darkHAM_unet') 
    test_ddi_df, val_ddi_df, train_ddi_df = split_data(df_DDI) # train: 20%, val: 10%, test: 70%
    pos_ratio = get_pos_ratio(pd.concat([train_ddi_df, df_AUG]))

    train_loader, val_loader, test_loader = get_dataloaders_8(df_AUG, train_ddi_df, val_ddi_df, test_ddi_df, '../data/darkHAM_resnet', '../data/ddi_cropped/images', batch_size=32)
    model = get_efficientnet(num_classes=1, pretrained=True)
    train_metrics_ddi = train(model, train_loader, val_loader, pos_ratio, device, results_path=args.results_path, epochs=args.num_epochs, lr=args.learning_rate)

    # plot loss
    plot_loss(train_metrics_ddi, args.results_path, "model_8_ddi")

    # evaluate best model
    best_ckpt = os.path.join(args.results_path, 'best_model.pth')
    state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    metrics_ddi = evaluate(model, test_loader, args.results_path, "model_8_ddi", device)
    print("Best model:")
    print(" DDI accuracy: ", metrics_ddi['accuracy'])
    print(" DDI precision: ", metrics_ddi['precision'])
    print(" DDI recall: ", metrics_ddi['recall'])
    print(" DDI f1: ", metrics_ddi['f1'])

if __name__ == "__main__":
    main()
