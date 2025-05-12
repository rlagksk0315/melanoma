import torch
import os
from data_loading import load_ddi_metadata, split_data, get_dataloaders, get_pos_ratio
from models import get_efficientnet
from plot import plot_loss
from train import train
from evaluate import evaluate
import argparse

"""
Training for malignant/benign lesion classification on DDI dataset.
"""

parser = argparse.ArgumentParser(description='Train malignant/benign classifier')
parser.add_argument('--results_path', type=str, default='../results/model_2', help='Path to save the results')
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=1e-3)


args = parser.parse_args()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df_DDI = load_ddi_metadata('../data/ddi_cropped/ddi_metadata.csv')
    train_df, val_df, test_df = split_data(df_DDI) # train: 70%, val: 10%, test: 20%
    pos_ratio = get_pos_ratio(train_df)

    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df, '../data/ddi_cropped/images', batch_size=32)
    model = get_efficientnet(num_classes=1, pretrained=True)
    train_metrics = train(model, train_loader, val_loader, pos_ratio, device, results_path=args.results_path, epochs=args.num_epochs, lr=args.learning_rate)

    # plot loss
    plot_loss(train_metrics, args.results_path, "model_2")

    # evaluate best model
    best_ckpt = os.path.join(args.results_path, 'best_model.pth')
    state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    metrics_ddi = evaluate(model, test_loader, args.results_path, "model_2", device)
    print("Best model:")
    print(" DDI accuracy: ", metrics_ddi['accuracy'])
    print(" DDI precision: ", metrics_ddi['precision'])
    print(" DDI recall: ", metrics_ddi['recall'])
    print(" DDI f1: ", metrics_ddi['f1'])

if __name__ == "__main__":
    main()
