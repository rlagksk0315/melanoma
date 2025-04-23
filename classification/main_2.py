import torch
import os
from data_loading import load_ddi_metadata, split_data, get_dataloaders, get_pos_ratio
from models import get_efficientnet
from main_1 import plot_loss
from train import train
from evaluate import evaluate
import argparse

"""
Training for malignant/benign lesion classification on DDI dataset.
"""

parser = argparse.ArgumentParser(description='Train malignant/benign classifier')
parser.add_argument('--results_path', type=str, default='results/(your_name)', help='Path to save the results')
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=1e-3)


args = parser.parse_args()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #TODO: change the load_ham_metadata function to the right name
    df_DDI = load_ddi_metadata('../data/ddi_cropped/ddi_metadata.csv')
    train_df, val_df, test_df = split_data(df_DDI)
    pos_ratio = get_pos_ratio(train_df)

    #TODO: change the get_dataloaders_1 function to the right name
    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df, '../data/ddi_cropped/images', batch_size=32)
    model = get_efficientnet(num_classes=1, pretrained=True)
    train_metrics = train(model, train_loader, val_loader, pos_ratio, device, epochs=args.num_epochs, lr=args.learning_rate, results_path=args.results_path)

    #TODO: add appropriate results_path when running the code
    plot_loss(train_metrics, args.results_path, "model2")

    # Save the model 
    #TODO: change the model path to the right name
    best_loss_ckpt = os.path.join(args.results_path, 'model2_best_loss.pth')
    state = torch.load(best_loss_ckpt, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    print("Best-loss: ", evaluate(model, test_loader, device))

    # evaluate checkpoint with best validation accuracy
    best_acc_ckpt = os.path.join(args.results_path, 'model2_best_acc.pth')
    state = torch.load(best_acc_ckpt, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    print("Best-acc: ", evaluate(model, test_loader, device))

if __name__ == "__main__":
    main()
