import torch
import os
from data_loading import load_ham_metadata, load_ddi_metadata, split_data, get_dataloaders, get_pos_ratio
from models import get_efficientnet
from train import train
from plot import plot_loss
from evaluate import evaluate
import argparse

"""
Training for malignant/benign lesion classification on HAM10000 dataset.
"""
parser = argparse.ArgumentParser(description='Train malignant/benign classifier')
parser.add_argument('--results_path', type=str, default='results/isalis/model_1', help='Path to save the results')
parser.add_argument('--num_epochs', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=1e-3)

args = parser.parse_args()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df_HAM = load_ham_metadata('../data/HAM10000/HAM10000_metadata.csv')
    df_DDI = load_ddi_metadata('../data/ddi_cropped/ddi_metadata.csv')
    train_ham_df, val_ham_df, test_ham_df = split_data(df_HAM) # train: 70%, val: 10%, test: 20%
    test_ddi_df, val_ddi_df, train_ddi_df = split_data(df_DDI) # train: 20%, val: 10%, test: 70% - to test on same "batch" of images as the other models
    pos_ratio = get_pos_ratio(train_ham_df)

    train_ham_loader, val_ham_loader, test_ham_loader = get_dataloaders(train_ham_df, val_ham_df, test_ham_df, '../data/HAM10000/images', batch_size=32)
    train_ddi_loader, val_ddi_loader, test_ddi_loader = get_dataloaders(train_ddi_df, val_ddi_df, test_ddi_df, '../data/ddi_cropped/images', batch_size=32)
    model = get_efficientnet(num_classes=2, pretrained=True)
    train_metrics = train(model, train_ham_loader, val_ham_loader, pos_ratio, device, results_path=args.results_path, epochs=args.num_epochs, lr=args.learning_rate)

    # plot loss
    plot_loss(train_metrics, args.results_path, "model_1")

    # evaluate best model
    best_ckpt = os.path.join(args.results_path, 'best_model.pth')
    state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    metrics_ham = evaluate(model, test_ham_loader, args.results_path, "model_1", device)
    metrics_ddi = evaluate(model, test_ddi_loader, args.results_path, "model_1", device)
    print("Best model:")
    print(" DDI accuracy: ", metrics_ddi['accuracy'])
    print(" DDI precision: ", metrics_ddi['precision'])
    print(" DDI recall: ", metrics_ddi['recall'])
    print(" DDI f1: ", metrics_ddi['f1'])
    print(" HAM accuracy: ", metrics_ham['accuracy'])
    print(" HAM precision: ", metrics_ham['precision'])
    print(" HAM recall: ", metrics_ham['recall'])
    print(" HAM f1: ", metrics_ham['f1'])

if __name__ == "__main__":
    main()
