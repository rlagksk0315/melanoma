import torch
import os
from data.preprocessing import load_ham_metadata, split_data, get_dataloaders_1, get_pos_ratio
from models.efficientnet import get_efficientnet
from train import train
from evaluate import evaluate
import matplotlib.pyplot as plt
import argparse

"""
Training for malignant/benign lesion classification on HAM10000 dataset.
"""
parser = argparse.ArgumentParser(description='Train malignant/benign classifier')
parser.add_argument('--results_path', type=str, default='results/(your_name)', help='Path to save the results')

args = parser.parse_args()

def plot_loss(train_metrics, results_path):
    os.makedirs(results_path, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(train_metrics['train_losses'], label='Train Loss')
    plt.plot(train_metrics['val_losses'], label='Val Loss')
    plt.title('Loss Over Epochs for Train and Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()    
    plt.savefig(f'{results_path}/loss_curves.png', bbox_inches='tight', dpi=300)
    plt.close()

def main_1():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #TODO: change the load_ham_metadata function to the right name
    df_HAM = load_ham_metadata('../data/HAM10000/HAM10000_metadata.csv')
    train_df, val_df, test_df = split_data(df_HAM)
    pos_ratio = get_pos_ratio(train_df)

    #TODO: change the get_dataloaders_1 function to the right name
    train_loader, val_loader, test_loader = get_dataloaders_1(train_df, val_df, test_df, '../data/HAM10000/images', batch_size=32)
    model = get_efficientnet(num_classes=2, pretrained=True)
    train_metrics = train(model, train_loader, val_loader, pos_ratio, device, epochs=10, lr=1e-3)

    #TODO: add appropriate results_path when running the code
    plot_loss(train_metrics, args.results_path)

    # Save the model - changed code so that it would save in the same results_path
    #TODO: change the model path to the right name
    # model.load_state_dict(torch.load('models/(your_name)/model_best_loss.pth', map_location=device)['model_state_dict'])
    # test_metrics1 = evaluate(model, test_loader, device)
    # print(test_metrics1)

    # model.load_state_dict(torch.load('models/(your_name)/model_best_acc.pth', map_location=device)['model_state_dict'])
    # test_metrics2 = evaluate(model, test_loader, device)
    # print(test_metrics2)
    best_loss_ckpt = os.path.join(args.results_path, 'model_best_loss.pth')
    state = torch.load(best_loss_ckpt, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    print("Best-loss: ", evaluate(model, test_loader, device))

    # evaluate checkpoint with best validation accuracy
    best_acc_ckpt = os.path.join(args.results_path, 'model_best_acc.pth')
    state = torch.load(best_acc_ckpt, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    print("Best-acc: ", evaluate(model, test_loader, device))

if __name__ == "__main__":
    main_1()
