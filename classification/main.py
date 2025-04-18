import torch
import os
from data.preprocessing import load_metadata, split_data, get_dataloaders, get_pos_ratio
from models.efficientnet import get_efficientnet
from train import train
from evaluate import evaluate
import matplotlib.pyplot as plt

def main():
    os.chdir('/scratch4/en520-lmorove1/en520-ikarhul1/melanoma/classification')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = load_metadata('../data/HAM10000/HAM10000_metadata.csv')
    train_df, val_df, test_df = split_data(df)
    pos_ratio = get_pos_ratio(train_df)
    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df, '../data/HAM10000/images', batch_size=32)
    model = get_efficientnet(num_classes=2, pretrained=True)
    train_metrics = train(model, train_loader, val_loader, pos_ratio, device, epochs=10, lr=1e-3)
    
    plt.figure(figsize=(12, 5))
    plt.plot(train_metrics['train_losses'], label='Train Loss')
    plt.plot(train_metrics['val_losses'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()    
    plt.savefig('training_curves.png', bbox_inches='tight', dpi=300)
    plt.close()

    model.load_state_dict(torch.load('models/model_best_loss.pth', map_location=device)['model_state_dict'])
    test_metrics1 = evaluate(model, test_loader, device)
    print(test_metrics1)

    model.load_state_dict(torch.load('models/model_best_acc.pth', map_location=device)['model_state_dict'])
    test_metrics2 = evaluate(model, test_loader, device)
    print(test_metrics2)

if __name__ == "__main__":
    main()
