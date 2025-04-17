import torch
import os
from data.preprocessing import load_metadata, split_data, get_dataloaders, get_class_ratio
from models.efficientnet import get_efficientnet
from train import train
from evaluate import evaluate

def main():
    os.chdir('/scratch4/en520-lmorove1/en520-ikarhul1/melanoma/classification')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = load_metadata('../data/HAM10000/HAM10000_metadata.csv')
    train_df, val_df, test_df = split_data(df)
    class_ratio = get_class_ratio(train_df)
    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df, '../data/HAM10000/images', batch_size=32)
    model = get_efficientnet(num_classes=2, pretrained=True)
    train(model, train_loader, val_loader, class_ratio, device, epochs=10, lr=1e-3)
    metrics = evaluate(model, test_loader, device)
    print(metrics)

if __name__ == "__main__":
    main()
