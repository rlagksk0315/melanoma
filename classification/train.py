import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

def train(model, train_loader, val_loader, class_ratio, device, epochs=10, lr=1e-3, save_path='models/best_model.pth'):
    class_weights = torch.tensor([class_ratio], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.unsqueeze(1).float().to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(val_loader):
                data = data.to(device)
                targets = targets.unsqueeze(1).float().to(device)
                outputs = model(data)
                probs = torch.sigmoid(outputs).squeeze()
                preds = (probs > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(targets.cpu().numpy())
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc,
            }, save_path)
            print(f"Best model saved at epoch {best_epoch} with val_acc: {best_val_acc:.4f}")

    print(f"Training complete. Best model at epoch {best_epoch} with val_acc: {best_val_acc:.4f}")

