import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

def train(model, train_loader, val_loader, pos_ratio, device, epochs=10, lr=1e-3, save_path='models/model_best_loss.pth', save_path2='models/model_best_acc.pth'):
    pos_weight = torch.tensor([pos_ratio], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []

    min_val_loss = float('inf')
    max_val_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        epoch_train_loss = 0.0
        train_correct = 0
        total_samples = 0
        
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            data, targets = data.to(device), targets.float().to(device)
            
            outputs = model(data).squeeze(1)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_acc = train_correct / total_samples
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, targets in tqdm(val_loader, desc=f"Validation"):
                data, targets = data.to(device), targets.float().to(device)
                
                outputs = model(data).squeeze(1)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save({'epoch': best_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': min_val_loss}, save_path)
            print(f"Best model saved at epoch {best_epoch} with val_loss: {min_val_loss:.4f}")

        if val_acc > max_val_acc:
            max_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({'epoch': best_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': max_val_acc}, save_path2)
            print(f"Best model saved at epoch {best_epoch} with val_acc: {max_val_acc:.4f}")
            

    print(f"\nTraining complete. Best model at epoch {best_epoch} with val_loss: {min_val_loss:.4f}")
    print(f"\nTraining complete. Best model at epoch {best_epoch} with val_acc: {max_val_acc:.4f}")
    
    return {'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs}


