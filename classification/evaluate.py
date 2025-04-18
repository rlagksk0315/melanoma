import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tqdm.auto import tqdm

def evaluate(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc="Testing"):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data).squeeze()
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }
    return metrics
