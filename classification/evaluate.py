import os
import torch
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve

def evaluate(model, test_loader, results_path, model_name, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for data, targets in tqdm(test_loader, desc="Testing"):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data).squeeze()
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs > 0.5).float()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())    

    p, r, _ = precision_recall_curve(all_labels, all_probs) # PR
    fpr, tpr, _ = roc_curve(all_labels, all_probs) # ROC

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        'p' : p.tolist(),
        'r' : r.tolist(),
        'fpr' : fpr.tolist(),
        'tpr' : tpr.tolist(),
    }

    # Save metrics
    os.makedirs(results_path, exist_ok=True)
    with open(os.path.join(results_path, f"{model_name}_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics