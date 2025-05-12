import os
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import json
import pandas as pd
import numpy as np

def plot_loss(train_metrics, results_path, model_num):
    os.makedirs(results_path, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.plot(train_metrics['train_losses'], label='Train Loss')
    plt.plot(train_metrics['val_losses'], label='Val Loss')
    plt.title('Loss Over Epochs for Train and Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()    
    plt.savefig(f'{results_path}/{model_num}_loss_curves.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_prs(pr_df, results_path, dataset):
    os.makedirs(results_path, exist_ok=True)
    plt.figure()
    for model, group in pr_df.groupby('model_name'):
        recall = np.concatenate(group['recall'].values).ravel()
        precision = np.concatenate(group['precision'].values).ravel()
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'Model {model} (AUC={pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curves {dataset}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, f"pr_curves.png"))
    plt.close()
    
def plot_rocs(roc_df, results_path, dataset):
    os.makedirs(results_path, exist_ok=True)
    plt.figure()
    for model, group in roc_df.groupby('model_name'):
        fpr = np.concatenate(group['fpr'].values).ravel()
        tpr = np.concatenate(group['tpr'].values).ravel()
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Model {model} (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves {dataset}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_path, "roc_curves.png"))
    plt.close()


def main():
    with open('../results/hana/final_main_1_classification/model_1_ham_metrics.json', 'r') as f:
        ham_metrics_1 = json.load(f)
    with open('../results/hana/final_main_1_classification/model_1_ddi_metrics.json', 'r') as f:
        ddi_metrics_1 = json.load(f)

    with open('../results/isalis/classification_2/model_2_metrics.json', 'r') as f:
        ddi_metrics_2 = json.load(f)

    with open('../results/isalis/classification_3/model_3_ham_metrics.json', 'r') as f:
        ham_metrics_3 = json.load(f)
    with open('../results/isalis/classification_3/model_3_ddi_metrics.json', 'r') as f:
        ddi_metrics_3 = json.load(f)

    with open('../results/isalis/classification_4/model_4_ham_metrics.json', 'r') as f:
        ham_metrics_4 = json.load(f)
    with open('../results/isalis/classification_4/model_4_ddi_metrics.json', 'r') as f:
        ddi_metrics_4 = json.load(f)

    with open('../results/hana/final_main_5_classification/model_5_ham_metrics.json', 'r') as f:
        ham_metrics_5 = json.load(f)
    with open('../results/hana/final_main_5_classification/model_5_ddi_metrics.json', 'r') as f:
        ddi_metrics_5 = json.load(f)

    with open('../results/isalis/classification_6/model_6_ham_metrics.json', 'r') as f:
        ham_metrics_6 = json.load(f)
    with open('../results/isalis/classification_6/model_6_ddi_metrics.json', 'r') as f:
        ddi_metrics_6 = json.load(f)

    with open('../results/isalis/classification_7/model_7_ham_metrics.json', 'r') as f:
        ham_metrics_7 = json.load(f)
    with open('../results/isalis/classification_7/model_7_ddi_metrics.json', 'r') as f:
        ddi_metrics_7 = json.load(f)

    with open('../results/isalis/classification_8/model_8_ddi_metrics.json', 'r') as f:
        ddi_metrics_8 = json.load(f)
    

    # HAM
    ham_pr_df = pd.DataFrame({'model_name': [1, 3, 4, 5, 6, 7],
                              'recall': [ham_metrics_1['r'], ham_metrics_3['r'],
                                         ham_metrics_4['r'], ham_metrics_5['r'],
                                         ham_metrics_6['r'], ham_metrics_7['r']],
                              'precision': [ham_metrics_1['p'], ham_metrics_3['p'],
                                            ham_metrics_4['p'], ham_metrics_5['p'],
                                            ham_metrics_6['p'], ham_metrics_7['p']]})
    
    ham_roc_df = pd.DataFrame({'model_name': [1, 3, 4, 5, 6, 7],
                               'fpr': [ham_metrics_1['fpr'], ham_metrics_3['fpr'],
                                       ham_metrics_4['fpr'], ham_metrics_5['fpr'],
                                       ham_metrics_6['fpr'], ham_metrics_7['fpr']],
                               'tpr': [ham_metrics_1['tpr'], ham_metrics_3['tpr'],
                                       ham_metrics_4['tpr'], ham_metrics_5['tpr'],
                                       ham_metrics_6['tpr'], ham_metrics_7['tpr']]})
    
    # DDI
    ddi_pr_df = pd.DataFrame({'model_name': [1, 2, 3, 4, 5, 6, 7, 8],
                              'recall': [ddi_metrics_1['r'], ddi_metrics_2['r'],
                                         ddi_metrics_3['r'], ddi_metrics_4['r'],
                                         ddi_metrics_5['r'], ddi_metrics_6['r'],
                                         ddi_metrics_7['r'], ddi_metrics_8 ['r']],
                              'precision': [ddi_metrics_1['p'], ddi_metrics_2['p'],
                                            ddi_metrics_3['p'], ddi_metrics_4['p'],
                                            ddi_metrics_5['p'], ddi_metrics_6['p'],
                                            ddi_metrics_7['p'], ddi_metrics_8 ['p']]})
    
    ddi_roc_df = pd.DataFrame({'model_name': [1, 2, 3, 4, 5, 6, 7, 8],
                               'fpr': [ddi_metrics_1['fpr'], ddi_metrics_2['fpr'],
                                       ddi_metrics_3['fpr'], ddi_metrics_4['fpr'],
                                       ddi_metrics_5['fpr'], ddi_metrics_6['fpr'],
                                       ddi_metrics_7['fpr'], ddi_metrics_8 ['fpr']],
                               'tpr': [ddi_metrics_1['tpr'], ddi_metrics_2['tpr'],
                                       ddi_metrics_3['tpr'], ddi_metrics_4['tpr'],
                                       ddi_metrics_5['tpr'], ddi_metrics_6['tpr'],
                                       ddi_metrics_7['tpr'], ddi_metrics_8 ['tpr']]})
    
    results_path_ham = '../results/HAM'
    results_path_ddi = '../results/DDI'
    
    plot_prs(ham_pr_df, results_path_ham, "HAM") # HAM PR
    plot_rocs(ham_roc_df, results_path_ham, "HAM") # HAM ROC
    plot_prs(ddi_pr_df, results_path_ddi, "DDI") # DDI PR
    plot_rocs(ddi_roc_df, results_path_ddi, "DDI") # DDI ROC

if __name__ == "__main__":
    main()