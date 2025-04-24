import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc

def evaluate(preds_path: str, labels_path: str):
    # Load predictions
    # preds: CSV with columns [src,dst,etype,t0,t1,score] or just score
    try:
        preds_df = pd.read_csv(preds_path, header=None)
        # If the file has 6 columns, assume last is score
        if preds_df.shape[1] >= 1:
            scores = preds_df.iloc[:, -1].values
        else:
            raise ValueError
    except Exception:
        # fallback: single-column file
        preds_df = pd.read_csv(preds_path)
        scores = preds_df['score'].values if 'score' in preds_df.columns else preds_df.iloc[:,0].values

    # Load ground-truth labels
    # labels CSV must have a 'label' column or last column is label
    labels_df = pd.read_csv(labels_path, header=None)
    if 'label' in labels_df.columns:
        labels = labels_df['label'].values
    else:
        labels = labels_df.iloc[:, -1].values

    if len(scores) != len(labels):
        raise ValueError(f"Number of predictions ({len(scores)}) does not match number of labels ({len(labels)})")

    # Compute metrics
    roc_auc = roc_auc_score(labels, scores)
    avg_precision = average_precision_score(labels, scores)

    # Compute Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    # Print results
    print("Evaluation Results:")
    print(f"  ROC AUC           : {roc_auc:.4f}")
    print(f"  Average Precision : {avg_precision:.4f}")
    print(f"  PR AUC           : {pr_auc:.4f}")

    return roc_auc, avg_precision, pr_auc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate link prediction results')
    parser.add_argument('--preds', required=True, help='Path to predictions CSV')
    parser.add_argument('--labels', required=True, help='Path to ground-truth labels CSV')
    args = parser.parse_args()

    evaluate(args.preds, args.labels)
