import numpy as np
import wandb

from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")

    metrics = {"accuracy": accuracy, "f1_weighted": f1}

    wandb.log(metrics)
    return metrics
