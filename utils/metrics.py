import torch
from sklearn.metrics import roc_auc_score, accuracy_score


def compute_auc(preds, item_ids, labels):
    preds = preds[labels >= 0]
    item_ids = item_ids[labels >= 0]
    labels = labels[labels >= 0].float()
    preds = preds[torch.arange(preds.size(0)), item_ids]
    if len(torch.unique(labels)) == 1: # Only one class
        auc = accuracy_score(labels, preds.round())
    else:
        auc = roc_auc_score(labels, preds)
    return auc


def compute_loss(preds, item_ids, labels, criterion):
    preds = preds[labels >= 0]
    item_ids = item_ids[labels >= 0]
    labels = labels[labels >= 0].float()
    preds = preds[torch.arange(preds.size(0)), item_ids]
    return criterion(preds, labels)


class Metrics:
    """Keep track of metrics over time in a dictionary.
    """
    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def store(self, new_metrics):
        for key in new_metrics:
            if key in self.metrics:
                self.metrics[key] += new_metrics[key]
                self.counts[key] += 1
            else:
                self.metrics[key] = new_metrics[key]
                self.counts[key] = 1

    def average(self):
        average = {k: v / self.counts[k] for k, v in self.metrics.items()}
        self.metrics, self.counts = {}, {}
        return average