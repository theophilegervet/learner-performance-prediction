import random
from random import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score

import torch
from torch.nn.utils.rnn import pad_sequence


def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    
    
def get_data(df, train_split=0.8):
    num_items = df["item_id"].nunique()
    data = [(torch.tensor(u_df["item_id"].values, dtype=torch.long),
             torch.tensor(u_df["correct"].values, dtype=torch.long))
            for _, u_df in df.groupby("user_id")]
    data = [(torch.cat((torch.zeros(1, dtype=torch.long), item_ids + labels * num_items + 1))[:-1], item_ids, labels)
            for (item_ids, labels) in data]
    shuffle(data)

    # Train-test split across users
    train_size = int(train_split * len(data))
    train_data, val_data = data[:train_size], data[train_size:]
    return train_data, val_data


def prepare_batches(data, batch_size):
    """Prepare batches grouping padded sequences.
    
    Arguments:
        data (list of tuples of torch Tensor)
        batch_size (int): number of sequences per batch
        
    Output:
        batches (list of tuples of torch Tensor)
    """
    shuffle(data)
    batches = []

    for k in range(0, len(data), batch_size):
        batch = data[k:k + batch_size]
        inputs, item_ids, labels = zip(*batch)

        inputs = pad_sequence(inputs, batch_first=True, padding_value=0)     # Pad with 0
        item_ids = pad_sequence(item_ids, batch_first=True, padding_value=0) # Don't care
        labels = pad_sequence(labels, batch_first=True, padding_value=-1)    # Pad with -1

        batches.append([inputs, item_ids, labels])
        
    return batches


def compute_auc(preds, item_ids, labels):
    labels = labels.view(-1)
    item_ids = item_ids.view(-1)[labels >= 0]
    preds = preds.view(-1, preds.shape[-1])[labels >= 0]
    preds = preds[torch.arange(preds.shape[0]), item_ids]
    labels = labels[labels >= 0].float()

    if len(torch.unique(labels)) == 1: # Only one class
        auc = accuracy_score(labels, preds.round())
    else:
        auc = roc_auc_score(labels, preds)
    return auc


def compute_loss(preds, item_ids, labels, criterion):
    labels = labels.view(-1)
    item_ids = item_ids.view(-1)[labels >= 0]
    preds = preds.view(-1, preds.shape[-1])[labels >= 0]
    preds = preds[torch.arange(preds.shape[0]), item_ids]
    labels = labels[labels >= 0].float()
    return criterion(preds, labels)