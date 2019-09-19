import os
import argparse
import pandas as pd
from random import shuffle
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence

from dkt import DKT
from utils.misc import Logger, Metrics


def prepare_batches(data, batch_size):
    """Prepare batches grouping padded sequences.
    
    Arguments:
        data (list of tuples of arrays)
        batch_size (int): number of sequences per batch
        
    Output:
        batches (list of tuples of torch Tensor)
    """
    shuffle(data)
    batches = []

    for k in range(0, len(data), batch_size):
        batch = data[k:k + batch_size]

        item_ids, labels = zip(*batch)
        item_ids = list(map(lambda x: torch.tensor(x + 1, dtype=torch.long), item_ids))
        labels = list(map(lambda x: torch.tensor(x, dtype=torch.long), labels))

        inputs = pad_sequence(item_ids, batch_first=True, padding_value=0) # Pad with 0
        labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # Pad with -1

        batches.append([inputs, labels])
        
    return batches


def get_metrics(preds, indexes, labels, criterion):
    labels = labels.view(-1)
    indexes = indexes.view(-1)[labels >= 0]
    preds = preds.view(-1, preds.shape[-1])[labels >= 0]
    preds = preds[torch.arange(preds.shape[0]), indexes]
    labels = labels[labels >= 0]
    
    loss = criterion(preds, labels.float())
    auc = roc_auc_score(labels.detach().cpu(), 
                        torch.sigmoid(preds).detach().cpu())
    return loss, auc


def train(df, model, optimizer, logger, num_epochs, bptt, batch_size, train_split=0.8):
    """Train DKT model.
    
    Arguments:
        df (pandas DataFrame): output by prepare_data.py
        model (torch Module)
        optimizer (torch optimizer)
        logger: wrapper for TensorboardX logger
        num_epochs (int): number of epochs to train for
        bptt (int): length of backprop through time chunks
        batch_size (int)
        train_split (float): proportion of users used for training
        
    Output:
        batches (list of tuples of torch Tensor)
    """
    # Train-test split across users
    data = [(u_df["item_id"].values, u_df["correct"].values) for _, u_df in df.groupby("user_id")]
    shuffle(data)
    train_size = int(train_split * len(data))
    train_data, val_data = data[:train_size], data[train_size:]
    
    criterion = nn.BCEWithLogitsLoss()
    metrics = Metrics()
    step = 0
    
    for epoch in range(num_epochs):
        train_batches = prepare_batches(train_data, batch_size)
        val_batches = prepare_batches(train_data, batch_size)

        # Training
        for inputs, labels in train_batches:
            inputs, labels = inputs.cuda(), labels.cuda()
            batch_size, length = inputs.shape
            preds = torch.zeros(batch_size, length, model.num_items).cuda()
            
            # Truncated backprop through time
            for i in range(0, length, bptt):
                inp = inputs[:, i:i+bptt]
                if i == 0:
                    pred, hidden = model(inp)
                else:
                    hidden = model.repackage_hidden(hidden, inp.shape[1])
                    pred, hidden = model(inp, hidden)
                preds[:, i:i+bptt] = pred
            
            loss, train_auc = get_metrics(preds, inputs, labels, criterion)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            metrics.store({'loss/train': loss.item()})
            metrics.store({'auc/train': train_auc})
            
            # Logging
            if step % 20 == 0:
                for k, v in metrics.average().items():
                    logger.log_scalar(k, v, step * batch_size)
                print(f'Step {step}, loss={loss.item()}, train_auc={train_auc}')
            
        # Validation
        model.eval()
        for inputs, labels in val_batches:
            inputs, labels = inputs.cuda(), labels.cuda()
            with torch.no_grad():
                preds, _ = model(inputs)
            _, val_auc = get_metrics(preds, inputs, labels, criterion)
            metrics.store({'auc/val': val_auc})
        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DKT.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--logdir', type=str, default='runs/dkt')
    parser.add_argument('--embed_items', action='store_true')
    parser.add_argument('--embed_size', type=int, default=100)
    parser.add_argument('--hid_size', type=int, default=100)
    parser.add_argument('--drop_prob', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--bptt', type=int, default=50)
    args = parser.parse_args()
    
    df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data.csv'), sep="\t")
    
    num_items = df["item_id"].nunique() + 1 # Add padding index
    model = DKT(num_items, args.embed_items, args.embed_size, args.hid_size, args.drop_prob).cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    param_str = f'{args.dataset}, embed={args.embed_items}, dropout={args.drop_prob}'
    logger = Logger(os.path.join(args.logdir, param_str))
    
    train(df, model, optimizer, logger, args.num_epochs, args.bptt, args.batch_size)
    
    logger.close()