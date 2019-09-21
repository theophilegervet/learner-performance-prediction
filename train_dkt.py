import os
import argparse
import pandas as pd
from random import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence

from dkt import DKT
from utils.logger import Logger
from utils.metrics import Metrics


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


def train(df, model, optimizer, logger, num_epochs, bptt, batch_size, low_gpu_mem=False, train_split=0.8):
    """Train DKT model.
    
    Arguments:
        df (pandas DataFrame): output by prepare_data.py
        model (torch Module)
        optimizer (torch optimizer)
        logger: wrapper for TensorboardX logger
        num_epochs (int): number of epochs to train for
        bptt (int): length of backprop through time chunks
        batch_size (int)
        low_gpu_mem (bool): if True, put sequence on gpu by chunks to minimize gpu memory usage
        train_split (float): proportion of users used for training
        
    Output:
        batches (list of tuples of torch Tensor)
    """
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

    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    metrics = Metrics()
    step = 0
    
    for epoch in range(num_epochs):
        train_batches = prepare_batches(train_data, batch_size)
        val_batches = prepare_batches(val_data, batch_size)

        # Training
        for inputs, item_ids, labels in train_batches:
            batch_size, length = inputs.shape
            preds = torch.zeros(batch_size, length, model.num_items)

            # Put sequence on gpu by chunks (low gpu memory usage, slow)
            if low_gpu_mem:
                loss = 0

                # Truncated backprop through time
                for i in range(0, length, bptt):
                    inp = inputs[:, i:i + bptt].cuda()
                    it_ids =  item_ids[:, i:i + bptt].cuda()
                    lab = labels[:, i:i + bptt].cuda()

                    if i == 0:
                        pred, hidden = model(inp)
                    else:
                        hidden = model.repackage_hidden(hidden, inp.shape[1])
                        pred, hidden = model(inp, hidden)

                    loss += compute_loss(pred, it_ids, lab, criterion)
                    preds[:, i:i + bptt] = torch.sigmoid(pred).detach().cpu()

            # Put sequence on gpu all at once (high gpu memory usage, fast)
            else:
                inputs, preds = inputs.cuda(), preds.cuda()

                # Truncated backprop through time
                for i in range(0, length, bptt):
                    inp = inputs[:, i:i + bptt]

                    if i == 0:
                        pred, hidden = model(inp)
                    else:
                        hidden = model.repackage_hidden(hidden, inp.shape[1])
                        pred, hidden = model(inp, hidden)

                    preds[:, i:i + bptt] = pred

                loss = compute_loss(preds, item_ids.cuda(), labels.cuda(), criterion)
                preds = preds.detach().cpu()

            loss /= labels[labels >= 0].sum()
            train_auc = compute_auc(preds, item_ids, labels)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            metrics.store({'loss/train': loss.item()})
            metrics.store({'auc/train': train_auc})
            
            # Logging
            if step % 20 == 0:
                logger.log_scalars(metrics.average(), step * batch_size)
            
        # Validation
        model.eval()
        for inputs, item_ids, labels in val_batches:
            batch_size, length = inputs.shape
            preds = torch.zeros(batch_size, length, model.num_items)

            # Split computation to avoid memory overflow even though we don't backprop
            with torch.no_grad():
                for i in range(0, length, bptt):
                    inp = inputs[:, i:i + bptt].cuda()
                    if i == 0:
                        pred, hidden = model(inp)
                    else:
                        hidden = model.repackage_hidden(hidden, inp.shape[1])
                        pred, hidden = model(inp, hidden)
                    preds[:, i:i + bptt] = torch.sigmoid(pred).cpu()

            val_auc = compute_auc(preds, item_ids, labels)
            metrics.store({'auc/val': val_auc})
        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DKT.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--logdir', type=str, default='runs/dkt')
    parser.add_argument('--embed_inputs', action='store_true')
    parser.add_argument('--embed_size', type=int, default=100)
    parser.add_argument('--hid_size', type=int, default=100)
    parser.add_argument('--drop_prob', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--bptt', type=int, default=10000)
    parser.add_argument('--low_gpu_mem', action='store_true')
    args = parser.parse_args()
    
    df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data.csv'), sep="\t")

    model = DKT(df["item_id"].nunique(), args.embed_inputs, args.embed_size, args.hid_size, args.drop_prob).cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    param_str = f'{args.dataset}, embed={args.embed_inputs}, dropout={args.drop_prob}'
    logger = Logger(os.path.join(args.logdir, param_str))
    
    train(df, model, optimizer, logger, args.num_epochs, args.bptt, args.batch_size, args.low_gpu_mem)
    
    logger.close()