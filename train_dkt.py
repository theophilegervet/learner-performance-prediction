import argparse
import pandas as pd
from random import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score

import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence

from dkt import DKT
from utils import *


def get_data(df, item_in, skill_in, item_out, skill_out, train_split=0.8):
    """Extract sequences from dataframe.

    Arguments:
        df (pandas Dataframe): output by prepare_data.py
        item_in (bool): if True, use items as inputs
        skill_in (bool): if True, use skills as inputs
        item_out (bool): if True, use items as outputs
        skill_out (bool): if True, use skills as outputs
        train_split (float): proportion of data to use for training
    """
    item_ids = [torch.tensor(u_df["item_id"].values, dtype=torch.long)
                for _, u_df in df.groupby("user_id")]
    skill_ids = [torch.tensor(u_df["skill_id"].values, dtype=torch.long)
                 for _, u_df in df.groupby("user_id")]
    labels = [torch.tensor(u_df["correct"].values, dtype=torch.long)
              for _, u_df in df.groupby("user_id")]

    item_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), i * 2 + l + 1))[:-1]
                   for (i, l) in zip(item_ids, labels)]
    skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), s * 2 + l + 1))[:-1]
                    for (s, l) in zip(skill_ids, labels)]

    item_inputs = item_inputs if item_in else [None] * len(item_inputs)
    skill_inputs = skill_inputs if skill_in else [None] * len(skill_inputs)
    item_ids = item_ids if item_out else [None] * len(item_ids)
    skill_ids = skill_ids if skill_out else [None] * len(skill_ids)

    data = list(zip(item_inputs, skill_inputs, item_ids, skill_ids, labels))
    shuffle(data)

    # Train-test split across users
    train_size = int(train_split * len(data))
    train_data, val_data = data[:train_size], data[train_size:]
    return train_data, val_data


def prepare_batches(data, batch_size):
    """Prepare batches grouping padded sequences.

    Arguments:
        data (list of lists of torch Tensor): output by get_data
        batch_size (int): number of sequences per batch

    Output:
        batches (list of lists of torch Tensor)
    """
    shuffle(data)
    batches = []

    for k in range(0, len(data), batch_size):
        batch = data[k:k + batch_size]
        seq_lists = list(zip(*batch))
        inputs_and_ids = [pad_sequence(seqs, batch_first=True, padding_value=0)
                          if (seqs[0] is not None) else None for seqs in seq_lists[:4]]
        labels = pad_sequence(seq_lists[-1], batch_first=True, padding_value=-1)  # Pad labels with -1
        batches.append([*inputs_and_ids, labels])

    return batches


def get_preds(preds, item_ids, skill_ids, labels, num_items):
    preds = preds[labels >= 0]
    item_ids = item_ids[labels >= 0] if item_ids is not None else None
    skill_ids = skill_ids[labels >= 0] if skill_ids is not None else None

    if (item_ids is not None) and (skill_ids is not None):
        preds_items = preds[torch.arange(preds.size(0)), item_ids]
        preds_skills = preds[torch.arange(preds.size(0)), skill_ids + num_items]
        preds = preds_items + preds_skills
    elif (item_ids is not None):
        preds = preds[torch.arange(preds.size(0)), item_ids]
    elif (skill_ids is not None):
        preds = preds[torch.arange(preds.size(0)), skill_ids]
    return preds


def compute_auc(preds, item_ids, skill_ids, labels, num_items):
    preds = get_preds(preds, item_ids, skill_ids, labels, num_items)
    labels = labels[labels >= 0].float()

    if len(torch.unique(labels)) == 1: # Only one class
        auc = accuracy_score(labels, torch.sigmoid(preds).round())
    else:
        auc = roc_auc_score(labels, preds)
    return auc


def compute_loss(preds, item_ids, skill_ids, labels, criterion, num_items):
    preds = get_preds(preds, item_ids, skill_ids, labels, num_items)
    labels = labels[labels >= 0].float()
    return criterion(preds, labels)


def train(train_data, val_data, model, optimizer, logger, num_epochs, batch_size, bptt=50):
    """Train DKT model.
    
    Arguments:
        train_data (list of lists of torch Tensor)
        val_data (list of lists of torch Tensor)
        model (torch Module)
        optimizer (torch optimizer)
        logger: wrapper for TensorboardX logger
        num_epochs (int): number of epochs to train for
        batch_size (int)
        bptt (int): length of truncated backprop through time chunks
    """
    criterion = nn.BCEWithLogitsLoss()
    metrics = Metrics()
    step = 0
    
    for epoch in range(num_epochs):
        train_batches = prepare_batches(train_data, batch_size)
        val_batches = prepare_batches(val_data, batch_size)

        # Training
        for item_inputs, skill_inputs, item_ids, skill_ids, labels in train_batches:
            length = labels.size(1)
            preds = torch.empty(labels.size(0), length, model.output_size)
            item_inputs = item_inputs.cuda() if item_inputs is not None else None
            skill_inputs = skill_inputs.cuda() if skill_inputs is not None else None
            preds = preds.cuda()

            # Truncated backprop through time
            for i in range(0, length, bptt):
                item_inp = item_inputs[:, i:i + bptt] if item_inputs is not None else None
                skill_inp = skill_inputs[:, i:i + bptt] if skill_inputs is not None else None
                if i == 0:
                    pred, hidden = model(item_inp, skill_inp)
                else:
                    hidden = model.repackage_hidden(hidden)
                    pred, hidden = model(item_inp, skill_inp, hidden)
                preds[:, i:i + bptt] = pred

            loss = compute_loss(preds, item_ids, skill_ids, labels.cuda(), criterion, model.num_items)
            train_auc = compute_auc(preds.detach().cpu(), item_ids, skill_ids, labels, model.num_items)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            metrics.store({'loss/train': loss.item()})
            metrics.store({'auc/train': train_auc})

            # Logging
            if step % 20 == 0:
                logger.log_scalars(metrics.average(), step)
                #weights = {"weight/" + name: param for name, param in model.named_parameters()}
                #grads = {"grad/" + name: param.grad
                #         for name, param in model.named_parameters() if param.grad is not None}
                #logger.log_histograms(weights, step)
                #logger.log_histograms(grads, step)

        # Validation
        model.eval()
        for item_inputs, skill_inputs, item_ids, skill_ids, labels in val_batches:
            with torch.no_grad():
                item_inputs = item_inputs.cuda() if item_inputs is not None else None
                skill_inputs = skill_inputs.cuda() if skill_inputs is not None else None
                preds, _ = model(item_inputs, skill_inputs)
            val_auc = compute_auc(preds.cpu(), item_ids, skill_ids, labels, model.num_items)
            metrics.store({'auc/val': val_auc})
        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DKT.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--logdir', type=str, default='runs/dkt')
    parser.add_argument('--item_in', action='store_true',
                        help='If True, use items as inputs.')
    parser.add_argument('--skill_in', action='store_true',
                        help='If True, use skills as inputs.')
    parser.add_argument('--item_out', action='store_true',
                        help='If True, use items as outputs.')
    parser.add_argument('--skill_out', action='store_true',
                        help='If True, use skills as outputs.')
    parser.add_argument('--hid_size', type=int, default=200)
    parser.add_argument('--num_hid_layers', type=int, default=1)
    parser.add_argument('--drop_prob', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--num_epochs', type=int, default=30)
    args = parser.parse_args()

    # Check that there is at least one of items or skills as input and output
    assert (args.item_in or args.skill_in)
    assert (args.item_out or args.skill_out)

    df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data.csv'), sep="\t")

    train_data, val_data = get_data(df, args.item_in, args.skill_in, args.item_out, args.skill_out)

    num_items = int(df["item_id"].max() + 1) + 1
    num_skills = int(df["skill_id"].max() + 1) + 1

    model = DKT(num_items, num_skills, args.hid_size, args.num_hid_layers, args.drop_prob,
                args.item_in, args.skill_in, args.item_out, args.skill_out).cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    param_str = (f'{args.dataset},item_in={args.item_in},skill_in={args.skill_in}'
                 f'item_out={args.item_out},skill_out={args.skill_out}')
    logger = Logger(os.path.join(args.logdir, param_str))

    train(train_data, val_data, model, optimizer, logger, args.num_epochs, args.batch_size)

    logger.close()
