import argparse
import pandas as pd
from random import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score

import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence

from sakt import SAKT
from utils import *


def get_data(df, item_inputs, item_outputs, train_split=0.8):
    item_ids = [torch.tensor(u_df["item_id"].values, dtype=torch.long)
                for _, u_df in df.groupby("user_id")]
    skill_ids = [torch.tensor(u_df["skill_id"].values, dtype=torch.long)
                 for _, u_df in df.groupby("user_id")]
    labels = [torch.tensor(u_df["correct"].values, dtype=torch.long)
              for _, u_df in df.groupby("user_id")]

    # Pad inputs with 0, this explains the +1
    input_ids = item_ids if item_inputs else skill_ids
    output_ids = item_ids if item_outputs else skill_ids
    inputs = [torch.cat((torch.zeros(1, dtype=torch.long), i * 2 + l + 1))[:-1]
                   for (i, l) in zip(input_ids, labels)]
    data = list(zip(inputs, output_ids, labels))
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
        inputs, output_ids, labels = zip(*batch)

        inputs = pad_sequence(inputs, batch_first=True, padding_value=0)          # Pad with 0
        output_ids = pad_sequence(output_ids, batch_first=True, padding_value=0)  # Don't care
        labels = pad_sequence(labels, batch_first=True, padding_value=-1)         # Pad with -1

        batches.append([inputs, output_ids, labels])

    return batches


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


def train(train_data, val_data, model, optimizer, logger, num_epochs, batch_size, grad_clip):
    """Train SAKT model.
    
    Arguments:
        train_data (list of tuples of torch Tensor)
        val_data (list of tuples of torch Tensor)
        model (torch Module)
        optimizer (torch optimizer)
        logger: wrapper for TensorboardX logger
        num_epochs (int): number of epochs to train for
        batch_size (int)
        grad_clip (float): max norm of the gradients
    """
    criterion = nn.BCEWithLogitsLoss()
    metrics = Metrics()
    step = 0
    
    for epoch in range(num_epochs):
        train_batches = prepare_batches(train_data, batch_size)
        val_batches = prepare_batches(val_data, batch_size)

        # Training
        for inputs, output_ids, labels in train_batches:
            inputs = inputs.cuda()
            preds = model(inputs)
            loss = compute_loss(preds, output_ids, labels.cuda(), criterion)
            preds = torch.sigmoid(preds).detach().cpu()
            train_auc = compute_auc(preds, output_ids, labels)

            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            step += 1
            metrics.store({'loss/train': loss.item()})
            metrics.store({'auc/train': train_auc})
            
            # Logging
            if step % 30 == 0:
                logger.log_scalars(metrics.average(), step)
                weights = {"weight/" + name: param for name, param in model.named_parameters()}
                grads = {"grad/" + name: param.grad
                         for name, param in model.named_parameters() if param.grad is not None}
                logger.log_histograms(weights, step)
                logger.log_histograms(grads, step)
            
        # Validation
        model.eval()
        for inputs, output_ids, labels in val_batches:
            inputs = inputs.cuda()
            with torch.no_grad():
                preds = torch.sigmoid(model(inputs)).cpu()
            val_auc = compute_auc(preds, output_ids, labels)
            metrics.store({'auc/val': val_auc})
        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SAKT.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--logdir', type=str, default='runs/sakt')
    parser.add_argument('--item_inputs', action='store_true',
                        help='If True, use items as inputs instead of skills.')
    parser.add_argument('--item_outputs', action='store_true',
                        help='If True, use items as outputs instead of skills.')
    parser.add_argument('--embed_size', type=int, default=200)
    parser.add_argument('--num_attn_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=5)
    parser.add_argument('--encode_pos', action='store_true')
    parser.add_argument('--drop_prob', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=10)
    parser.add_argument('--num_epochs', type=int, default=30)
    args = parser.parse_args()
    
    df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data.csv'), sep="\t")

    train_data, val_data = get_data(df, args.item_inputs, args.item_outputs)

    num_items = int(df["item_id"].max() + 1) + 1
    num_skills = int(df["skill_id"].max() + 1) + 1
    num_inputs = (2 * num_items + 1) if args.item_inputs else (2 * num_skills + 1)  # Pad with 0
    num_outputs = num_items if args.item_outputs else num_skills

    model = SAKT(num_inputs, num_outputs, args.embed_size, args.num_attn_layers, args.num_heads,
                 args.encode_pos, args.drop_prob).cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    param_str = (f'{args.dataset}, encode_pos={args.encode_pos}, '
                 f'item_inputs={args.item_inputs}, item_outputs={args.item_outputs}, '
                 f'num_attn_layers={args.num_attn_layers}')
    logger = Logger(os.path.join(args.logdir, param_str))
    
    train(train_data, val_data, model, optimizer, logger, args.num_epochs, args.batch_size,
          args.grad_clip)
    
    logger.close()