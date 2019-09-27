import os
import argparse
import pandas as pd

import torch.nn as nn
from torch.optim import Adam

from sakt import SAKT
from utils.logger import Logger
from utils.metrics import Metrics
from utils.misc import *


def train(df, model, optimizer, logger, num_epochs, batch_size):
    """Train SAKT model.
    
    Arguments:
        df (pandas DataFrame): output by prepare_data.py
        model (torch Module)
        optimizer (torch optimizer)
        logger: wrapper for TensorboardX logger
        num_epochs (int): number of epochs to train for
        batch_size (int)
    """
    train_data, val_data = get_data(df)

    criterion = nn.BCEWithLogitsLoss()
    metrics = Metrics()
    step = 0
    
    for epoch in range(num_epochs):
        train_batches = prepare_batches(train_data, batch_size)
        val_batches = prepare_batches(val_data, batch_size)

        # Training
        for inputs, item_ids, labels in train_batches:
            inputs = inputs.cuda()
            preds = model(inputs)
            loss = compute_loss(preds, item_ids.cuda(), labels.cuda(), criterion)
            #loss = compute_loss(preds, item_ids, labels, criterion)
            train_auc = compute_auc(preds.detach().cpu(), item_ids, labels)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            metrics.store({'loss/train': loss.item()})
            metrics.store({'auc/train': train_auc})
            
            # Logging
            if step % 20 == 0:
                logger.log_scalars(metrics.average(), step)
                weights = {"weight/" + name: param for name, param in model.named_parameters()}
                grads = {"grad/" + name: param.grad
                         for name, param in model.named_parameters() if param.grad is not None}
                logger.log_histograms(weights, step)
                logger.log_histograms(grads, step)
            
        # Validation
        model.eval()
        for inputs, item_ids, labels in val_batches:
            inputs = inputs.cuda()
            with torch.no_grad():
                preds = model(inputs)
            val_auc = compute_auc(preds.cpu(), item_ids, labels)
            metrics.store({'auc/val': val_auc})
        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SAKT.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--logdir', type=str, default='runs/sakt')
    parser.add_argument('--embed_inputs', action='store_true')
    parser.add_argument('--embed_size', type=int, default=200)
    parser.add_argument('--hid_size', type=int, default=200)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--encode_pos', action='store_true')
    parser.add_argument('--drop_prob', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=25)
    args = parser.parse_args()
    
    df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data.csv'), sep="\t")

    num_items = int(df["item_id"].max() + 1)
    model = SAKT(num_items, args.embed_inputs, args.embed_size, args.hid_size,
                 args.num_heads, args.encode_pos, args.drop_prob).cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    param_str = (f'{args.dataset}, embed={args.embed_inputs}, dropout={args.drop_prob}, batch_size={args.batch_size} '
                 f'embed_size={args.embed_size}, hid_size={args.hid_size}, encode_pos={args.encode_pos}')
    logger = Logger(os.path.join(args.logdir, param_str))
    
    train(df, model, optimizer, logger, args.num_epochs, args.batch_size)
    
    logger.close()