import os
import argparse
import pandas as pd

import torch.nn as nn
from torch.optim import Adam

from dkt import DKT
from utils.logger import Logger
from utils.metrics import Metrics
from utils.misc import *


def train(df, model, optimizer, logger, num_epochs, bptt, batch_max_size, low_gpu_mem=False):
    """Train DKT model.
    
    Arguments:
        df (pandas DataFrame): output by prepare_data.py
        model (torch Module)
        optimizer (torch optimizer)
        logger: wrapper for TensorboardX logger
        num_epochs (int): number of epochs to train for
        bptt (int): length of backprop through time chunks
        batch_max_size (int)
        low_gpu_mem (bool): if True, put sequence on gpu by chunks to minimize gpu memory usage
        train_split (float): proportion of users used for training
    """
    train_data, val_data = get_data(df)

    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    metrics = Metrics()
    step = 0
    
    for epoch in range(num_epochs):
        train_batches = prepare_batches(train_data, batch_max_size)
        val_batches = prepare_batches(val_data, batch_max_size)

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
                logger.log_scalars(metrics.average(), step)
                weights = {"weight/" + name: param for name, param in model.named_parameters()}
                grads = {"grad/" + name: param.grad
                         for name, param in model.named_parameters() if param.grad is not None}
                logger.log_histograms(weights, step)
                logger.log_histograms(grads, step)

        # Validation
        model.eval()
        for inputs, item_ids, labels in val_batches:
            batch_size, length = inputs.shape

            with torch.no_grad():
                if low_gpu_mem:
                    preds = torch.zeros(batch_size, length, model.num_items)
                    for i in range(0, length, bptt):
                        inp = inputs[:, i:i + bptt].cuda()
                        if i == 0:
                            pred, hidden = model(inp)
                        else:
                            hidden = model.repackage_hidden(hidden, inp.shape[1])
                            pred, hidden = model(inp, hidden)
                        preds[:, i:i + bptt] = torch.sigmoid(pred).cpu()
                        
                else:
                    inputs = inputs.cuda()
                    preds, _ = model(inputs)
                    preds = torch.sigmoid(preds).cpu()
        
            val_auc = compute_auc(preds, item_ids, labels)
            metrics.store({'auc/val': val_auc})
        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DKT.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--logdir', type=str, default='runs/dkt')
    parser.add_argument('--embed_inputs', action='store_true')
    parser.add_argument('--embed_size', type=int, default=200)
    parser.add_argument('--hid_size', type=int, default=200)
    parser.add_argument('--num_hid_layers', type=int, default=1)
    parser.add_argument('--drop_prob', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=25)
    parser.add_argument('--bptt', type=int, default=50)
    parser.add_argument('--low_gpu_mem', action='store_true')
    args = parser.parse_args()
    
    df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data.csv'), sep="\t")

    num_items = int(df["item_id"].max() + 1)
    model = DKT(num_items, args.embed_inputs, args.embed_size, args.hid_size,
                args.num_hid_layers, args.drop_prob).cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    param_str = (f'{args.dataset}, embed={args.embed_inputs}, dropout={args.drop_prob}, batch_size={args.batch_size} '
                 f'embed_size={args.embed_size}, hid_size={args.hid_size}, num_hid_layers={args.num_hid_layers}')
    logger = Logger(os.path.join(args.logdir, param_str))
    
    train(df, model, optimizer, logger, args.num_epochs, args.bptt, args.batch_size, args.low_gpu_mem)
    
    logger.close()