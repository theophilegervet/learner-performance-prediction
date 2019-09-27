import os
import argparse
import numpy as np
from scipy.sparse import load_npz, csr_matrix

import torch.nn as nn
from torch.optim import Adam

from utils.logger import Logger
from utils.metrics import Metrics
from ffw import FeedForward
from utils.misc import *


def get_tensors(sparse):
    dense = torch.tensor(sparse.toarray())
    inputs = dense[:, 4:].float()
    item_ids = dense[:, 1].long()
    labels = dense[:, 3].float()
    return inputs, item_ids, labels


def train(X_train, X_val, model, optimizer, logger, num_epochs, batch_size):
    """Train SAKT model.

    Arguments:
        X (sparse matrix): output by encode_ffw.py
        model (torch Module)
        optimizer (torch optimizer)
        logger: wrapper for TensorboardX logger
        num_epochs (int): number of epochs to train for
        batch_size (int)
    """
    criterion = nn.BCEWithLogitsLoss()
    metrics = Metrics()
    train_idxs = np.arange(X_train.shape[0])
    val_idxs = np.arange(X_val.shape[0])
    step = 0

    for epoch in range(num_epochs):
        shuffle(train_idxs)
        shuffle(val_idxs)

        # Training
        for k in range(0, len(train_idxs), batch_size):
            inputs, item_ids, labels = get_tensors(X_train[train_idxs[k:k + batch_size]])
            inputs = inputs.cuda()
            preds = model(inputs)
            loss = compute_loss(preds, item_ids.cuda(), labels.cuda(), criterion)
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

        # Validation
        model.eval()
        for k in range(0, len(val_idxs), batch_size):
            inputs, item_ids, labels = get_tensors(X_val[val_idxs[k:k + batch_size]])
            inputs = inputs.cuda()
            with torch.no_grad():
                preds = model(inputs)
            val_auc = compute_auc(preds.cpu(), item_ids, labels)
            metrics.store({'auc/val': val_auc})
        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train feedforward neural network on dense feature matrix.')
    parser.add_argument('X_file', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--logdir', type=str, default='runs/ffw')
    parser.add_argument('--embed_size', type=int, default=200)
    parser.add_argument('--hid_size', type=int, default=500)
    parser.add_argument('--drop_prob', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=25)
    args = parser.parse_args()

    num_prev_interactions = int((args.X_file.split("-")[-1]).split(".")[0])
    features_suffix = args.X_file.split("-")[-2]

    # First four columns are original dataset
    # then previous interaction encodings and wins/attempts statistics
    X = csr_matrix(load_npz(args.X_file))

    # Student-level train-val split
    user_ids = X[:, 0].toarray().flatten()
    users = np.unique(user_ids)
    np.random.shuffle(users)
    split = int(0.8 * len(users))
    users_train, users_val = users[:split], users[split:]

    X_train = X[np.where(np.isin(user_ids, users_train))]
    X_val = X[np.where(np.isin(user_ids, users_val))]

    statistics_size = X_train.shape[1] - 4 - num_prev_interactions
    num_items = int(X[:, 1].max() + 1)

    model = FeedForward(statistics_size, num_prev_interactions, args.embed_size, args.hid_size,
                        num_items, args.drop_prob).cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)

    param_str = f'{args.dataset}, {features_suffix}, {num_prev_interactions}'
    logger = Logger(os.path.join(args.logdir, param_str))

    train(X_train, X_val, model, optimizer, logger, args.num_epochs, args.batch_size)

    logger.close()