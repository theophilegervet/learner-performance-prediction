import argparse
import numpy as np
from random import shuffle
from scipy.sparse import load_npz, csr_matrix
from sklearn.metrics import roc_auc_score, accuracy_score

import torch.nn as nn
from torch.optim import Adam

from model_ffw import FeedForward
from utils import *


def get_tensors(sparse, item_outputs):
    dense = torch.tensor(sparse.toarray())
    inputs = dense[:, 5:].float()
    output_ids = dense[:, 1].long() if item_outputs else dense[:, 4].long()
    labels = dense[:, 3].float()
    return inputs, output_ids, labels


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


def train(X_train, X_val, model, optimizer, logger, num_epochs, batch_size, item_outputs):
    """Train feedforward baseline.

    Arguments:
        X (sparse matrix): output by encode_ffw.py
        model (torch Module)
        optimizer (torch optimizer)
        logger: wrapper for TensorboardX logger
        num_epochs (int): number of epochs to train for
        batch_size (int)
        item_outputs (bool): if True, use items as outputs instead of skills
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
            inputs, output_ids, labels = get_tensors(X_train[train_idxs[k:k + batch_size]],
                                                     item_outputs)
            inputs = inputs.cuda()
            preds = model(inputs)
            loss = compute_loss(preds, output_ids, labels.cuda(), criterion)
            train_auc = compute_auc(preds.detach().cpu(), output_ids, labels)

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
            inputs, output_ids, labels = get_tensors(X_val[val_idxs[k:k + batch_size]],
                                                     item_outputs)
            inputs = inputs.cuda()
            with torch.no_grad():
                preds = model(inputs)
            val_auc = compute_auc(preds.cpu(), output_ids, labels)
            metrics.store({'auc/val': val_auc})
        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train feedforward neural network on dense feature matrix.')
    parser.add_argument('X_file', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--logdir', type=str, default='runs/ffw')
    parser.add_argument('--item_outputs', action='store_true',
                        help='If True, use items as outputs instead of skills.')
    parser.add_argument('--hid_size', type=int, default=500)
    parser.add_argument('--drop_prob', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=30)
    args = parser.parse_args()

    num_prev_interactions = int((args.X_file.split("-")[-1]).split(".")[0])
    features_suffix = args.X_file.split("-")[-2]

    # First 5 columns are original dataset
    # then previous interaction encodings in [0, 2 * num_skills - 1]
    # then wins/attempts counts
    X = csr_matrix(load_npz(args.X_file))

    # Student-level train-val split
    user_ids = X[:, 0].toarray().flatten()
    users = np.unique(user_ids)
    np.random.shuffle(users)
    split = int(0.8 * len(users))
    users_train, users_val = users[:split], users[split:]
    X_train = X[np.where(np.isin(user_ids, users_train))]
    X_val = X[np.where(np.isin(user_ids, users_val))]

    counts_size = X_train.shape[1] - 5 - num_prev_interactions
    num_skills = int(X[:, 4].max() + 1)
    num_items = int(X[:, 1].max() + 1)
    num_outputs = num_items if args.item_outputs else num_skills

    model = FeedForward(counts_size, num_prev_interactions, num_skills, num_outputs,
                        args.hid_size, args.drop_prob).cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)

    param_str = (f'{args.dataset}, features={features_suffix}, num_prev={num_prev_interactions}, '
                 f'item_outputs={args.item_outputs}')
    logger = Logger(os.path.join(args.logdir, param_str))

    train(X_train, X_val, model, optimizer, logger, args.num_epochs, args.batch_size, 
          args.item_outputs)

    logger.close()