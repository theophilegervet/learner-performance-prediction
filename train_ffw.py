import argparse
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
from sklearn.metrics import roc_auc_score

import torch.nn as nn
from torch.optim import Adam

from model_ffw import FeedForward
from utils import *


def get_tensors(sparse):
    # First 5 columns are the original dataset, including label in column 3
    dense = torch.tensor(sparse.toarray())
    inputs = dense[:, 5:].float()
    labels = dense[:, 3].float()
    return inputs, labels


def train_ffw(train, val, model, optimizer, logger, saver, num_epochs, batch_size):
    """Train feedforward baseline.

    Arguments:
        train (sparse matrix): output by encode.py
        val (sparse matrix): output by encode.py
        model (torch Module)
        optimizer (torch optimizer)
        logger: wrapper for TensorboardX logger
        saver: wrapper for torch saving
        num_epochs (int): number of epochs to train for
        batch_size (int)
    """
    criterion = nn.BCEWithLogitsLoss()
    metrics = Metrics()
    train_idxs = np.arange(train.shape[0])
    val_idxs = np.arange(val.shape[0])
    step = 0

    for epoch in range(num_epochs):
        np.random.shuffle(train_idxs)
        np.random.shuffle(val_idxs)

        # Training
        for k in range(0, len(train_idxs), batch_size):
            inputs, labels = get_tensors(train[train_idxs[k:k + batch_size]])
            inputs = inputs.cuda()
            preds = model(inputs).flatten()
            loss = criterion(preds, labels.cuda())
            train_auc = roc_auc_score(labels, torch.sigmoid(preds).detach().cpu())

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
            inputs, labels = get_tensors(val[val_idxs[k:k + batch_size]])
            inputs = inputs.cuda()
            with torch.no_grad():
                preds = model(inputs).flatten()
            val_auc = roc_auc_score(labels, torch.sigmoid(preds).cpu())
            metrics.store({'auc/val': val_auc})
        model.train()

        # Save model
        average_metrics = metrics.average()
        logger.log_scalars(average_metrics, step)
        stop = saver.save(average_metrics['auc/val'], model)
        if stop:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train feedforward neural network on sparse feature matrix.')
    parser.add_argument('--X_file', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--logdir', type=str, default='runs/ffw')
    parser.add_argument('--savedir', type=str, default='save/ffw')
    parser.add_argument('--hid_size', type=int, default=500)
    parser.add_argument('--drop_prob', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--num_epochs', type=int, default=30)
    args = parser.parse_args()

    features_suffix = (args.X_file.split("-")[-1]).split(".")[0]

    # Load sparse dataset
    X = csr_matrix(load_npz(args.X_file))

    train_df = pd.read_csv(f'data/{args.dataset}/preprocessed_data_train.csv', sep="\t")
    test_df = pd.read_csv(f'data/{args.dataset}/preprocessed_data_test.csv', sep="\t")

    # Student-wise train-val-test split
    user_ids = X[:, 0].toarray().flatten()
    users_test = test_df["user_id"].unique()
    users_train_val = train_df["user_id"].unique()
    split = int(0.8 * len(users_train_val))
    users_train, users_val = users_train_val[:split], users_train_val[split:]
    train = X[np.where(np.isin(user_ids, users_train))]
    val = X[np.where(np.isin(user_ids, users_val))]
    test = X[np.where(np.isin(user_ids, users_test))]

    model = FeedForward(train.shape[1] - 5, args.hid_size, args.drop_prob).cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Train
    param_str = f'{args.dataset}, features={features_suffix}'
    logger = Logger(os.path.join(args.logdir, param_str))
    saver = Saver(args.savedir, param_str)
    train_ffw(train, val, model, optimizer, logger, saver, args.num_epochs, args.batch_size)
    logger.close()

    model.eval()
    pred_test = np.zeros(len(test_df))
    for k in range(0, test.shape[0], args.batch_size):
        inputs, labels = get_tensors(test[k:k + args.batch_size])
        inputs = inputs.cuda()
        with torch.no_grad():
            pred_test[k:k + args.batch_size] = torch.sigmoid(model(inputs)).flatten().cpu().numpy()

    # Write predictions to csv
    test_df[f"FFW_{features_suffix}"] = pred_test
    test_df.to_csv(f'data/{args.dataset}/preprocessed_data_test.csv', sep="\t", index=False)

    print("auc_test = ", roc_auc_score(test_df["correct"], pred_test))

