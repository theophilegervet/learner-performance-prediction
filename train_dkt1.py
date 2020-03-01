import argparse
import pandas as pd
from random import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score

import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence

from model_dkt1 import DKT1
from utils import *


def cuda(tensor):
    return tensor.cuda() if tensor is not None else None


def get_data(df, item_in, skill_in, item_out, skill_out, skill_separate, train_split=0.8, randomize=True):
    """Extract sequences from dataframe.

    Arguments:
        df (pandas Dataframe): output by prepare_data.py
        item_in (bool): if True, use items as inputs
        skill_in (bool): if True, use skills as inputs
        item_out (bool): if True, use items as outputs
        skill_out (bool): if True, use skills as outputs
        train_split (float): proportion of data to use for training
    """
    idx = ["user_id", "skill_id"] if skill_separate else "user_id"
    item_ids = [torch.tensor(u_df["item_id"].values, dtype=torch.long)
                for _, u_df in df.groupby(idx)]
    skill_ids = [torch.tensor(u_df["skill_id"].values, dtype=torch.long)
                 for _, u_df in df.groupby(idx)]
    labels = [torch.tensor(u_df["correct"].values, dtype=torch.long)
              for _, u_df in df.groupby(idx)]

    item_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), i * 2 + l + 1))[:-1]
                   for (i, l) in zip(item_ids, labels)]
    skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), s * 2 + l + 1))[:-1]
                    for (s, l) in zip(skill_ids, labels)]

    item_inputs = item_inputs if item_in else [None] * len(item_inputs)
    skill_inputs = skill_inputs if skill_in else [None] * len(skill_inputs)
    item_ids = item_ids if item_out else [None] * len(item_ids)
    skill_ids = skill_ids if skill_out else [None] * len(skill_ids)

    data = list(zip(item_inputs, skill_inputs, item_ids, skill_ids, labels))
    if randomize:
        shuffle(data)

    # Train-test split across users
    train_size = int(train_split * len(data))
    train_data, val_data = data[:train_size], data[train_size:]
    return train_data, val_data


def prepare_batches(data, batch_size, randomize=True):
    """Prepare batches grouping padded sequences.

    Arguments:
        data (list of lists of torch Tensor): output by get_data
        batch_size (int): number of sequences per batch

    Output:
        batches (list of lists of torch Tensor)
    """
    if randomize:
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


def get_preds(preds, item_ids, skill_ids, labels):
    preds = preds[labels >= 0]

    if (item_ids is not None):
        item_ids = item_ids[labels >= 0]
        preds = preds[torch.arange(preds.size(0)), item_ids]
    elif (skill_ids is not None):
        skill_ids = skill_ids[labels >= 0]
        preds = preds[torch.arange(preds.size(0)), skill_ids]

    return preds


def compute_auc(preds, item_ids, skill_ids, labels):
    preds = get_preds(preds, item_ids, skill_ids, labels)
    labels = labels[labels >= 0].float()

    if len(torch.unique(labels)) == 1:  # Only one class
        auc = accuracy_score(labels, preds.round())
    else:
        auc = roc_auc_score(labels, preds)

    return auc


def compute_loss(preds, item_ids, skill_ids, labels, criterion):
    preds = get_preds(preds, item_ids, skill_ids, labels)
    labels = labels[labels >= 0].float()
    return criterion(preds, labels)


def train(train_data, val_data, model, optimizer, logger, saver, num_epochs, batch_size, bptt=50):
    """Train DKT model.
    
    Arguments:
        train_data (list of lists of torch Tensor)
        val_data (list of lists of torch Tensor)
        model (torch Module)
        optimizer (torch optimizer)
        logger: wrapper for TensorboardX logger
        saver: wrapper for torch saving
        num_epochs (int): number of epochs to train for
        batch_size (int)
        bptt (int): length of truncated backprop through time chunks
        savepath (str): directory where to save the trained model
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
            preds = preds.cuda()
            item_inputs = cuda(item_inputs)
            skill_inputs = cuda(skill_inputs)

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

            loss = compute_loss(preds, item_ids, skill_ids, labels.cuda(), criterion)
            train_auc = compute_auc(torch.sigmoid(preds).detach().cpu(), item_ids, skill_ids, labels)

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
                item_inputs = cuda(item_inputs)
                skill_inputs = cuda(skill_inputs)
                preds, _ = model(item_inputs, skill_inputs)
            val_auc = compute_auc(torch.sigmoid(preds).cpu(), item_ids, skill_ids, labels)
            metrics.store({'auc/val': val_auc})
        model.train()

        # Save model
        average_metrics = metrics.average()
        logger.log_scalars(average_metrics, step)
        stop = saver.save(average_metrics['auc/val'], model)
        if stop:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DKT1.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--logdir', type=str, default='runs/dkt1')
    parser.add_argument('--savedir', type=str, default='save/dkt1')
    parser.add_argument('--item_in', action='store_true',
                        help='If True, use items as inputs.')
    parser.add_argument('--skill_in', action='store_true',
                        help='If True, use skills as inputs.')
    parser.add_argument('--item_out', action='store_true',
                        help='If True, use items as outputs.')
    parser.add_argument('--skill_out', action='store_true',
                        help='If True, use skills as outputs.')
    parser.add_argument('--skill_separate', action='store_true',
                        help='If True, train a separate model for every skill.')
    parser.add_argument('--hid_size', type=int, default=200)
    parser.add_argument('--num_hid_layers', type=int, default=1)
    parser.add_argument('--drop_prob', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--num_epochs', type=int, default=300)
    args = parser.parse_args()

    assert (args.item_in or args.skill_in)    # Use at least one of skills or items as input
    assert (args.item_out != args.skill_out)  # Use exactly one of skills or items as output

    full_df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data.csv'), sep="\t")
    train_df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data_train.csv'), sep="\t")
    test_df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data_test.csv'), sep="\t")

    train_data, val_data = get_data(train_df, args.item_in, args.skill_in, args.item_out,
                                    args.skill_out, args.skill_separate)

    num_items = int(full_df["item_id"].max() + 1) + 1
    num_skills = int(full_df["skill_id"].max() + 1) + 1

    model = DKT1(num_items, num_skills, args.hid_size, args.num_hid_layers, args.drop_prob,
                 args.item_in, args.skill_in, args.item_out, args.skill_out).cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Reduce batch size until it fits on GPU
    while True:
        try:
            # Train
            param_str = (f'{args.dataset},'
                         f'batch_size={args.batch_size},'
                         f'item_in={args.item_in},'
                         f'skill_in={args.skill_in},'
                         f'item_out={args.item_out},'
                         f'skill_out={args.skill_out}'
                         f'skill_separate={args.skill_separate}')
            logger = Logger(os.path.join(args.logdir, param_str))
            saver = Saver(args.savedir, param_str)
            train(train_data, val_data, model, optimizer, logger, saver, args.num_epochs, args.batch_size)
            break
        except RuntimeError:
            args.batch_size = args.batch_size // 2
            print(f'Batch does not fit on gpu, reducing size to {args.batch_size}')

    logger.close()

    model = saver.load()
    test_data, _ = get_data(test_df, args.item_in, args.skill_in, args.item_out,
                            args.skill_out, args.skill_separate, train_split=1.0,
                            randomize=False)
    test_batches = prepare_batches(test_data, args.batch_size, randomize=False)
    test_preds = np.empty(0)

    # Predict on test set
    model.eval()
    for item_inputs, skill_inputs, item_ids, skill_ids, labels in test_batches:
        with torch.no_grad():
            item_inputs = cuda(item_inputs)
            skill_inputs = cuda(skill_inputs)
            preds, _ = model(item_inputs, skill_inputs)
            preds = torch.sigmoid(get_preds(preds, item_ids, skill_ids, labels)).cpu().numpy()
            test_preds = np.concatenate([test_preds, preds])

    # Write predictions to csv
    test_df["DKT1"] = test_preds
    test_df.to_csv(f'data/{args.dataset}/preprocessed_data_test.csv', sep="\t", index=False)

    print("auc_test = ", roc_auc_score(test_df["correct"], test_preds))

