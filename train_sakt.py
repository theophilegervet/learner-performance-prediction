import argparse
import pandas as pd
from random import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score

import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence

from model_sakt import SAKT
from utils import *


def get_data(df, max_length, train_split=0.8, randomize=True):
    """Extract sequences from dataframe.

    Arguments:
        df (pandas Dataframe): output by prepare_data.py
        max_length (int): maximum length of a sequence chunk
        train_split (float): proportion of data to use for training
    """
    item_ids = [torch.tensor(u_df["item_id"].values, dtype=torch.long)
                for _, u_df in df.groupby("user_id")]
    skill_ids = [torch.tensor(u_df["skill_id"].values, dtype=torch.long)
                 for _, u_df in df.groupby("user_id")]
    labels = [torch.tensor(u_df["correct"].values, dtype=torch.long)
              for _, u_df in df.groupby("user_id")]

    item_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), i + 1))[:-1] for i in item_ids]
    skill_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), s + 1))[:-1] for s in skill_ids]
    label_inputs = [torch.cat((torch.zeros(1, dtype=torch.long), l))[:-1] for l in labels]

    def chunk(list):
        if list[0] is None:
            return list
        list = [torch.split(elem, max_length) for elem in list]
        return [elem for sublist in list for elem in sublist]

    # Chunk sequences
    lists = (item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels)
    chunked_lists = [chunk(l) for l in lists]

    data = list(zip(*chunked_lists))
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
                          if (seqs[0] is not None) else None for seqs in seq_lists[:-1]]
        labels = pad_sequence(seq_lists[-1], batch_first=True, padding_value=-1)  # Pad labels with -1
        batches.append([*inputs_and_ids, labels])

    return batches


def compute_auc(preds, labels):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    if len(torch.unique(labels)) == 1:  # Only one class
        auc = accuracy_score(labels, preds.round())
    else:
        auc = roc_auc_score(labels, preds)
    return auc


def compute_loss(preds, labels, criterion):
    preds = preds[labels >= 0].flatten()
    labels = labels[labels >= 0].float()
    return criterion(preds, labels)


def train(train_data, val_data, model, optimizer, logger, saver, num_epochs, batch_size, grad_clip):
    """Train SAKT model.

    Arguments:
        train_data (list of tuples of torch Tensor)
        val_data (list of tuples of torch Tensor)
        model (torch Module)
        optimizer (torch optimizer)
        logger: wrapper for TensorboardX logger
        saver: wrapper for torch saving
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
        for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in train_batches:
            item_inputs = item_inputs.cuda()
            skill_inputs = skill_inputs.cuda()
            label_inputs = label_inputs.cuda()
            item_ids = item_ids.cuda()
            skill_ids = skill_ids.cuda()

            preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            loss = compute_loss(preds, labels.cuda(), criterion)
            preds = torch.sigmoid(preds).detach().cpu()
            train_auc = compute_auc(preds, labels)

            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            step += 1
            metrics.store({'loss/train': loss.item()})
            metrics.store({'auc/train': train_auc})

            # Logging
            if step % 20 == 0:
                logger.log_scalars(metrics.average(), step)
                # weights = {"weight/" + name: param for name, param in model.named_parameters()}
                # grads = {"grad/" + name: param.grad
                #         for name, param in model.named_parameters() if param.grad is not None}
                # logger.log_histograms(weights, step)
                # logger.log_histograms(grads, step)

        # Validation
        model.eval()
        for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in val_batches:
            item_inputs = item_inputs.cuda()
            skill_inputs = skill_inputs.cuda()
            label_inputs = label_inputs.cuda()
            item_ids = item_ids.cuda()
            skill_ids = skill_ids.cuda()
            with torch.no_grad():
                preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
                preds = torch.sigmoid(preds).cpu()
            val_auc = compute_auc(preds, labels)
            metrics.store({'auc/val': val_auc})
        model.train()

        # Save model
        average_metrics = metrics.average()
        logger.log_scalars(average_metrics, step)
        stop = saver.save(average_metrics['auc/val'], model)
        if stop:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SAKT.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--logdir', type=str, default='runs/sakt')
    parser.add_argument('--savedir', type=str, default='save/sakt')
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--embed_size', type=int, default=200)
    parser.add_argument('--num_attn_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=5)
    parser.add_argument('--encode_pos', action='store_true')
    parser.add_argument('--max_pos', type=int, default=10)
    parser.add_argument('--drop_prob', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=10)
    parser.add_argument('--num_epochs', type=int, default=300)
    args = parser.parse_args()

    full_df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data.csv'), sep="\t")
    train_df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data_train.csv'), sep="\t")
    test_df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data_test.csv'), sep="\t")

    train_data, val_data = get_data(train_df, args.max_length)

    num_items = int(full_df["item_id"].max() + 1)
    num_skills = int(full_df["skill_id"].max() + 1)

    model = SAKT(num_items, num_skills, args.embed_size, args.num_attn_layers, args.num_heads,
                  args.encode_pos, args.max_pos, args.drop_prob).cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Reduce batch size until it fits on GPU
    while True:
        try:
            # Train
            param_str = (f'{args.dataset},'
                         f'batch_size={args.batch_size},'
                         f'max_length={args.max_length},'
                         f'encode_pos={args.encode_pos},'
                         f'max_pos={args.max_pos}')
            logger = Logger(os.path.join(args.logdir, param_str))
            saver = Saver(args.savedir, param_str)
            train(train_data, val_data, model, optimizer, logger, saver, args.num_epochs,
                  args.batch_size, args.grad_clip)
            break
        except RuntimeError:
            args.batch_size = args.batch_size // 2
            print(f'Batch does not fit on gpu, reducing size to {args.batch_size}')

    logger.close()

    test_data, _ = get_data(test_df, args.max_length, train_split=1.0, randomize=False)
    test_batches = prepare_batches(test_data, args.batch_size, randomize=False)
    test_preds = np.empty(0)

    # Predict on test set
    model.eval()
    for item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, labels in test_batches:
        item_inputs = item_inputs.cuda()
        skill_inputs = skill_inputs.cuda()
        label_inputs = label_inputs.cuda()
        item_ids = item_ids.cuda()
        skill_ids = skill_ids.cuda()
        with torch.no_grad():
            preds = model(item_inputs, skill_inputs, label_inputs, item_ids, skill_ids)
            preds = torch.sigmoid(preds[labels >= 0]).flatten().cpu().numpy()
            test_preds = np.concatenate([test_preds, preds])

    # Write predictions to csv
    test_df["SAKT"] = test_preds
    test_df.to_csv(f'data/{args.dataset}/preprocessed_data_test.csv', sep="\t", index=False)

    print("auc_test = ", roc_auc_score(test_df["correct"], test_preds))
