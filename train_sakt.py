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


def get_data(df, max_length, item_in, skill_in, item_out, skill_out, train_split=0.8):
    """Extract sequences from dataframe.

    Arguments:
        df (pandas Dataframe): output by prepare_data.py
        max_length (int): maximum length of a sequence chunk
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

    def chunk(list):
        if list[0] is None:
            return list
        list = [torch.split(elem, max_length) for elem in list]
        return [elem for sublist in list for elem in sublist]

    # Chunk sequences
    lists = (item_inputs, skill_inputs, item_ids, skill_ids, labels)
    chunked_lists = [chunk(l)for l in lists]

    data = list(zip(*chunked_lists))
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
        for item_inputs, skill_inputs, item_ids, skill_ids, labels in train_batches:
            item_inputs = item_inputs.cuda() if item_inputs is not None else None
            skill_inputs = skill_inputs.cuda() if skill_inputs is not None else None
            item_ids = item_ids.cuda() if item_ids is not None else None
            skill_ids = skill_ids.cuda() if skill_ids is not None else None

            preds = model(item_inputs, skill_inputs, item_ids, skill_ids)
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
                #weights = {"weight/" + name: param for name, param in model.named_parameters()}
                #grads = {"grad/" + name: param.grad
                #         for name, param in model.named_parameters() if param.grad is not None}
                #logger.log_histograms(weights, step)
                #logger.log_histograms(grads, step)
            
        # Validation
        model.eval()
        for item_inputs, skill_inputs, item_ids, skill_ids, labels in val_batches:
            item_inputs = item_inputs.cuda() if item_inputs is not None else None
            skill_inputs = skill_inputs.cuda() if skill_inputs is not None else None
            item_ids = item_ids.cuda() if item_ids is not None else None
            skill_ids = skill_ids.cuda() if skill_ids is not None else None
            with torch.no_grad():
                preds = model(item_inputs, skill_inputs, item_ids, skill_ids)
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
    parser.add_argument('--item_in', action='store_true',
                        help='If True, use items as inputs.')
    parser.add_argument('--skill_in', action='store_true',
                        help='If True, use skills as inputs.')
    parser.add_argument('--item_out', action='store_true',
                        help='If True, use items as outputs.')
    parser.add_argument('--skill_out', action='store_true',
                        help='If True, use skills as outputs.')
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--embed_size', type=int, default=200)
    parser.add_argument('--num_attn_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=5)
    parser.add_argument('--pos_encoding', type=str, default='none', 
                        help='One of "none", "key", "key_value"')
    parser.add_argument('--max_pos', type=int, default=10)
    parser.add_argument('--drop_prob', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--grad_clip', type=float, default=10)
    parser.add_argument('--num_epochs', type=int, default=100)
    args = parser.parse_args()

    assert (args.item_in or args.skill_in)    # Use at least one of skills or items as input
    assert (args.item_out or args.skill_out)  # Use at least one of skills or items as output
    assert args.pos_encoding in ["none", "key", "key_value"]

    # TODO when write results to file, train on train set only
    df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data.csv'), sep="\t")

    train_data, val_data = get_data(df, args.max_length, args.item_in, args.skill_in, args.item_out,
                                    args.skill_out)

    num_items = int(df["item_id"].max() + 1) + 1
    num_skills = int(df["skill_id"].max() + 1) + 1

    model = SAKT(num_items, num_skills, args.embed_size, args.num_attn_layers, args.num_heads,
                 args.pos_encoding, args.max_pos, args.drop_prob, args.item_in, args.skill_in,
                 args.item_out, args.skill_out).cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Reduce batch size until it fits on GPU
    while True:
        try:
            param_str = (f'{args.dataset},'
                         f'batch_size={args.batch_size},'
                         f'max_length={args.max_length},'
                         f'pos_encoding={args.pos_encoding},'
                         f'max_pos={args.max_pos},'
                         f'item_in={args.item_in},'
                         f'skill_in={args.skill_in},'
                         f'item_out={args.item_out},'
                         f'skill_out={args.skill_out}')
            logger = Logger(os.path.join(args.logdir, param_str))
            saver = Saver(args.savedir, param_str)
            train(train_data, val_data, model, optimizer, logger, saver, args.num_epochs,
                  args.batch_size, args.grad_clip)
            break
        except RuntimeError:
            args.batch_size = args.batch_size // 2
            print(f'Batch does not fit on gpu, reducing size to {args.batch_size}')

    logger.close()
