import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score

import torch.nn as nn
from torch.optim import Adam

from model_dkt import DKT
from utils import *


def get_preds(preds, item_ids, skill_ids, labels):
    preds = preds[labels >= 0]

    if (item_ids is not None):
        item_ids = item_ids[labels >= 0]
        preds = preds[torch.arange(preds.size(0)), item_ids]
    elif (skill_ids is not None):
        skill_ids = skill_ids[labels >= 0]
        preds = preds[torch.arange(preds.size(0)), skill_ids]
    else:
        raise ValueError("Use exactly one of skills or items as output")

    return preds


def compute_auc(preds, item_ids, skill_ids, labels):
    preds = get_preds(preds, item_ids, skill_ids, labels)
    labels = labels[labels >= 0].float()

    if len(torch.unique(labels)) == 1:  # Only one class
        auc = accuracy_score(labels, torch.sigmoid(preds).round())
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

            loss = compute_loss(preds, item_ids, skill_ids, labels.cuda(), criterion)
            train_auc = compute_auc(preds.detach().cpu(), item_ids, skill_ids, labels)

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
            val_auc = compute_auc(preds.cpu(), item_ids, skill_ids, labels)
            metrics.store({'auc/val': val_auc})
        model.train()

        # Save model
        #saver.save(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DKT.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--logdir', type=str, default='runs/dkt')
    parser.add_argument('--savedir', type=str, default='save/dkt')
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
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=30)
    args = parser.parse_args()

    assert (args.item_in or args.skill_in)    # Use at least one of skills or items as input
    assert (args.item_out != args.skill_out)  # Use exactly one of skills or items as output

    df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data.csv'), sep="\t")

    train_data, val_data = get_data(df, args.item_in, args.skill_in, args.item_out, args.skill_out)

    num_items = int(df["item_id"].max() + 1) + 1
    num_skills = int(df["skill_id"].max() + 1) + 1

    model = DKT(num_items, num_skills, args.hid_size, args.num_hid_layers, args.drop_prob,
                args.item_in, args.skill_in, args.item_out, args.skill_out).cuda()
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    param_str = (f'{args.dataset},item_in={args.item_in},skill_in={args.skill_in},'
                 f'item_out={args.item_out},skill_out={args.skill_out}')
    logger = Logger(os.path.join(args.logdir, param_str))
    saver = Saver(args.savedir, param_str)

    train(train_data, val_data, model, optimizer, logger, saver, args.num_epochs, args.batch_size)

    logger.close()
