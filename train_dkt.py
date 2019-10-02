import argparse
import pandas as pd

import torch.nn as nn
from torch.optim import Adam

from dkt import DKT
from utils import *


def train(train_data, val_data, model, optimizer, logger, num_epochs, batch_size, bptt=50):
    """Train DKT model.
    
    Arguments:
        train_data (list of tuples of torch Tensor)
        val_data (list of tuples of torch Tensor)
        model (torch Module)
        optimizer (torch optimizer)
        logger: wrapper for TensorboardX logger
        num_epochs (int): number of epochs to train for
        batch_size (int)
        bptt (int): length of truncated backprop through time chunks
    """
    criterion = nn.BCEWithLogitsLoss()
    metrics = Metrics()
    step = 0
    
    for epoch in range(num_epochs):
        train_batches = prepare_batches(train_data, batch_size)
        val_batches = prepare_batches(val_data, batch_size)

        # Training
        for inputs, output_ids, labels in train_batches:
            length = inputs.size(1)
            preds = torch.empty(inputs.size(0), length, model.num_outputs)
            inputs, preds = inputs.cuda(), preds.cuda()

            # Truncated backprop through time
            for i in range(0, length, bptt):
                inp = inputs[:, i:i + bptt]
                if i == 0:
                    pred, hidden = model(inp)
                else:
                    hidden = model.repackage_hidden(hidden)
                    pred, hidden = model(inp, hidden)
                preds[:, i:i + bptt] = pred

            loss = compute_loss(preds, output_ids, labels.cuda(), criterion)
            preds = torch.sigmoid(preds).detach().cpu()
            train_auc = compute_auc(preds, output_ids, labels)

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
        for inputs, output_ids, labels in val_batches:
            with torch.no_grad():
                inputs = inputs.cuda()
                preds, _ = model(inputs)
                preds = torch.sigmoid(preds).cpu()
            val_auc = compute_auc(preds, output_ids, labels)
            metrics.store({'auc/val': val_auc})
        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DKT.')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--logdir', type=str, default='runs/dkt')
    parser.add_argument('--item_inputs', action='store_true',
                        help='If True, use items as inputs instead of skills.')
    parser.add_argument('--item_outputs', action='store_true',
                        help='If True, use items as outputs instead of skills.')
    parser.add_argument('--embed_inputs', action='store_true',
                        help='If True, embed inputs else use one hot encoding.')
    parser.add_argument('--embed_size', type=int, default=200)
    parser.add_argument('--hid_size', type=int, default=200)
    parser.add_argument('--num_hid_layers', type=int, default=1)
    parser.add_argument('--drop_prob', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--num_epochs', type=int, default=30)
    args = parser.parse_args()

    df = pd.read_csv(os.path.join('data', args.dataset, 'preprocessed_data.csv'), sep="\t")

    train_data, val_data = get_data(df, args.item_inputs, args.item_outputs)

    num_items = int(df["item_id"].max() + 1) + 1
    num_skills = int(df["skill_id"].max() + 1) + 1
    num_inputs = (2 * num_items + 1) if args.item_inputs else (2 * num_skills + 1)  # Pad with 0
    num_outputs = num_items if args.item_outputs else num_skills

    model = DKT(num_inputs, num_outputs, args.embed_inputs, args.embed_size,
                args.hid_size, args.num_hid_layers, args.drop_prob).cuda()

    optimizer = Adam(model.parameters(), lr=args.lr)
    
    param_str = (f'{args.dataset}, embed={args.embed_inputs}, embed_size={args.embed_size}, '
                 f'hid_size={args.hid_size}, num_hid_layers={args.num_hid_layers}, '
                 f'item_inputs={args.item_inputs}, item_outputs={args.item_outputs}')
    logger = Logger(os.path.join(args.logdir, param_str))
    
    train(train_data, val_data, model, optimizer, logger, args.num_epochs, args.batch_size)
    
    logger.close()