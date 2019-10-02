import torch
from random import shuffle
from torch.nn.utils.rnn import pad_sequence


def get_data(df, item_inputs, item_outputs, train_split=0.8):
    item_ids = [torch.tensor(u_df["item_id"].values, dtype=torch.long)
                for _, u_df in df.groupby("user_id")]
    skill_ids = [torch.tensor(u_df["skill_id"].values, dtype=torch.long)
                 for _, u_df in df.groupby("user_id")]
    labels = [torch.tensor(u_df["correct"].values, dtype=torch.long)
              for _, u_df in df.groupby("user_id")]

    # Pad inputs with 0, this explains the +1
    input_ids = item_ids if item_inputs else skill_ids
    output_ids = item_ids if item_outputs else skill_ids
    inputs = [torch.cat((torch.zeros(1, dtype=torch.long), i * 2 + l + 1))[:-1]
                   for (i, l) in zip(input_ids, labels)]
    data = list(zip(inputs, output_ids, labels))
    shuffle(data)

    # Train-test split across users
    train_size = int(train_split * len(data))
    train_data, val_data = data[:train_size], data[train_size:]
    return train_data, val_data


def prepare_batches(data, batch_size):
    """Prepare batches grouping padded sequences.

    Arguments:
        data (list of tuples of torch Tensor)
        batch_size (int): number of sequences per batch

    Output:
        batches (list of tuples of torch Tensor)
    """
    shuffle(data)
    batches = []

    for k in range(0, len(data), batch_size):
        batch = data[k:k + batch_size]
        inputs, output_ids, labels = zip(*batch)

        inputs = pad_sequence(inputs, batch_first=True, padding_value=0)          # Pad with 0
        output_ids = pad_sequence(output_ids, batch_first=True, padding_value=0)  # Don't care
        labels = pad_sequence(labels, batch_first=True, padding_value=-1)         # Pad with -1

        batches.append([inputs, output_ids, labels])

    return batches