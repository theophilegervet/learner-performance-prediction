import torch.nn as nn
import torch.nn.functional as F


class DKT(nn.Module):
    def __init__(self, num_items, embed_inputs, embed_size, hid_size, drop_prob):
        """
        :param num_items (int): Number of items
        :param embed_inputs (bool): If True embed inputs, else one hot encoding
        :param embed_size (int): Item embedding dimension
        :param hid_size (int): Hidden layer dimension
        :param drop_prob (float): Dropout probability
        """
        super(DKT, self).__init__()
        self.num_items = num_items
        self.embed_inputs = embed_inputs

        if self.embed_inputs:
            self.input_embeds = nn.Embedding(2 * num_items + 1, embed_size, padding_idx=0)
            self.input_embeds.weight.requires_grad = False
            self.lstm = nn.LSTM(embed_size, hid_size)
        else:
            self.lstm = nn.LSTM(2 * num_items + 1, hid_size)
        
        self.dropout = nn.Dropout(p=drop_prob)
        self.out = nn.Linear(hid_size, num_items)

    def forward(self, items, hidden=None):
        if self.embed_inputs:
            embeds = self.input_embeds(items)
        else:
            embeds = F.one_hot(items, 2 * self.num_items + 1).float()
        out, hidden = self.lstm(embeds, hx=hidden)
        return self.out(self.dropout(out)), hidden
    
    def repackage_hidden(self, hidden, length):
        # Return detached hidden of given length for TBPTT
        return tuple((v[:, -length:].detach() for v in hidden))