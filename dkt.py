import torch.nn as nn
import torch.nn.functional as F


class DKT(nn.Module):
    def __init__(self, num_items, embed_inputs, embed_size, hid_size, num_hid_layers, drop_prob):
        """Deep knowledge tracing.
        
        Arguments:
            num_items (int): Number of items
            embed_inputs (bool): If True embed inputs, else one hot encoding
            embed_size (int): Input embedding dimension
            hid_size (int): Hidden layer dimension
            num_hid_layers (int): Number of hidden layers
            drop_prob (float): Dropout probability
        """
        super(DKT, self).__init__()
        self.num_items = num_items
        self.embed_inputs = embed_inputs

        if self.embed_inputs:
            self.input_embeds = nn.Embedding(2 * num_items + 1, embed_size, padding_idx=0)
            self.input_embeds.weight.requires_grad = False
            self.lstm = nn.LSTM(embed_size, hid_size, num_hid_layers)
        else:
            self.lstm = nn.LSTM(2 * num_items + 1, hid_size, num_hid_layers)
        
        self.dropout = nn.Dropout(p=drop_prob)
        self.out = nn.Linear(hid_size, num_items)

    def forward(self, inputs, hidden=None):
        if self.embed_inputs:
            embeds = self.input_embeds(inputs)
        else:
            embeds = F.one_hot(inputs, 2 * self.num_items + 1).float()
            
        out, hidden = self.lstm(embeds, hx=hidden)
        return self.out(self.dropout(out)), hidden
    
    def repackage_hidden(self, hidden, length):
        # Return detached hidden of given length for TBPTT
        return tuple((v[:, -length:].detach().contiguous() for v in hidden))