import torch.nn as nn
import torch.nn.functional as F


class DKT(nn.Module):
    def __init__(self, num_inputs, num_outputs, embed_inputs, embed_size, hid_size,
                 num_hid_layers, drop_prob):
        """Deep knowledge tracing.
        
        Arguments:
            num_inputs (int)
            num_outputs (int)
            embed_inputs (bool): If True embed inputs, else one hot encoding
            embed_size (int): Input embedding dimension
            hid_size (int): Hidden layer dimension
            num_hid_layers (int): Number of hidden layers
            drop_prob (float): Dropout probability
        """
        super(DKT, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.embed_inputs = embed_inputs

        if self.embed_inputs:
            self.input_embeds = nn.Embedding(num_inputs, embed_size, padding_idx=0)
            self.lstm = nn.LSTM(embed_size, hid_size, num_hid_layers, batch_first=True)
        else:
            self.lstm = nn.LSTM(num_inputs, hid_size, num_hid_layers, batch_first=True)
        
        self.dropout = nn.Dropout(p=drop_prob)
        self.out = nn.Linear(hid_size, num_outputs)

    def forward(self, inputs, hidden=None):
        if self.embed_inputs:
            embeds = self.input_embeds(inputs)
        else:
            embeds = F.one_hot(inputs, self.num_inputs).float()

        out, hidden = self.lstm(embeds, hx=hidden)
        return self.out(self.dropout(out)), hidden

    def repackage_hidden(self, hidden):
        # Return detached hidden for TBPTT
        return tuple((v.detach() for v in hidden))
