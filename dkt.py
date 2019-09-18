import torch
import torch.nn as nn
import torch.nn.functional as F


class DKT(nn.Module):
    def __init__(self, num_items, embed_size, hid_size, drop_prob):
        super(DKT, self).__init__()
        self.num_items = num_items
        
        self.item_embeds = nn.Embedding(num_items, embed_size, padding_idx=0)
        self.item_embeds.weight.requires_grad=False
        
        #self.lstm = nn.LSTM(num_items, hid_size)
        self.lstm = nn.LSTM(embed_size, hid_size)
        
        self.dropout = nn.Dropout(p=drop_prob)
        self.out = nn.Linear(hid_size, num_items)

    def forward(self, items, hidden=None):
        #embeds = F.one_hot(items, self.num_items).float()
        embeds = self.item_embeds(items)
        
        if hidden is None:
            out, hidden = self.lstm(embeds)
        else:
            out, hidden = self.lstm(embeds, hidden)
            
        return self.out(self.dropout(out)), hidden
    
    def repackage_hidden(self, hidden, length):
        # Return detached hidden of given length for TBPTT
        return tuple((v[:, -length:].detach() for v in hidden))
