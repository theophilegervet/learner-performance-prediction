import torch
import torch.nn as nn


class DKT2(nn.Module):
    def __init__(self, num_items, num_skills, hid_size, embed_size, num_hid_layers, drop_prob):
        """Deep knowledge tracing (https://papers.nips.cc/paper/5654-deep-knowledge-tracing.pdf)
        with some changes inspired by Deep Hierarchical Knowledge Tracing (https://arxiv.org/pdf/1908.02146.pdf).

        Arguments:
            num_items (int): number of items
            num_skills (int): number of skills
            hid_size (int): hidden layer dimension
            embed_size (int): query embedding dimension
            num_hid_layers (int): number of hidden layers
            drop_prob (float): dropout probability
        """
        super(DKT2, self).__init__()
        self.embed_size = embed_size

        self.item_embeds = nn.Embedding(num_items + 1, embed_size // 2, padding_idx=0)
        self.skill_embeds = nn.Embedding(num_skills + 1, embed_size // 2, padding_idx=0)
        
        self.lstm = nn.LSTM(2 * embed_size, hid_size, num_hid_layers, batch_first=True)
        self.dropout = nn.Dropout(p=drop_prob)
        self.lin = nn.Linear(hid_size, embed_size)

    def forward(self, item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, hidden=None):
        inputs = self.get_inputs(item_inputs, skill_inputs, label_inputs)

        item_ids = self.item_embeds(item_ids)
        skill_ids = self.skill_embeds(skill_ids)
        query = torch.cat([item_ids, skill_ids], dim=-1)

        x, hidden = self.lstm(inputs, hx=hidden)
        output = (self.lin(self.dropout(x)) * query).sum(-1)
        return output, hidden

    def get_inputs(self, item_inputs, skill_inputs, label_inputs):
        item_inputs = self.item_embeds(item_inputs)
        skill_inputs = self.skill_embeds(skill_inputs)
        label_inputs = label_inputs.unsqueeze(-1).float()

        inputs = torch.cat([item_inputs, skill_inputs, item_inputs, skill_inputs], dim=-1)
        inputs[..., :self.embed_size] *= label_inputs
        inputs[..., self.embed_size:] *= 1 - label_inputs
        return inputs

    def get_KS(self, item_inputs, skill_inputs, label_inputs, hidden=None):
        inputs = self.get_inputs(item_inputs, skill_inputs, label_inputs)
        KS, _ = self.lstm(inputs, hx=hidden)
        return KS

    def repackage_hidden(self, hidden):
        # Return detached hidden for TBPTT
        return tuple((v.detach() for v in hidden))
