import torch
import torch.nn as nn


class DKT2(nn.Module):
    def __init__(self, num_items, num_skills, hid_size, embed_size, num_hid_layers, drop_prob):
        """Deep Knowledge Tracing (https://papers.nips.cc/paper/5654-deep-knowledge-tracing.pdf)
        with some changes inspired by
        Deep Hierarchical Knowledge Tracing (https://arxiv.org/pdf/1908.02146.pdf).

        Arguments:
            num_items (int): number of items
            num_skills (int): number of knowledge points
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

        self.lin1 = nn.Linear(hid_size + embed_size, hid_size)
        self.lin2 = nn.Linear(hid_size, 1)

    def forward(self, item_inputs, skill_inputs, label_inputs, item_ids, skill_ids):
        inputs = self.get_inputs(item_inputs, skill_inputs, label_inputs)
        query = self.get_query(item_ids, skill_ids)

        x, _ = self.lstm(inputs)
        x = self.lin1(torch.cat([self.dropout(x), query], dim=-1))
        x = self.lin2(torch.relu(self.dropout(x))).squeeze(-1)
        return x

    def get_inputs(self, item_inputs, skill_inputs, label_inputs):
        item_inputs = self.item_embeds(item_inputs)
        skill_inputs = self.skill_embeds(skill_inputs)
        label_inputs = label_inputs.unsqueeze(-1).float()

        inputs = torch.cat([item_inputs, skill_inputs, item_inputs, skill_inputs], dim=-1)
        inputs[..., :self.embed_size] *= label_inputs
        inputs[..., self.embed_size:] *= 1 - label_inputs
        return inputs

    def get_query(self, item_ids, skill_ids):
        item_ids = self.item_embeds(item_ids)
        skill_ids = self.skill_embeds(skill_ids)
        query = torch.cat([item_ids, skill_ids], dim=-1)
        return query
