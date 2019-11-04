import torch
import torch.nn as nn


class DHKT(nn.Module):
    def __init__(self, Q_mat, hid_size, embed_size, num_hid_layers, drop_prob):
        """Deep Hierarchical Knowledge Tracing (https://arxiv.org/pdf/1908.02146.pdf).

        Arguments:
            Q_mat (torch Tensor): Q-matrix adapted for hinge loss
            hid_size (int): hidden layer dimension
            embed_size (int): query embedding dimension
            num_hid_layers (int): number of hidden layers
            drop_prob (float): dropout probability
        """
        super(DHKT, self).__init__()
        self.Q_mat = Q_mat
        self.num_items, self.num_skills = Q_mat.shape
        self.embed_size = embed_size

        self.item_embeds = nn.Embedding(self.num_items, embed_size // 2, padding_idx=0)
        self.skill_embeds = nn.Embedding(self.num_skills, embed_size // 2, padding_idx=0)
        
        self.lstm = nn.LSTM(2 * embed_size, hid_size, num_hid_layers, batch_first=True)
        self.dropout = nn.Dropout(p=drop_prob)
        self.lin = nn.Linear(hid_size + embed_size, 1)

    def forward(self, item_inputs, skill_inputs, label_inputs, item_ids, skill_ids, hidden=None):
        item_inputs = self.item_embeds(item_inputs)
        skill_inputs = self.skill_embeds(skill_inputs)
        label_inputs = label_inputs.unsqueeze(-1).float()
        inputs = torch.cat([item_inputs, skill_inputs, item_inputs, skill_inputs], dim=-1)
        inputs[..., :self.embed_size] *= label_inputs
        inputs[..., self.embed_size:] *= 1 - label_inputs

        item_ids = self.item_embeds(item_ids)
        skill_ids = self.skill_embeds(skill_ids)
        query = torch.cat([item_ids, skill_ids], dim=-1)

        x, hidden = self.lstm(inputs, hx=hidden)
        output = self.lin(self.dropout(torch.cat((x, query), dim=-1))).squeeze(-1)
        return output, hidden

    def hinge_loss(self):
        dot_products = self.item_embeds.weight @ self.skill_embeds.weight.T
        return torch.mean(torch.clamp(1 + dot_products * self.Q_mat, min=0))

    def repackage_hidden(self, hidden):
        # Return detached hidden for TBPTT
        return tuple((v.detach() for v in hidden))
