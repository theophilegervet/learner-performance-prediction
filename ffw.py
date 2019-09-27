import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, statistics_size, num_prev_interactions, embed_size, hid_size, num_items, drop_prob):
        super(FeedForward, self).__init__()
        self.num_prev_interactions = num_prev_interactions
        self.prev_interaction_embeds = nn.Embedding(2 * num_items, embed_size)
        self.prev_interaction_embeds.weight.requires_grad = False
        self.lin1 = nn.Linear(statistics_size + num_prev_interactions * embed_size, hid_size)
        self.lin2 = nn.Linear(hid_size, num_items)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, inputs):
        idxs = inputs[:, :self.num_prev_interactions].long()
        embeds = self.prev_interaction_embeds(idxs).view(inputs.size(0), -1)
        statistics = inputs[:, self.num_prev_interactions:]
        out = self.lin1(torch.cat((embeds, statistics), 1))
        out = self.lin2(self.dropout(F.relu(out)))
        return out