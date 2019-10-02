import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, counts_size, num_prev_interactions, num_skills, num_outputs,
                 hid_size, drop_prob):
        super(FeedForward, self).__init__()
        self.num_prev_interactions = num_prev_interactions
        self.num_skills = num_skills
        self.lin1 = nn.Linear(counts_size + 2 * num_prev_interactions * num_skills, hid_size)
        self.lin2 = nn.Linear(hid_size, num_outputs)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, inputs):
        prev_ids = inputs[:, :self.num_prev_interactions].long()
        prev_interactions = F.one_hot(prev_ids, 2 * self.num_skills).float()
        prev_interactions = prev_interactions.view(inputs.size(0), -1)
        counts = inputs[:, self.num_prev_interactions:]
        out = self.lin1(torch.cat((prev_interactions, counts), 1))
        out = self.lin2(self.dropout(F.relu(out)))
        return out