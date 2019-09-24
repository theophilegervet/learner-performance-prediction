import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, input_size, hid_size, num_items, drop_prob):
        super(FeedForward, self).__init__()
        self.lin1 = nn.Linear(input_size, hid_size)
        self.lin2 = nn.Linear(hid_size, num_items)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, inputs):
        return self.lin2(self.dropout(F.relu(self.lin1(inputs))))