import torch
import torch.nn as nn
import torch.nn.functional as F


class DKT1(nn.Module):
    def __init__(self, num_items, num_skills, hid_size, num_hid_layers, drop_prob,
                 item_in, skill_in, item_out, skill_out):
        """Deep knowledge tracing (https://papers.nips.cc/paper/5654-deep-knowledge-tracing.pdf).

        Arguments:
            num_items (int): number of items
            num_skills (int): number of skills
            hid_size (int): hidden layer dimension
            num_hid_layers (int): number of hidden layers
            drop_prob (float): dropout probability
            item_in (bool): if True, use items as inputs
            skill_in (bool): if True, use skills as inputs
            item_out (bool): if True, use items as outputs
            skill_out (bool): if True, use skills as outputs
        """
        super(DKT1, self).__init__()
        self.num_items = num_items
        self.num_skills = num_skills
        self.item_in = item_in
        self.skill_in = skill_in
        self.item_out = item_out
        self.skill_out = skill_out
        self.input_size = (2 * num_items + 1) * item_in + (2 * num_skills + 1) * skill_in
        self.output_size = num_items * item_out + num_skills * skill_out
        
        self.lstm = nn.LSTM(self.input_size, hid_size, num_hid_layers, batch_first=True)
        self.dropout = nn.Dropout(p=drop_prob)
        self.out = nn.Linear(hid_size, self.output_size)

    def forward(self, item_inputs, skill_inputs, hidden=None):
        # Pad inputs with 0, this explains the +1
        if (item_inputs is not None) and (skill_inputs is not None):
            item_onehots = F.one_hot(item_inputs, 2 * self.num_items + 1).float()
            skill_onehots = F.one_hot(skill_inputs, 2 * self.num_skills + 1).float()
            input = torch.cat((item_onehots, skill_onehots), -1)
        elif (item_inputs is not None):
            input = F.one_hot(item_inputs, 2 * self.num_items + 1).float()
        elif (skill_inputs is not None):
            input = F.one_hot(skill_inputs, 2 * self.num_skills + 1).float()

        output, hidden = self.lstm(input, hx=hidden)
        return self.out(self.dropout(output)), hidden

    def repackage_hidden(self, hidden):
        # Return detached hidden for TBPTT
        return tuple((v.detach() for v in hidden))
