import torch
import torch.nn as nn
import torch.nn.functional as F


class KQN(nn.Module):
    def __init__(self, num_items, num_skills, hid_size, embed_size, num_hid_layers, drop_prob,
                 item_in, skill_in, item_out, skill_out):
        """Knowledge query network (https://arxiv.org/pdf/1908.02146.pdf).

        Arguments:
            num_items (int): number of items
            num_skills (int): number of skills
            hid_size (int): hidden layer dimension
            embed_size (int): query embedding dimension
            num_hid_layers (int): number of hidden layers
            drop_prob (float): dropout probability
            item_in (bool): if True, use items as inputs
            skill_in (bool): if True, use skills as inputs
            item_out (bool): if True, use items as outputs
            skill_out (bool): if True, use skills as outputs
        """
        super(KQN, self).__init__()
        self.num_items = num_items
        self.num_skills = num_skills
        self.item_in = item_in
        self.skill_in = skill_in
        self.item_out = item_out
        self.skill_out = skill_out
        self.input_size = (2 * num_items + 1) * item_in + (2 * num_skills + 1) * skill_in

        if item_out and skill_out:
            self.item_query_embeds = nn.Embedding(num_items, embed_size // 2)
            self.skill_query_embeds = nn.Embedding(num_items, embed_size // 2)
        elif item_out:
            self.item_query_embeds = nn.Embedding(num_items, embed_size)
        elif skill_out:
            self.skill_query_embeds = nn.Embedding(num_items, embed_size)

        self.lstm = nn.LSTM(self.input_size, hid_size, num_hid_layers, batch_first=True)
        self.dropout = nn.Dropout(p=drop_prob)
        self.lin = nn.Linear(hid_size, embed_size)

    def forward(self, item_inputs, skill_inputs, item_ids, skill_ids, hidden=None):
        # Pad inputs with 0, this explains the +1
        if (item_inputs is not None) and (skill_inputs is not None):
            item_onehots = F.one_hot(item_inputs, 2 * self.num_items + 1).float()
            skill_onehots = F.one_hot(skill_inputs, 2 * self.num_skills + 1).float()
            input = torch.cat((item_onehots, skill_onehots), -1)
        elif (item_inputs is not None):
            input = F.one_hot(item_inputs, 2 * self.num_items + 1).float()
        elif (skill_inputs is not None):
            input = F.one_hot(skill_inputs, 2 * self.num_skills + 1).float()

        if (item_ids is not None) and (skill_ids is not None):
            item_query = self.item_query_embeds(item_ids)
            skill_query = self.skill_query_embeds(skill_ids)
            x, y, z = item_query.size()
            query = torch.zeros(x, y, z * 2)
            if item_ids.is_cuda:
                query = query.cuda()
            query[..., ::2] = item_query
            query[..., 1::2] = skill_query
        elif (item_ids is not None):
            query = self.item_query_embeds(item_ids)
        elif (skill_ids is not None):
            query = self.skill_query_embeds(skill_ids)

        x, hidden = self.lstm(input, hx=hidden)
        knowledge_state = self.lin(self.dropout(x))
        output = (knowledge_state * query).sum(dim=-1)
        return output, hidden

    def repackage_hidden(self, hidden):
        # Return detached hidden for TBPTT
        return tuple((v.detach() for v in hidden))