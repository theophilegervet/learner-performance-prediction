import torch
import torch.nn as nn


class DKT3(nn.Module):
    def __init__(self, num_items, num_KPs, q_mat, hid_size, embed_size, num_hid_layers, drop_prob):
        """DKT2 + embedding regularization.
        
        Arguments:
            num_items (int): number of items
            num_KPs (int): number of knowledge points
            q_mat (torch Tensor): Q matrix
            hid_size (int): hidden layer dimension
            embed_size (int): query embedding dimension
            num_hid_layers (int): number of hidden layers
            drop_prob (float): dropout probability
        """
        super(DKT3, self).__init__()
        self.embed_size = embed_size
        self.q_mat = q_mat.cuda().float()

        self.item_embeds = nn.Embedding(num_items + 1, embed_size // 2, padding_idx=0)
        self.KP_embeds = nn.Embedding(num_KPs + 1, embed_size // 2, padding_idx=0)

        self.lstm = nn.LSTM(2 * embed_size, hid_size, num_hid_layers, batch_first=True)
        self.dropout = nn.Dropout(p=drop_prob)

        self.lin1 = nn.Linear(hid_size + embed_size, hid_size)
        self.lin2 = nn.Linear(hid_size, 1)

    def forward(self, item_inputs, KP_inputs, label_inputs, item_ids, KP_ids):
        inputs = self.get_inputs(item_inputs, KP_inputs, label_inputs)
        query = self.get_query(item_ids, KP_ids)

        x, _ = self.lstm(inputs)
        x = self.lin1(torch.cat([self.dropout(x), query], dim=-1))
        x = self.lin2(torch.relu(self.dropout(x))).squeeze(-1)
        return x

    def get_inputs(self, item_inputs, KP_inputs, label_inputs):
        item_inputs = self.item_embeds(item_inputs)
        KP_inputs = self.KP_embeds(KP_inputs)
        label_inputs = label_inputs.unsqueeze(-1).float()

        inputs = torch.cat([item_inputs, KP_inputs, item_inputs, KP_inputs], dim=-1)
        inputs[..., :self.embed_size] *= label_inputs
        inputs[..., self.embed_size:] *= 1 - label_inputs
        return inputs

    def get_query(self, item_ids, KP_ids):
        item_ids = self.item_embeds(item_ids)
        KP_ids = self.KP_embeds(KP_ids)
        query = torch.cat([item_ids, KP_ids], dim=-1)
        return query

    def regularization(self):
        item_norms = torch.norm(self.item_embeds.weight, 2, dim=1, keepdim=True)
        KP_norms = torch.norm(self.KP_embeds.weight, 2, dim=1, keepdim=True)
        loss = (item_norms ** 2).mean() + (KP_norms ** 2).mean()
        loss -= 2 * (self.item_embeds.weight @ self.KP_embeds.weight.T).mean()
        return loss