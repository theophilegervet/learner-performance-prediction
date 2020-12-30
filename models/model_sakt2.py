import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_sakt import future_mask, clone, attention, relative_attention, MultiHeadedAttention

class SAKT(nn.Module):
    def __init__(self, num_items, num_skills, embed_size, num_attn_layers, num_heads,
                 encode_pos, max_pos, drop_prob):
        """Self-attentive knowledge tracing.

        Arguments:
            num_items (int): number of items
            num_skills (int): number of skills
            embed_size (int): input embedding and attention dot-product dimension
            num_attn_layers (int): number of attention layers
            num_heads (int): number of parallel attention heads
            encode_pos (bool): if True, use relative position embeddings
            max_pos (int): number of position embeddings to use
            drop_prob (float): dropout probability
        """
        super(SAKT, self).__init__()
        self.embed_size = embed_size
        self.encode_pos = encode_pos

        if 1: 
            self.item_embeds = nn.Embedding(num_items + 1, embed_size // 2, padding_idx=0)
            self.skill_embeds = nn.Embedding(num_skills + 1, embed_size // 2, padding_idx=0)
        else:
            self.item_embeds = nn.Embedding(num_items + 1, embed_size, padding_idx=0)
            self.skill_embeds = nn.Embedding(num_skills + 1, embed_size, padding_idx=0)

        self.pos_key_embeds = nn.Embedding(max_pos, embed_size // num_heads)
        self.pos_value_embeds = nn.Embedding(max_pos, embed_size // num_heads)

        self.lin_in = nn.Linear(2 * embed_size, embed_size)
        self.attn_layers = clone(MultiHeadedAttention(embed_size, num_heads, drop_prob), num_attn_layers)
        self.layer_norms = clone(nn.LayerNorm(embed_size), num_attn_layers)
        self.dropouts = clone(nn.Dropout(p=drop_prob), num_attn_layers)
        self.lin_out = nn.Linear(embed_size, 1)
        # self.lin_out = nn.Linear(embed_size, num_skills)
        
    def get_inputs(self, item_inputs, skill_inputs, label_inputs):
        if 1:
            item_inputs = self.item_embeds(item_inputs)
        skill_inputs = self.skill_embeds(skill_inputs)
        label_inputs = label_inputs.unsqueeze(-1).float()

        if 1:
            inputs = torch.cat([item_inputs, skill_inputs, item_inputs, skill_inputs], dim=-1)
        else:
            inputs = torch.cat([skill_inputs, skill_inputs], dim=-1)
        inputs[..., :self.embed_size] *= label_inputs
        inputs[..., self.embed_size:] *= 1 - label_inputs
        return inputs # Interaction: For Key and Value

    def get_query(self, item_ids, skill_ids):
        if 1:
            item_ids = self.item_embeds(item_ids)
            skill_ids = self.skill_embeds(skill_ids)
            query = torch.cat([item_ids, skill_ids], dim=-1)
            return query
        else:
            skill_ids = self.skill_embeds(skill_ids)
            return skill_ids # Exercise: For Query

    def forward(self, item_inputs, skill_inputs, label_inputs, item_ids, skill_ids):
        inputs = self.get_inputs(item_inputs, skill_inputs, label_inputs)
        inputs = self.lin_in(inputs)
        # inputs = F.relu(self.lin_in(inputs))

        query = self.get_query(item_ids, skill_ids)

        mask = future_mask(inputs.size(-2))
        if inputs.is_cuda:
            mask = mask.cuda()

        # outputs = self.layer_norms[0](
        #     self.dropouts[0](
        #         self.attn_layers[0](
        #             query, inputs, inputs, self.encode_pos,
        #             self.pos_key_embeds, self.pos_value_embeds, mask)))
        
        outputs = self.dropouts[0](self.attn_layers[0](
            query, inputs, inputs, self.encode_pos,
                    self.pos_key_embeds, self.pos_value_embeds, mask))

        for i, l in enumerate(self.attn_layers[1:]):
            residual = l(outputs, outputs, outputs, self.encode_pos, self.pos_key_embeds,
                         self.pos_value_embeds, mask)
            outputs = self.dropouts[i+1](outputs + F.relu(residual))

        return self.lin_out(outputs)