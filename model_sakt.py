import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def future_mask(seq_length):
    future_mask = np.triu(np.ones((1, seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


def clone(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])


def attention(query, key, value, pos_embeds=None, mask=None, dropout=None):
    """Compute scaled dot product attention. 
    """
    scores = torch.matmul(query, key.transpose(-2, -1))

    if pos_embeds:
        tmp = torch.arange(scores.size(-1))
        if query.is_cuda:
            tmp = tmp.cuda()
        idxs = tmp.view(-1, 1) - tmp.view(1, -1)
        idxs = torch.clamp(idxs, 0, pos_embeds.num_embeddings - 1)
        pos = pos_embeds(idxs)
        pos_scores = torch.matmul(query.unsqueeze(-2), pos.transpose(-2, -1)).squeeze(-2)
        scores = scores + pos_scores

    scores = scores / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    prob_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        prob_attn = dropout(prob_attn)
    return torch.matmul(prob_attn, value), prob_attn
    
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, total_size, num_heads, drop_prob):
        super(MultiHeadedAttention, self).__init__()
        assert total_size % num_heads == 0
        self.total_size = total_size
        self.head_size = total_size // num_heads
        self.num_heads = num_heads
        self.linear_layers = clone(nn.Linear(total_size, total_size), 4)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, query, key, value, pos_embeds=None, mask=None):
        batch_size, seq_length = query.shape[:2]
        
        # Apply mask to all heads
        if mask is not None:
            mask = mask.unsqueeze(1)
            
        # Project inputs
        query, key, value = [l(x).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        
        # Apply attention 
        out, self.prob_attn = attention(query, key, value, pos_embeds, mask, self.dropout)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.total_size)
        
        # Project output
        return self.linear_layers[-1](out)


class SAKT(nn.Module):
    def __init__(self, num_items, num_skills, embed_size, num_attn_layers, num_heads,
                 encode_pos, drop_prob):
        """Self-attentive knowledge tracing.

        Arguments:
            num_items (int): number of items
            num_skills (int): number of skills
            embed_size (int): Input embedding and attention dot-product dimension
            num_attn_layers (int): Number of attention layers
            num_heads (int): Number of parallel attention heads
            encode_pos (bool): If True, add positional encoding
            drop_prob (float): Dropout probability
        """
        super(SAKT, self).__init__()
        self.num_items = num_items
        self.num_skills = num_skills

        self.skill_input_embeds = nn.Embedding(2 * num_skills + 1, embed_size, padding_idx=0)
        self.skill_embeds = nn.Embedding(num_skills, embed_size, padding_idx=0)
        self.pos_embeds = nn.Embedding(20, embed_size // num_heads) if encode_pos else None

        self.attn_layers = clone(MultiHeadedAttention(embed_size, num_heads, drop_prob),
                                 num_attn_layers)
        self.dropout = nn.Dropout(p=drop_prob)
        self.out = nn.Linear(embed_size, 1)
        
    def forward(self, item_inputs, skill_inputs, item_ids, skill_ids):
        input = self.skill_input_embeds(skill_inputs)
        query = self.skill_embeds(skill_ids)

        mask = future_mask(input.size(-2))
        if input.is_cuda:
            mask = mask.cuda()

        output = input
        for l in self.attn_layers:
            output = self.dropout(output + F.relu(l(query, output, output, self.pos_embeds, mask)))
        return self.out(output)
