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


def attention(query, key, value, mask=None, dropout=None):
    """Compute scaled dot product attention. 
    """
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    prob_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        prob_attn = dropout(prob_attn)
    return torch.matmul(prob_attn, value), prob_attn


def relative_attention(query, key, value, pos_key_embeds, pos_value_embeds, mask=None, dropout=None):
    """Compute scaled dot product attention with relative position embeddings.
    (https://arxiv.org/pdf/1803.02155.pdf)
    """
    assert pos_key_embeds.num_embeddings == pos_value_embeds.num_embeddings

    scores = torch.matmul(query, key.transpose(-2, -1))

    idxs = torch.arange(scores.size(-1))
    if query.is_cuda:
        idxs = idxs.cuda()
    idxs = idxs.view(-1, 1) - idxs.view(1, -1)
    idxs = torch.clamp(idxs, 0, pos_key_embeds.num_embeddings - 1)

    pos_key = pos_key_embeds(idxs).transpose(-2, -1)
    pos_scores = torch.matmul(query.unsqueeze(-2), pos_key)
    scores = scores.unsqueeze(-2) + pos_scores
    scores = scores / math.sqrt(query.size(-1))

    pos_value = pos_value_embeds(idxs)
    value = value.unsqueeze(-3) + pos_value

    if mask is not None:
        scores = scores.masked_fill(mask.unsqueeze(-2), -1e9)
    prob_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        prob_attn = dropout(prob_attn)

    output = torch.matmul(prob_attn, value).unsqueeze(-2)
    prob_attn = prob_attn.unsqueeze(-2)
    return output, prob_attn
    
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, total_size, num_heads, drop_prob):
        super(MultiHeadedAttention, self).__init__()
        assert total_size % num_heads == 0
        self.total_size = total_size
        self.head_size = total_size // num_heads
        self.num_heads = num_heads
        self.linear_layers = clone(nn.Linear(total_size, total_size), 4)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, query, key, value, encode_pos, pos_key_embeds, pos_value_embeds, mask=None):
        batch_size, seq_length = query.shape[:2]
        
        # Apply mask to all heads
        if mask is not None:
            mask = mask.unsqueeze(1)
            
        # Project inputs
        query, key, value = [l(x).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        
        # Apply attention
        if encode_pos:
            out, self.prob_attn = relative_attention(
                query, key, value, pos_key_embeds, pos_value_embeds, mask, self.dropout)
        else:
            out, self.prob_attn = attention(query, key, value, mask, self.dropout)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.total_size)
        
        # Project output
        return self.linear_layers[-1](out)


class SAKT(nn.Module):
    def __init__(self, num_items, num_skills, embed_size, num_attn_layers, num_heads,
                 encode_pos, max_pos, drop_prob, item_in, skill_in, item_out, skill_out):
        """Self-attentive knowledge tracing (https://arxiv.org/pdf/1907.06837.pdf).

        Arguments:
            num_items (int): number of items
            num_skills (int): number of skills
            embed_size (int): input embedding and attention dot-product dimension
            num_attn_layers (int): number of attention layers
            num_heads (int): number of parallel attention heads
            encode_pos (bool): if True, use relative position embeddings
            max_pos (int): number of position embeddings to use
            drop_prob (float): dropout probability
            item_in (bool): if True, use items as inputs
            skill_in (bool): if True, use skills as inputs
            item_out (bool): if True, use items as outputs
            skill_out (bool): if True, use skills as outputs
        """
        super(SAKT, self).__init__()
        self.num_items = num_items
        self.num_skills = num_skills
        self.encode_pos = encode_pos
        self.item_in = item_in
        self.skill_in = skill_in
        self.item_out = item_out
        self.skill_out = skill_out

        # Pad inputs with 0, this explains the +1
        if item_in and skill_in:
            self.item_input_embeds = nn.Embedding(2 * num_items + 1, embed_size // 2, padding_idx=0)
            self.skill_input_embeds = nn.Embedding(2 * num_skills + 1, embed_size // 2, padding_idx=0)
        elif item_in:
            self.item_input_embeds = nn.Embedding(2 * num_items + 1, embed_size, padding_idx=0)
        elif skill_in:
            self.skill_input_embeds = nn.Embedding(2 * num_skills + 1, embed_size, padding_idx=0)

        if item_out and skill_out:
            self.item_query_embeds = nn.Embedding(num_items, embed_size // 2)
            self.skill_query_embeds = nn.Embedding(num_items, embed_size // 2)
        elif item_out:
            self.item_query_embeds = nn.Embedding(num_items, embed_size)
        elif skill_out:
            self.skill_query_embeds = nn.Embedding(num_items, embed_size)

        self.pos_key_embeds = nn.Embedding(max_pos, embed_size // num_heads)
        self.pos_value_embeds = nn.Embedding(max_pos, embed_size // num_heads)

        self.attn_layers = clone(MultiHeadedAttention(embed_size, num_heads, drop_prob), num_attn_layers)
        self.dropout = nn.Dropout(p=drop_prob)
        self.out = nn.Linear(embed_size, 1)
        
    def forward(self, item_inputs, skill_inputs, item_ids, skill_ids):
        if (item_inputs is not None) and (skill_inputs is not None):
            item_inputs = self.item_input_embeds(item_inputs)
            skill_inputs = self.skill_input_embeds(skill_inputs)
            x, y, z = item_inputs.size()
            input = torch.zeros(x, y, z * 2)
            if item_inputs.is_cuda:
                input = input.cuda()
            input[..., ::2] = item_inputs
            input[..., 1::2] = skill_inputs
        elif (item_inputs is not None):
            input = self.item_input_embeds(item_inputs)
        elif (skill_inputs is not None):
            input = self.skill_input_embeds(skill_inputs)

        if (item_ids is not None) and (skill_ids is not None):
            item_query = self.item_query_embeds(item_ids)
            skill_query = self.skill_query_embeds(skill_ids)
            x, y, z = item_query.size()
            query = torch.zeros(x, y, z * 2)
            if item_inputs.is_cuda:
                query = query.cuda()
            query[..., ::2] = item_query
            query[..., 1::2] = skill_query
        elif (item_ids is not None):
            query = self.item_query_embeds(item_ids)
        elif (skill_ids is not None):
            query = self.skill_query_embeds(skill_ids)

        mask = future_mask(input.size(-2))
        if input.is_cuda:
            mask = mask.cuda()

        output = input
        for l in self.attn_layers:
            residual = l(query, output, output, self.encode_pos, self.pos_key_embeds,
                         self.pos_value_embeds, mask)
            output = self.dropout(output + F.relu(residual))
        return self.out(output)


