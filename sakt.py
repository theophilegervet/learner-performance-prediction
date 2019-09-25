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
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    prob_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        prob_attn = dropout(prob_attn)
    return torch.matmul(prob_attn, value), prob_attn


def positional_encoding(seq_length, embed_size):
    """Create sinusoidal positional encoding to be added to embedding, each dimension is a sine
    wave with a different frequency and offset.
    """
    pe = torch.zeros(seq_length, embed_size, dtype=torch.float)
    position = torch.arange(seq_length).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(math.log(1e4) / embed_size))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe
    
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, input_size, output_size, num_heads, drop_prob):
        super(MultiHeadedAttention, self).__init__()
        assert output_size % num_heads == 0
        self.output_size = output_size
        self.head_size = output_size // num_heads
        self.num_heads = num_heads
        self.input_linears = clone(nn.Linear(input_size, output_size), 3)
        self.output_linear = nn.Linear(output_size, output_size)
        self.dropout = nn.Dropout(p=drop_prob)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_length = query.size()[:2]
        
        # Apply mask to all heads
        if mask is not None:
            mask = mask.unsqueeze(1)
            
        # Project inputs
        query, key, value = [l(x).view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
                             for l, x in zip(self.input_linears, (query, key, value))]
        
        # Apply attention 
        out, self.prob_attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.output_size)
        
        # Apply non-linearity
        out = self.dropout(F.relu(self.output_linear(out)))
        return out


class SAKT(nn.Module):
    """Self-attentive knowledge tracing.
    
    Arguments:
            num_items (int): Number of items
            embed_inputs (bool): If True embed inputs, else one hot encoding
            embed_size (int): Input embedding dimension
            hid_size (int): Attention dot-product dimension
            num_heads (int): Number of parallel attention heads
            encode_pos (bool): If True, add positional encoding
            drop_prob (float): Dropout probability
    """
    def __init__(self, num_items, embed_inputs, embed_size, hid_size, num_heads, encode_pos, drop_prob):
        super(SAKT, self).__init__()
        self.embed_inputs = embed_inputs
        self.encode_pos = encode_pos
        
        if self.embed_inputs:
            self.input_embeds = nn.Embedding(2 * num_items + 1, embed_size, padding_idx=0)
            self.input_embeds.weight.requires_grad = False
            self.attn = MultiHeadedAttention(embed_size, hid_size, num_heads, drop_prob)
        else:
            self.attn = MultiHeadedAttention(2 * num_items + 1, hid_size, num_heads, drop_prob)
        
        self.out = nn.Linear(hid_size, num_items)
        
    def forward(self, inputs):
        if self.embed_inputs:
            embeds = self.input_embeds(inputs)
        else:
            embeds = F.one_hot(inputs, 2 * self.num_items + 1).float()

        if self.encode_pos:
            pe = positional_encoding(embeds.size(-2), embeds.size(-1))
            if inputs.is_cuda:
                pe = pe.cuda()
            embeds = embeds + pe

        mask = future_mask(inputs.size(1))
        if inputs.is_cuda:
            mask = mask.cuda()

        out = self.attn(embeds, embeds, embeds, mask)
        return self.out(out)