import torch
import torch.nn as nn
import torch.nn.functional as F
from model_sakt2 import future_mask, clone, attention


class NonSelfAttentionLayer(torch.nn.TransformerEncoderLayer):
    def __init__(
        self, d_model, nhead, dim_feedforward=128, dropout=0.1, activation="relu"
    ):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation)

    def forward(self, srcs, src_mask, src_key_padding_mask):
        src_query, src_key, src_value = srcs
        src2 = self.self_attn(
            src_query,
            src_key,
            src_value,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src = src_query + self.dropout1(src2)  # TODO: Test residual
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src
