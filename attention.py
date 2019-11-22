import torch
import math
import torch.nn as nn
import numpy as np


def attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -np.inf)
    att = nn.Softmax(-1)(scores)
    if dropout is not None:
        att = dropout(att)
    return torch.matmul(scores, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout

        self.d_k = d_model // num_heads
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = q.size(0)
        query, key, value = [linear(x).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
                             for linear, x in zip(self.linears, (q, k, v))]
        x = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)
        return self.linears[-1](x)


# x = torch.randn(2, 4, 12)
# ma = MultiHeadAttention(12, 3, 0.8)
# mask = torch.randint(0, 2, (2, 4, 4))
# o = ma(x, x, x, mask)
# print(x.size())
# print(o.size())