import math

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, dim, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x: torch.Tensor):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


# Positional encodings for lower transformer encoder

class ConstPositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=100):
        super(ConstPositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        return self.pe[:, :x.size(1), :]


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, emb_dim, max_len=100):
        super(LearnablePositionalEncoding, self).__init__()
        self.emb = nn.Embedding(max_len, emb_dim)

    def forward(self, x: torch.Tensor):
        pos = torch.arange(x.shape[1], dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)
        return self.emb(pos)


class LevelEncoding(nn.Embedding):
    def __init__(self, max_level, emb_dim):
        super().__init__(max_level, emb_dim)


class SegmentEncoding(nn.Embedding):
    def __init__(self, emb_dim):
        super().__init__(3, emb_dim)
