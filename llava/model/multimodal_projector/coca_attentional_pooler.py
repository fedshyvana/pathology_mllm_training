import logging
import pdb
from collections import OrderedDict
from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import nn, einsum
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many

class AttentionalPoolProjector(nn.Module):
    def __init__(
            self,
            embed_dim,
            context_dim,
            projector = None,
            n_head = 8,
            n_queries = 128,
            norm_layer: Callable = nn.LayerNorm):
        super().__init__()
        self.attn_pool = AttentionalPooler(d_model=embed_dim, 
                                            context_dim=context_dim, 
                                            n_head=n_head, 
                                            n_queries=n_queries)
        self.ln = norm_layer(embed_dim)
        self.proj = projector if projector else nn.Identity()
    
    def forward(self, x: torch.Tensor):
        tokens = self.attn_pool(x)
        tokens = self.ln(tokens)
        tokens = self.proj(tokens)
        return tokens

class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = nn.LayerNorm
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))

        dim_head = d_model // n_head

        self.scale = dim_head ** -0.5
        self.heads = n_head
        inner_dim = dim_head * n_head

        self.ln_k = norm_layer(context_dim)
        self.ln_q = norm_layer(d_model)

        self.to_q = nn.Linear(d_model, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')

        q = repeat(self.query, 'n d -> b m n d', b=x.shape[0], m=x.shape[1])

        x = self.ln_k(x)
        q = self.ln_q(q)
        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(q)

        kv_input = x
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h=h)

        q = q * self.scale

        # attention
        sim = einsum('... i d, ... j d  -> ... i j', q, k)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h=h)
        return self.to_out(out).squeeze(dim=1)
