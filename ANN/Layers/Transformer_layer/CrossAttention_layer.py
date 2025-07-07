import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from TransformerConfig import TransformerConfig
from InferenceCache import FlashAttentionKVCache
from einops import rearrange
from typing import Optional

class CrossAttention(nn.Module):
    def __init__(self, args: TransformerConfig, device: torch.device):
        super(CrossAttention, self).__init__()
        self.args = args
        self.device = device
        self.q_proj = nn.Linear(args.d_model, args.d_model, device=device)
        self.Wkv = nn.Linear(args.d_model, 2 * args.d_model, device=device)
        self.out_proj = nn.Linear(args.d_model, args.d_model, device=device)
    
    def prepare_encoder_kv(self, encoder_output: Tensor) -> FlashAttentionKVCache:
        key_encoder, value_encoder = self.Wkv(encoder_output).chunk(2, dim=-1)
        key_encoder = rearrange(key_encoder, 'b s (h d) -> b h s d', h=self.args.nheads)
        value_encoder = rearrange(value_encoder, 'b s (h d) -> b h s d', h=self.args.nheads)
        B, H, S_enc, D_head = key_encoder.shape
        cache = FlashAttentionKVCache.alloc(B, S_enc, self.args, key_encoder.device, key_encoder.dtype)
        cache.update(key_encoder, value_encoder)
        return cache

    def forward(self, 
                query: Tensor,
                encoder_kvCache: FlashAttentionKVCache,
                ) -> Tensor:
        query = self.q_proj(query)
        query = rearrange(query, 'b s (h d) -> b h s d', h=self.args.nheads)
        key_encoder, value_encoder = encoder_kvCache.get()
        attn_output = F.scaled_dot_product_attention(query, key_encoder, value_encoder, is_causal=False)
        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
        return self.out_proj(attn_output)