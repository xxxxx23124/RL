import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ANN.Layers.Transformer_layer.TransformerConfig import TransformerConfig
from ANN.Layers.Transformer_layer.RotaryEmbedding import RotaryEmbedding
from ANN.Layers.Transformer_layer.InferenceCache import FlashAttentionKVCache
from einops import rearrange
from typing import Optional

class SelfAttention(nn.Module):
    def __init__(self, args: TransformerConfig, device: torch.device):
        super(SelfAttention, self).__init__()
        self.args = args
        self.device = device
        self.Wqkv = nn.Linear(args.d_model, 3 * args.d_model, device=device)
        self.out_proj = nn.Linear(args.d_model, args.d_model, device=device)

    def forward(self, 
                x: Tensor,
                rotary_emb: Optional[RotaryEmbedding] = None,
                kv_cache: Optional[FlashAttentionKVCache] = None
                ) -> Tensor:

        B, S, D = x.shape
        use_causal_mask = (kv_cache is None) and S > 1

        query, key, value = self.Wqkv(x).chunk(3, dim=-1)

        query = rearrange(query, 'b s (h d) -> b h s d', h=self.args.nheads)
        key = rearrange(key, 'b s (h d) -> b h s d', h=self.args.nheads)
        value = rearrange(value, 'b s (h d) -> b h s d', h=self.args.nheads)

        # 注意，如果在推理阶段要先rotary_emb再更新kvcache
        if rotary_emb is not None:
            if kv_cache is not None:
                seq_len_offset = kv_cache.seq_len
            else: 
                seq_len_offset = 0
            query = rotary_emb(query, seq_len=S, seq_len_offset=seq_len_offset)
            key = rotary_emb(key, seq_len=S, seq_len_offset=seq_len_offset)
        
        if kv_cache is not None:
            kv_cache.update(key, value)
            key_full, value_full = kv_cache.get()
        else:
            key_full, value_full = key, value
            
        attn_output = F.scaled_dot_product_attention(query, key_full, value_full, is_causal=use_causal_mask)
        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')
        return self.out_proj(attn_output)