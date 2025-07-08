import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
from typing import Optional

from ANN.Layers.Norm_layer.RMSNorm import RMSNorm
from ANN.Layers.FeedForward_layer.SwiGLUMlp import SwiGLUFeedForward
from ANN.Layers.Transformer_layer.RotaryEmbedding import RotaryEmbedding
from ANN.Layers.Transformer_layer.TransformerConfig import TransformerConfig
from ANN.Layers.FeedForward_layer.FeedForwardConfig import FeedForwardConfig

class SelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        self.Wqkv = nn.Linear(config.d_model, 3 * config.d_model, device=device)
        self.out_proj = nn.Linear(config.d_model, config.d_model, device=device)

    def forward(self, x: Tensor, rotary_emb: Optional[RotaryEmbedding] = None) -> Tensor:
        B, L, D = x.shape

        q, k, v = self.Wqkv(x).chunk(3, dim=-1)

        q = rearrange(q, 'b l (h d) -> b h l d', h=self.config.nheads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.config.nheads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.config.nheads)

        if rotary_emb is not None:
            q = rotary_emb(q, L)
            k = rotary_emb(k, L)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        attn_output = rearrange(attn_output, 'b h l d -> b l (h d)')
        return self.out_proj(attn_output)

class SpatialFusion_block(nn.Module):
    def __init__(self, config: TransformerConfig, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        # 1. 第一个规范化层 (在分支前)
        self.pre_fusion_norm = RMSNorm(config.d_model, device)

        # 2. 全局路径 (Transformer Self-Attention)
        self.attention = SelfAttention(config, device)

        # 3. 局部路径 (CNN Branch)
        self.cnn_conv = nn.Conv2d(
            config.d_model, config.d_model,
            kernel_size=config.cnn_kernel_size,
            padding='same',
            groups=config.nheads,
            device=device
        )
        
        # 4. 前馈网络 (MLP)
        self.mlp_norm = RMSNorm(config.d_model, device) # MLP前的Norm
        self.mlp = SwiGLUFeedForward(FeedForwardConfig(config.d_model))

    def forward(self, x: Tensor, H: int, W: int, rotary_emb: Optional[RotaryEmbedding] = None) -> Tensor:
        B, S, L, D = x.shape
        original_shape = (B, S, L, D)
        x_reshaped = x.reshape(B * S, L, D)

        # --- 第一个残差连接 ---
        residual = x_reshaped
        
        # 1. 预归一化 (在分支前进行一次归一化)
        x_norm = self.pre_fusion_norm(x_reshaped)

        # --- 并行处理 ---
        # 路径1: 全局Transformer路径
        attn_out = self.attention(x_norm, rotary_emb)
        
        # 路径2: 局部CNN路径
        x_cnn_in = rearrange(x_norm, 'bs (h w) d -> bs d h w', h=H, w=W)
        cnn_out_img = self.cnn_conv(x_cnn_in)
        # 这里在卷积后可以加一个激活函数，如GELU
        cnn_out_seq = rearrange(F.gelu(cnn_out_img), 'bs d h w -> bs (h w) d')
        
        # --- 融合与第一个残差连接 ---
        fused_out = residual + attn_out + cnn_out_seq
        
        # --- MLP模块与第二个残差连接 ---
        residual = fused_out
        mlp_out = self.mlp(self.mlp_norm(fused_out))
        final_out = residual + mlp_out
        
        return final_out.view(original_shape)