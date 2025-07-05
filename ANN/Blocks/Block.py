import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from ANN.Blocks.InferenceCache import BlockInferenceCache
from ANN.Blocks.BlockConfig import BlockConfig
from ANN.Layers.FeedForward_layer.SwiGLU import SwiGLUFeedForward
from ANN.Layers.Mamba2_layer.mamba2_block import Mamba2
from ANN.Layers.Transformer_layer.SelfAttention_block import SelfAttention
from ANN.Layers.Norm_layer.RMSNorm import RMSNorm
from ANN.Layers.Transformer_layer.RotaryEmbedding import RotaryEmbedding

class Block(nn.Module):
    def __init__(self, args: BlockConfig, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device
        self.backbone = nn.ModuleDict(
            dict(
                mamba2_blocks=nn.ModuleList(
                    [
                        nn.ModuleDict(
                            dict(
                            mixer=Mamba2(args=args.mamba2_args, device=device),
                            norm=RMSNorm(args.d_model),
                            )
                        )
                        for _ in range(args.mamba2_layers)
                    ]
                ),
                attention_blocks=nn.ModuleList(
                    [
                        nn.ModuleDict(
                            dict(
                            self_attention=SelfAttention(args=args.transformer_args, device=device),
                            norm_attention=RMSNorm(args.d_model),
                            ffd=SwiGLUFeedForward(args=args.feedforward_args, device=device),
                            norm_ffd=RMSNorm(args.d_model),
                            )
                        )
                        for _ in range(args.self_attention_layers)
                    ]
                ),
            )
        )
    
    def forward(        
            self,
            x:Tensor,
            rotary_emb: Optional[RotaryEmbedding] = None,
            cache:Optional[BlockInferenceCache]=None
            ) -> Tensor:
        if cache:
            mamba2_cache, attention_cache = cache.get()
        else:
            mamba2_cache, attention_cache = None, None
        
        for i, mamba2_block in enumerate(self.backbone.mamba2_blocks):
            y = mamba2_block.mixer(mamba2_block.norm(x), mamba2_cache[i])
            x = y + x
        
        for i, attention_block in enumerate(self.backbone.self_attention_blocks):
            y = attention_block.self_attention(attention_block.norm_attention(x), rotary_emb, attention_cache[i])
            x = y + x
            y = attention_block.ffd(attention_block.norm_ffd(x))
            x = y + x
        return x
        