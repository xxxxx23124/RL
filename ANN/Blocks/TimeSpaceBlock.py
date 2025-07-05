import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from ANN.Layers.Mamba2_layer.InferenceCache import Mamba2InferenceCache
from ANN.Blocks.BlockConfig import BlockConfig
from ANN.Layers.Mamba2_layer.mamba2_block import Mamba2
from ANN.Layers.Transformer_layer.IntraFrameAttention_layer import SpatialFusion_Layer
from ANN.Layers.Transformer_layer.RotaryEmbedding import RotaryEmbedding

class TimeSpaceBlock(nn.Module):
    def __init__(self, args: BlockConfig, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device
        self.time = Mamba2(args=args.mamba2_args, device=device)
        self.space = SpatialFusion_Layer(args=args.transformer_args, device=device)
    
    def forward(        
            self,
            x:Tensor,
            H: int,
            W: int,
            rotary_emb: Optional[RotaryEmbedding] = None,
            cache:Optional[Mamba2InferenceCache]=None
            ) -> Tensor:
        # 输入形状x (B, S, L, D) B是batchsize, S是时间步timestep, L是输入图像的大小H*W, D是模型内在维度
        # 这俩模块内部就自带残差连接了
        x = self.space(x, H, W, rotary_emb)
        x = self.time(x, cache)
        return x

        