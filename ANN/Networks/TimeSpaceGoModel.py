import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List
from einops import rearrange
from ANN.Layers.Mamba2_layer.InferenceCache import Mamba2InferenceCache
from ANN.Networks.NetworkConfig import NetworkConfig
from ANN.Blocks.TimeSpaceBlock import TimeSpaceBlock
from ANN.Blocks.SpatialFusion_block import SpatialFusion_block
from ANN.Layers.Transformer_layer.RotaryEmbedding import RotaryEmbedding

class TimeSpaceChunk(nn.Module):
    def __init__(self, args: NetworkConfig, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device

        # TimeSpaceBlocks
        if self.args.timespaceblock_num > 0:
            self.timespace_blocks = nn.ModuleList(
                [TimeSpaceBlock(args.block_args, device) for _ in range(args.timespaceblock_num)]
            )
        else:
            self.timespace_blocks = None

    def forward(        
            self,
            x:Tensor,
            H: int,
            W: int,
            rotary_emb: Optional[RotaryEmbedding] = None,
            cache_list:Optional[List[Mamba2InferenceCache]]=None,
            initial_ssm_states_list:Optional[List[Tensor]]=None
            ) -> tuple[Tensor, List[Tensor]]:
        # 这里的输入形状为: B, S, L, D
        ssm_states_list=[]
        # 1. 处理 TimeSpaceBlocks
        if self.timespace_blocks is not None:
            for i, block in enumerate(self.timespace_blocks):
                if cache_list is not None:
                    cache = cache_list[i]
                else:
                    cache = None
                if initial_ssm_states_list is not None:
                    initial_ssm_states = initial_ssm_states_list[i]
                else:
                    initial_ssm_states = None
                x, ssm_states = block(x, H, W, rotary_emb, cache, initial_ssm_states)
                ssm_states_list.append(ssm_states)
        
        return x, ssm_states_list
        
class SpaceChunk(nn.Module):
    def __init__(self, args: NetworkConfig, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device

        # SpaceBlocks
        if self.args.spatialfusion_block_num > 0:
            self.space_blocks = nn.ModuleList(
                [SpatialFusion_block(args.block_args, device) for _ in range(args.spatialfusion_block_num)]
            )
        else:
            self.space_blocks = None

    def forward(        
            self,
            x:Tensor,
            H: int,
            W: int,
            rotary_emb: Optional[RotaryEmbedding] = None,
            ) -> Tensor:
        # 这里的输入形状为: B, S, L, D_low
        # 输出形状为: B, S, L, D_low
        # 处理空间讯息
        if self.space_blocks is not None:
            for i, block in enumerate(self.space_blocks):
                x = block(x, H, W, rotary_emb)
        return x


class TimeSpaceGoModel(nn.Module):
    def __init__(self, H:int, W:int ,device: torch.device):
        super().__init__()
        # 为了方便直接在这里配置超参数了
        self.d_model = 512
        self.value_d_model = 128

        # 配置rotary_emb，rotary_emb是更具headdim来的，headdim在Transformer的配置文件中写死为64了

        # 卷积升维
        self.conv = nn.Conv2d(
            3,
            self.d_model,
            3,
            1,
            'same',
            device=device
        )

        # 配置主干网络
        self.backbone_args = NetworkConfig(self.d_model,
                                           (H*W)*2,
                                           timespaceblock_num=20,
                                           )
        self.backbone = TimeSpaceChunk(self.backbone_args, device=device)

        # 配置策略网络
        
        self.policyhead



    def forward(        
            self,
            x:Tensor,
            H: int,
            W: int,
            rotary_emb: Optional[RotaryEmbedding] = None,
            cache_list2:Optional[List[List[Mamba2InferenceCache]]]=None,
            initial_ssm_states_list2:Optional[List[List[Tensor]]]=None
            ) -> tuple[Tensor, List[List[Tensor]]]:
        B, S, H, W, C = x.shape
        # 这里的输入形状为: x(B, S, H, W, C)
        x = rearrange(x, "b s h w c -> (b s) c h w")
        x = F.gelu(self.conv(x))
        x = rearrange(x, "(b s) d h w -> b s (h w) d", s=S)
        # x形状转变为：B, S, L, D
        initial_ssm_states_list2 = []
        x, initial_ssm_states_list = self.backbone(x,
                                                   H,
                                                   W,

                                                   )

        
        


