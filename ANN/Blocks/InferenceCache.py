from dataclasses import dataclass
import torch
from torch import Tensor
from BlockConfig import BlockConfig
from ANN.Layers.Mamba2_layer.InferenceCache import Mamba2InferenceCache
from ANN.Layers.Transformer_layer.InferenceCache import FlashAttentionKVCache
from typing import Optional

@dataclass
class BlockInferenceCache:
    mamba2_cashe_list:list = []
    attention_cashe_list:list = []
    @classmethod
    def alloc(cls, 
              batch_size: int, 
              config: BlockConfig, 
              device: Optional[torch.device] = None,
              dtype: Optional[torch.dtype] = None) -> "BlockInferenceCache":
        
        mamba2_cashe_list_ = [
            Mamba2InferenceCache.alloc(batch_size=batch_size,config=config.mamba2_args, device=device, dtype=dtype)
            for _ in range(config.mamba2_layers)
        ]
        attention_cashe_list_ = [
            FlashAttentionKVCache.alloc(batch_size=batch_size,config=config.transformer_args, device=device, dtype=dtype)
            for _ in range(config.self_attention_layers)
        ]
        
        # 使用指定的 dtype 初始化
        return cls(
            mamba2_cashe_list=mamba2_cashe_list_,
            attention_cashe_list=attention_cashe_list_
        )
