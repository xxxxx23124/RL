from dataclasses import dataclass
import torch
from torch import Tensor
from Mamba2Config import Mamba2Config
from typing import Optional

@dataclass
class InferenceCache:
    """
    存储 Mamba2 层推理时所需的状态。

    Attributes:
        conv_state (Tensor): 1D 卷积层的状态，形状为 (B, D_inner + 2*D_state, D_conv)。
        ssm_state (Tensor): SSM 层的状态，形状为 (B, N_heads, D_head, D_state)。
    """
    conv_state: Tensor
    ssm_state: Tensor

    @classmethod
    def alloc(cls, batch_size: int, config: Mamba2Config, device: Optional[torch.device] = None) -> "InferenceCache":
        """
        工厂方法：为推理开始时分配一个初始化的（全零）缓存。
        
        使用 @classmethod, 这样在子类化时也能正确工作。
        """
        conv_state_shape = (batch_size, config.d_inner + 2 * config.d_state, config.d_conv)
        ssm_state_shape = (batch_size, config.nheads, config.headdim, config.d_state)
        
        return cls(
            conv_state=torch.zeros(conv_state_shape, device=device),
            ssm_state=torch.zeros(ssm_state_shape, device=device)
        )

# 使用示例
# config = Mamba2Config(...)
# cache = InferenceCache.alloc(batch_size=1, config=config, device=device)