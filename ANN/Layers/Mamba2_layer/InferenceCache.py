from dataclasses import dataclass
import torch
from torch import Tensor
from Mamba2Config import Mamba2Config
from typing import Optional

@dataclass
class Mamba2InferenceCache:
    """
    存储 Mamba2 层在自回归推理时所需的、固定大小的状态。

    Attributes:
        conv_state (Tensor): 1D 卷积层的状态（滑动窗口）。
                             形状: (B, D_inner + 2*D_state, D_conv)。
        ssm_state (Tensor): SSM 层的状态向量。
                            形状: (B, N_heads, D_head, D_state)。
    """
    conv_state: Tensor
    ssm_state: Tensor

    def update(self, new_conv_input: Tensor, new_ssm_state: Tensor) -> None:
        """
        用新的输入和状态更新缓存。

        Args:
            new_conv_input (Tensor): 当前时间步进入卷积层的输入，形状为 (B, D_inner + 2*D_state)。
                                     注意：这里需要 squeeze(1) 后的张量。
            new_ssm_state (Tensor): 计算出的新 SSM 状态，形状为 (B, N_heads, D_head, D_state)。
        """
        # 1. 更新卷积状态：向左滚动，并将新输入放在末尾
        # torch.roll 比手动切片拼接更高效
        self.conv_state.copy_(torch.roll(self.conv_state, shifts=-1, dims=-1))
        self.conv_state[..., -1] = new_conv_input
        
        # 2. 更新 SSM 状态：直接用新的状态覆盖旧的
        self.ssm_state.copy_(new_ssm_state)

    def get(self) -> tuple[Tensor, Tensor]:
        """
        获取当前的卷积状态和 SSM 状态。

        Returns:
            Tuple[Tensor, Tensor]: (conv_state, ssm_state)。
        """
        return self.conv_state, self.ssm_state

    @classmethod
    def alloc(cls, 
              batch_size: int, 
              config: Mamba2Config, 
              device: Optional[torch.device] = None,
              dtype: Optional[torch.dtype] = None) -> "Mamba2InferenceCache":
        """
        工厂方法：为推理开始时分配一个初始化的（全零）缓存。
        
        使用 @classmethod, 这样在子类化时也能正确工作。
        """
        conv_dim = config.d_inner + 2 * config.d_state
        conv_state_shape = (batch_size, conv_dim, config.d_conv)
        ssm_state_shape = (batch_size, config.nheads, config.headdim, config.d_state)
        
        # 使用指定的 dtype 初始化
        return cls(
            conv_state=torch.zeros(conv_state_shape, device=device, dtype=dtype),
            ssm_state=torch.zeros(ssm_state_shape, device=device, dtype=dtype)
        )
