import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    论文: https://arxiv.org/abs/1910.07467
    Args:
        dim (int): 输入特征的维度。
        eps (float): 为保证数值稳定性加在分母上的一个很小的值。
    """
    def __init__(self, dim: int, device: torch.device, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.device = device
        # 'weight' 是 RMSNorm 的可学习增益参数
        self.weight = nn.Parameter(torch.ones(dim, device=device))

    def forward(self, x: Tensor, z: Optional[Tensor] = None) -> Tensor:
        """
        前向传播。

        Args:
            x (Tensor): 输入张量。
            z (Optional[Tensor]): 可选的门控输入张量。如果提供，
                                  会先执行 x = x * silu(z)。

        Returns:
            Tensor: 归一化后的输出张量。
        """
        # 可选的门控机制
        if z is not None:
            # 使用官方的 SiLU 实现
            x = x * F.silu(z)

        # 调用 PyTorch 内置的高效 RMSNorm
        # F.rms_norm 会处理归一化和乘以 weight 的所有操作
        # 注意：这里的输入x需要是 float 或 bfloat16 类型
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)
