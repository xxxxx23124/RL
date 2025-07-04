from dataclasses import dataclass
from TransformerConfig import TransformerConfig
import torch
from torch import Tensor
from typing import Optional


@dataclass
class FlashAttentionKVCache:
    """
    存储 Flash Attention 层在自回归推理时所需的 Key-Value 状态。

    Attributes:
        key_cache (Tensor): 存储历史 Key 的张量。
                            形状: (B, H, S_max, D_head)
        value_cache (Tensor): 存储历史 Value 的张量。
                              形状: (B, H, S_max, D_head)
        seq_len (int): 当前缓存中有效序列的长度。
    """
    key_cache: Tensor
    value_cache: Tensor
    seq_len: int = 0  # 追踪当前缓存的有效长度

    def update(self, new_k: Tensor, new_v: Tensor) -> None:
        """
        用新的 key 和 value 更新缓存。

        Args:
            new_k (Tensor): 新的 key, 形状为 (B, H, S_new, D_head)。
            new_v (Tensor): 新的 value, 形状为 (B, H, S_new, D_head)。
        """
        new_seq_len = new_k.size(2)
        # 计算更新后的总长度
        updated_seq_len = self.seq_len + new_seq_len

        # 检查是否超出缓存容量
        if updated_seq_len > self.key_cache.size(2):
            raise ValueError(
                f"Cache update exceeds max sequence length. "
                f"Current len: {self.seq_len}, new tokens: {new_seq_len}, "
                f"max capacity: {self.key_cache.shape[2]}"
            )

        # 在缓存的相应位置就地更新
        self.key_cache[:, :, self.seq_len:updated_seq_len, :] = new_k
        self.value_cache[:, :, self.seq_len:updated_seq_len, :] = new_v
        
        # 更新有效长度
        self.seq_len = updated_seq_len

    def get(self) -> tuple[Tensor, Tensor]:
        """
        获取当前有效的 key 和 value 缓存部分。
        
        Returns:
            Tuple[Tensor, Tensor]: 有效的 key 和 value, 形状为 (B, H, self.seq_len, D_head)。
        """
        return (
            self.key_cache[:, :, :self.seq_len, :],
            self.value_cache[:, :, :self.seq_len, :]
        )

    @classmethod
    def alloc(cls, 
              batch_size: int, 
              config: TransformerConfig, 
              device: Optional[torch.device] = None,
              dtype: Optional[torch.dtype] = None) -> "FlashAttentionKVCache":
        """
        工厂方法：为推理开始时分配一个预先定义好最大长度的（全零）缓存。

        使用 @classmethod, 这样在子类化时也能正确工作。
        """
        kv_shape = (
            batch_size,
            config.nheads,
            config.max_seq_len,  # 预分配最大序列长度
            config.headdim
        )
        
        # 创建一个初始 seq_len 为 0 的缓存实例
        return cls(
            key_cache=torch.zeros(kv_shape, device=device, dtype=dtype),
            value_cache=torch.zeros(kv_shape, device=device, dtype=dtype),
            seq_len=0
        )