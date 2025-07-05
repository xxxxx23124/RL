from dataclasses import dataclass
from TransformerConfig import TransformerConfig
import torch
from torch import Tensor
from typing import Optional
    
@dataclass
class FlashAttentionKVCache:
    """
    存储 Flash Attention K/V 状态的动态缓存。
    能适应 batch size 的变化，同时管理序列长度。
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
    seq_len: int
    # 记录当前缓存的 batch size 和最大容量
    batch_size: int
    max_batch_size: int
    config: 'TransformerConfig'
    device: torch.device
    dtype: torch.dtype
    
    def _adapt_batch_size(self, target_batch_size: int) -> None:
        """内部方法：根据目标 batch size 调整内部状态张量。"""
        if self.batch_size == target_batch_size:
            return
            
        if target_batch_size > self.max_batch_size:
            raise ValueError(
                f"目标 batch size {target_batch_size} 超出最大预分配容量 {self.max_batch_size}。"
            )

        # 这里不需要像Mamba那样做复杂的扩展/收缩，因为Transformer的batch是独立的。
        # 我们只需在get()和update()时使用正确的切片即可。
        # 这里更新batch_size，让其他方法知道当前的上下文。
        self.batch_size = target_batch_size

    def update(self, new_k: Tensor, new_v: Tensor) -> None:
        """
        用新的 key 和 value 更新缓存，并自适应 batch size。
        """
        target_batch_size = new_k.shape[0]
        self._adapt_batch_size(target_batch_size)

        new_seq_len = new_k.shape[2]
        updated_seq_len = self.seq_len + new_seq_len

        if updated_seq_len > self.key_cache.shape[2]:
            raise ValueError("更新超出最大序列长度容量。")

        # 使用切片来确保只更新当前有效的 batch 部分
        current_k_cache = self.key_cache[:self.batch_size]
        current_v_cache = self.value_cache[:self.batch_size]

        current_k_cache[:, :, self.seq_len:updated_seq_len, :] = new_k
        current_v_cache[:, :, self.seq_len:updated_seq_len, :] = new_v
        
        self.seq_len = updated_seq_len

    def get(self) -> tuple[Tensor, Tensor]:
        """获取当前有效的 key 和 value 缓存部分。"""
        # 返回当前 batch size 和序列长度对应的部分
        return (
            self.key_cache[:self.batch_size, :, :self.seq_len, :],
            self.value_cache[:self.batch_size, :, :self.seq_len, :]
        )

    def reset(self):
        """重置序列长度，用于开始新的序列生成。"""
        self.seq_len = 0
        # 可以选择性地清零缓存内容
        self.key_cache.zero_()
        self.value_cache.zero_()

    @classmethod
    def alloc(cls, 
              config: 'TransformerConfig', 
              device: Optional[torch.device] = None,
              dtype: Optional[torch.dtype] = None) -> "FlashAttentionKVCache":
        """
        工厂方法：预分配一个最大容量的缓存。
        初始 batch_size 可以设为1或一个典型值。
        """
        kv_shape = (
            config.max_batch_size,
            config.nheads,
            config.max_seq_len,
            config.headdim
        )
        
        return cls(
            key_cache=torch.zeros(kv_shape, device=device, dtype=dtype),
            value_cache=torch.zeros(kv_shape, device=device, dtype=dtype),
            seq_len=0,
            batch_size=config.max_batch_size, # 初始时，让它等于最大值，只是初始化
            max_batch_size=config.max_batch_size,
            config=config,
            device=device,
            dtype=dtype
        )