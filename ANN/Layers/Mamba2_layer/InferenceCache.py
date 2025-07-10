from dataclasses import dataclass
import torch
from torch import Tensor
from ANN.Layers.Mamba2_layer.Mamba2Config import Mamba2Config

@dataclass
class Mamba2InferenceCache:
    """
    存储 Mamba2 层在自回归推理时所需状态的动态缓存。
    能够适应推理过程中 batch size 的变化 (e.g., from B to B*L)。

    Attributes:
        conv_state (Tensor): 1D 卷积层的状态（滑动窗口）。
                             形状: (B/BL, D_inner + 2*D_state, D_conv)。
        ssm_state (Tensor): SSM 层的状态向量。
                            形状: (B/BL, N_heads, D_head, D_state)。
    """
    conv_state: Tensor
    ssm_state: Tensor
    # 记录当前缓存对应的 batch size
    batch_size: int
    config: 'Mamba2Config'
    device: torch.device
    dtype: torch.dtype

    def _adapt_batch_size(self, target_batch_size: int) -> None:
        """内部方法：根据目标 batch size 调整内部状态张量。"""
        if self.batch_size == target_batch_size:
            return

        # Case 1: 扩展 (e.g., B -> B*L)
        # B*L 总是 B 的整数倍，倍数为 L
        if target_batch_size > self.batch_size:
            if target_batch_size % self.batch_size != 0:
                raise ValueError(f"无法从 {self.batch_size} 扩展到 {target_batch_size}，不是整数倍。")
            factor = target_batch_size // self.batch_size
            self.conv_state = self.conv_state.repeat_interleave(factor, dim=0)
            self.ssm_state = self.ssm_state.repeat_interleave(factor, dim=0)
        
        # Case 2: 收缩 (e.g., B*L -> B)
        # 我们假设收缩意味着从每个 L 的块中取第一个元素
        else:
            if self.batch_size % target_batch_size != 0:
                 raise ValueError(f"无法从 {self.batch_size} 收缩到 {target_batch_size}，不是整数倍。")
            factor = self.batch_size // target_batch_size
            # 创建索引，只选择每个 L 块的第一个
            indices = torch.arange(0, self.batch_size, factor, device=self.device)
            self.conv_state = self.conv_state.index_select(0, indices)
            self.ssm_state = self.ssm_state.index_select(0, indices)
            
        self.batch_size = target_batch_size

    def update(self, new_conv_input: Tensor, new_ssm_state: Tensor) -> None:
        """
        用新的输入和状态更新缓存，并自适应 batch size。
        """
        # 1. 自适应 batch size
        # 从输入张量推断目标 batch_size
        target_batch_size = new_conv_input.shape[0]
        self._adapt_batch_size(target_batch_size)
        
        # 2. 更新卷积状态
        self.conv_state = torch.roll(self.conv_state, shifts=-1, dims=-1)
        self.conv_state[..., -1] = new_conv_input
        
        # 3. 更新 SSM 状态
        self.ssm_state = new_ssm_state

    def get(self) -> tuple[Tensor, Tensor]:
        """获取当前的卷积状态和 SSM 状态。"""
        return self.conv_state, self.ssm_state

    @classmethod
    def alloc(cls, 
              batch_size: int, 
              config: 'Mamba2Config', 
              device: torch.device | None = None,
              dtype: torch.dtype | None = None) -> "Mamba2InferenceCache":
        """工厂方法：为推理开始时分配一个初始化的缓存。"""
        conv_dim = config.d_inner + 2 * config.d_state
        conv_state_shape = (batch_size, conv_dim, config.d_conv)
        ssm_state_shape = (batch_size, config.nheads, config.headdim, config.d_state)
        
        return cls(
            conv_state=torch.zeros(conv_state_shape, device=device, dtype=dtype),
            ssm_state=torch.zeros(ssm_state_shape, device=device, dtype=dtype),
            batch_size=batch_size,
            config=config,
            device=device,
            dtype=dtype
        )