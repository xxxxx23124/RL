"""
这个文件定义了单独一层mamba2怎么工作
"""
from core.ssd import ssd
from Mamba2Config import Mamba2Config
from InferenceCache import InferenceCache
from Norm_layer.RMSNorm import RMSNorm
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional

class Mamba2(nn.Module):
    """
    Mamba-2 层。这是模型的核心构建块，结合了卷积和状态空间模型 (SSM)。

    该模块实现了两种操作模式：
    1. 并行模式 (`forward`): 高效处理整个输入序列，适用于训练和长文本一次性推理。
    2. 循环/步进模式 (`step`): 一次处理一个时间步(token), 适用于自回归生成。
    """
    def __init__(self, args: Mamba2Config, device: Optional[torch.device] = None):
        super().__init__()
        self.args = args
        self.device = device

        # 输入投影层，将 d_model 映射到 SSM 所需的各个分量
        # 顺序: (z, x, B, C, dt)
        d_in_proj = 2 * args.d_inner + 2 * args.d_state + args.nheads
        self.in_proj = nn.Linear(args.d_model, d_in_proj, bias=False, device=device)

        # 1D 深度卷积层
        conv_dim = args.d_inner + 2 * args.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=args.d_conv,
            groups=conv_dim,
            padding=args.d_conv - 1,
            device=device,
        )

        # SSM 的可学习参数
        self.dt_bias = nn.Parameter(torch.empty(args.nheads, device=device))  # dt 的偏置
        self.A_log = nn.Parameter(torch.empty(args.nheads, device=device))   # A 的对数形式，保持其为负
        self.D = nn.Parameter(torch.empty(args.nheads, device=device))       # 残差连接的 D 参数

        # 归一化层和输出投影层
        self.norm = RMSNorm(args.d_inner, device=device)
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=False, device=device)

    def forward(self, u: Tensor, h: Optional[InferenceCache] = None) -> tuple[Tensor, InferenceCache]:
        """
        Mamba-2 层的前向传播。

        根据是否存在推理缓存 `h`，自动选择并行模式或循环模式。

        参数:
            u (Tensor): 输入张量，形状为 `(batch, seqlen, d_model)`。
                        在并行模式下，`seqlen` 应为 `chunk_size` 的倍数。
            h (InferenceCache | None): 推理缓存。如果提供，则进入单步循环模式。

        返回:
            tuple[Tensor, InferenceCache]:
                - y (Tensor): 输出张量，形状为 `(batch, seqlen, d_model)`。
                - h (InferenceCache): 更新后的推理缓存。
        """
        if h:
            # 循环模式（单步推理）
            return self._step(u, h)
        else:
            # 并行模式（训练或批量推理）
            return self._parallel_forward(u)

    def _parallel_forward(self, u: Tensor) -> tuple[Tensor, InferenceCache]:
        """并行模式下的前向传播，处理整个序列。"""
        # 1. 输入投影和分量计算
        z, xBC, dt = self._compute_zxbcdt(u)

        # 准备卷积输入
        conv_inputs = xBC.transpose(1, 2)  # (B, D_in, L)

        # 执行卷积并应用 SiLU 激活函数
        # conv1d自己就有pad功能，[:, :u.size(1), :]就是一般的因果卷积
        xBC = F.silu(self.conv1d(conv_inputs).transpose(1, 2)[:, :u.size(1), :])
        # 从输入中截取需要缓存的部分，即最后 d_conv-1 个时间步, 因为 self.conv1d.weight 的形状是 (D_in, 1, d_conv)，我们只需要 d_conv-1 的历史
        state_len = self.args.d_conv - 1

        cache_inputs = conv_inputs[..., -min(u.size(1), state_len):]
        # 填充到 d_conv 的总长度以匹配 InferenceCache 的定义, 推理时，缓存通常包含 d_conv-1 的历史和1个当前步，总长 d_conv
        conv_state = F.pad(cache_inputs, (self.args.d_conv - cache_inputs.shape[-1], 0))

        # 3. 并行 SSM 计算 (SSD)
        y, ssm_state = self._ssm_parallel(xBC, dt)

        # 4. 门控归一化和输出投影
        y = self.norm(y, z)
        y = self.out_proj(y)

        # 5. 创建并返回推理缓存
        h = InferenceCache(conv_state, ssm_state)
        return y, h

    def _step(self, u: Tensor, h: InferenceCache) -> tuple[Tensor, InferenceCache]:
        """循环模式下的前向传播，一次处理一个 token。"""
        assert u.size(1) == 1, "在步进模式下，每次只能处理一个 token"

        # 输入投影和分量计算, 保持 (B, 1, D) 形状
        z, xBC, dt = self._compute_zxbcdt(u)

        # 2. 卷积步骤 (循环方式)
        # 更新卷积状态缓存
        h.conv_state.copy_(torch.roll(h.conv_state, shifts=-1, dims=-1))
        h.conv_state[..., -1] = xBC.squeeze(1)
        # 计算卷积输出
        xBC = torch.sum(h.conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
        if self.conv1d.bias is not None:
            xBC += self.conv1d.bias
        # 将结果从 (B, conv_dim) 变回 (B, 1, conv_dim) 以匹配后续流程
        xBC = F.silu(xBC).unsqueeze(1)

        # 3. 循环 SSM 计算
        y, ssm_state = self._ssm_recurrent(xBC, dt, h.ssm_state)

        # 4. 更新 SSM 状态
        h.ssm_state.copy_(ssm_state)

        # 5. 门控归一化和输出投影
        y = self.norm(y, z)
        y = self.out_proj(y)

        return y, h

    def _compute_zxbcdt(self, u: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """从输入 `u` 计算 z, x, B, C, dt。"""
        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(
            zxbcdt,
            [
                self.args.d_inner,
                self.args.d_inner + 2 * self.args.d_state,
                self.args.nheads,
            ],
            dim=-1,
        )
        # 计算 dt (delta_t)，并应用 softplus 保证其为正
        dt = F.softplus(dt + self.dt_bias)
        return z, xBC, dt

    def _ssm_parallel(self, xBC: Tensor, dt: Tensor) -> tuple[Tensor, Tensor]:
        """执行并行化的 SSM 计算 (SSD)。"""
        A = -torch.exp(self.A_log)  # (nheads,)
        # 将 xBC 分解为 x, B, C
        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )
        # 重塑张量以匹配 SSD 的输入格式，每个头都用同一个B和C
        x = rearrange(x, "b l (h p) -> b l h p", p=self.args.headdim)
        B = rearrange(B, "b l n -> b l 1 n")
        C = rearrange(C, "b l n -> b l 1 n")

        # 使用 SSD 算法计算 SSM 输出
        y, final_ssm_state = ssd(
            x * dt.unsqueeze(-1),  # 乘以 dt 以缩放输入 x
            A * dt,               # 乘以 dt 以离散化 A
            B, C,
            self.args.chunk_size
        )

        # 添加 D 残差连接
        y = y + x * self.D.unsqueeze(-1)
        y = rearrange(y, "b l h p -> b l (h p)")
        return y, final_ssm_state

    def _ssm_recurrent(self, xBC: Tensor, dt: Tensor, ssm_state: Tensor) -> tuple[Tensor, Tensor]:
        """执行循环式的 SSM 计算，输入形状为 (B, 1, ...)。"""
        A = -torch.exp(self.A_log)  # (nheads,)
        xBC, dt = xBC.squeeze(1), dt.squeeze(1)
        # 分解 x, B, C
        x, B, C = torch.split(
            xBC, [self.args.d_inner, self.args.d_state, self.args.d_state], dim=-1
        )
        x = rearrange(x, "b (h p) -> b h p", p=self.args.headdim)

        # 计算离散化的 A 和 B
        dA = torch.exp(dt * A)  # (batch, nheads)
        # B/C: (B, N) -> (B, 1, N) 以便广播
        # dt: (B, H) -> (B, H, 1) 以便广播
        # dt_B = dt.unsqueeze(-1) * B.unsqueeze(1) -> (B, H, N)
        dt_B = dt.unsqueeze(-1) * B.unsqueeze(1) 
        # (b,h,n) * (b,h,p) -> (b,h,p,n)
        dBx = torch.einsum('bhn,bhp->bhpn', dt_B, x)

        # 更新 SSM 状态: s_t = dA * s_{t-1} + dB * x_t
        new_ssm_state = ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx

        # 计算输出: y_t = C * s_t + D * x_t
        y = torch.einsum("bhpn, bn -> bhp", new_ssm_state, C)
        y = y + rearrange(self.D, "h -> h 1") * x
        # 将结果从 (B, H, P) -> (B, H*P) -> (B, 1, H*P)
        y = rearrange(y, "b h p -> b (h p)").unsqueeze(1)
        return y, new_ssm_state