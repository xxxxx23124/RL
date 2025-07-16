from ANN.Layers.Mamba2_layer.ssd import ssd
from ANN.Layers.Mamba2_layer.Mamba2Config import Mamba2Config
from ANN.Layers.Mamba2_layer.InferenceCache import Mamba2InferenceCache
from ANN.Layers.Norm_layer.RMSNorm import RMSNorm
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange

class Mamba2_block(nn.Module):
    """
    Mamba-2 块。这是模型的核心构建块，结合了卷积和状态空间模型 (SSM)。

    该模块实现了两种操作模式：
    1. 并行模式 (`forward`): 高效处理整个输入序列，适用于训练和长文本一次性推理。
    2. 循环/步进模式 (`step`): 一次处理一个时间步(token), 适用于自回归生成。
    """
    def __init__(self, args: Mamba2Config, device: torch.device):
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
            padding=0, # 不使用内置 padding
            device=device,
        )

        self.pre_norm = RMSNorm(args.d_model, device)

        # SSM 的可学习参数
        self.dt_bias = nn.Parameter(torch.empty(args.nheads, device=device))  # dt 的偏置
        self.A_log = nn.Parameter(torch.empty(args.nheads, device=device))   # A 的对数形式，保持其为负
        self.D = nn.Parameter(torch.empty(args.nheads, device=device))       # 残差连接的 D 参数

        # 归一化层和输出投影层
        self.norm = RMSNorm(args.d_inner, device=device)
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=False, device=device)

    def forward(self, u: Tensor, h: Mamba2InferenceCache | None = None) -> tuple[Tensor, Mamba2InferenceCache]:
        B, S, L, D = u.shape
        original_shape = (B, S, L, D)
        u_reshaped = u.reshape(B * L, S, D)
        u_norm = self.pre_norm(u_reshaped)
        if h is None:
            # 如果h是None 就创建一个新的缓存
            h = Mamba2InferenceCache.alloc(
                batch_size=B * L,
                config=self.args,
                device=self.device
            )
        if S == 1:
            # 循环模式（单步推理）
            y, h = self._step(u_norm, h)
            return (u_reshaped + y).view(original_shape), h
        else:
            # 并行模式（训练或批量推理）
            # 检查一下x的序列长度，期望的是8 16 32 64 128 这种的，不期望 15这种的
            assert S % 2 == 0 and S // 2 > 1
            y, h = self._parallel_forward(u_norm, h)
            return (u_reshaped + y).view(original_shape), h

    def _parallel_forward(self, u: Tensor, h: Mamba2InferenceCache) -> tuple[Tensor, Mamba2InferenceCache]:
        """并行模式下的前向传播，处理整个序列。"""
        # 输入投影和分量计算
        z, xBC_unactivated, dt = self._compute_zxbcdt(u)

        current_conv_state, current_ssm_state = h.get()
        # 卷积步骤 (循环方式)
        # 创建一个克隆体 (clone) 来模拟更新，以避免修改原始缓存
        temp_conv_state = current_conv_state.clone()

        # 准备卷积输入
        conv_inputs = xBC_unactivated.transpose(1, 2)  # (B, D_in, L)

        # 执行卷积并应用 SiLU 激活函数
        conv_inputs = torch.cat([temp_conv_state[:,:,1:], # 取出后三个卷积，作为padding
                                 conv_inputs], dim=2)
        xBC = F.silu(self.conv1d(conv_inputs).transpose(1, 2))

        # 并行 SSM 计算 (SSD)
        y, new_ssm_state = self._ssm_parallel(xBC, dt, current_ssm_state)

        h.update_parallel(new_conv_input=xBC_unactivated.transpose(1, 2)[:, :, -4:], # (B, D_in, 4)
                          new_ssm_state=new_ssm_state)

        # 门控归一化和输出投影
        y = self.norm(y, z)
        y = self.out_proj(y)

        return y, h

    def _step(self, u: Tensor, h: Mamba2InferenceCache) -> tuple[Tensor, Mamba2InferenceCache]:
        """循环模式下的前向传播，一次处理一个 token。"""

        # 输入投影和分量计算, 保持 (B, 1, D) 形状
        z, xBC_unactivated, dt = self._compute_zxbcdt(u)

        # 从缓存中获取当前状态
        current_conv_state, current_ssm_state = h.get()

        # 卷积步骤 (循环方式)
        # 创建一个克隆体 (clone) 来模拟更新，以避免修改原始缓存
        temp_conv_state = current_conv_state.clone() 
        # 准备更新卷积状态：将当前步的输入和历史拼接
        # 注意：这里我们只 peek，真正的更新在 h.update() 中完成
        # 模拟滚动和更新以进行计算
        temp_conv_state = torch.roll(temp_conv_state, shifts=-1, dims=-1)
        temp_conv_state[..., -1] = xBC_unactivated.squeeze(1)

        # 计算卷积输出
        xBC = torch.einsum("bdw,dw->bd", temp_conv_state, self.conv1d.weight.squeeze(1))
        if self.conv1d.bias is not None:
            xBC += self.conv1d.bias
        # 将结果从 (B, conv_dim) 变回 (B, 1, conv_dim) 以匹配后续流程
        xBC_activated = F.silu(xBC).unsqueeze(1)

        # 循环 SSM 计算
        y, new_ssm_state = self._ssm_recurrent(xBC_activated, dt, current_ssm_state)

        # 更新 cache 状态
        h.update_step(new_conv_input=xBC_unactivated.squeeze(1),
                      new_ssm_state=new_ssm_state)

        # 门控归一化和输出投影
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

    def _ssm_parallel(self, xBC: Tensor, dt: Tensor, initial_states: Tensor | None=None) -> tuple[Tensor, Tensor]:
        """执行并行化的 SSM 计算 (SSD)"""
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
            B, 
            C,
            min(self.args.max_chunk_size, x.shape[1] // 2),
            initial_states
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