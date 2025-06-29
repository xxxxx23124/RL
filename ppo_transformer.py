import gymnasium as gym
from gymnasium import ObservationWrapper
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import time
import math
import itertools
from dataclasses import dataclass

class LunarLanderPOMDPWrapper(ObservationWrapper):
    def __init__(self, env, p_flicker):
        super().__init__(env)
        self.p_flicker = p_flicker
        self.flicker_indices = [0, 1, 2, 3, 4, 5]

    '''
    obs[0]: 水平位置 (x-coordinate)。以停机坪中心为原点 (0.0)。
    obs[1]: 垂直位置 (y-coordinate)。以停机坪表面为原点 (0.0)。
    obs[2]: 水平速度 (x-velocity)。向右为正。
    obs[3]: 垂直速度 (y-velocity)。向上为正。
    obs[4]: 角度 (angle)。单位是弧度。直立时为 0.0。
    obs[5]: 角速度 (angular velocity)。顺时针为正。
    obs[6]: 左腿是否接触地面。接触为 1.0，否则为 0.0。这是一个布尔值，用浮点数表示。
    obs[7]: 右腿是否接触地面。接触为 1.0，否则为 0.0。
    '''

    def observation(self, obs):
        return self._modify_obs(obs)

    def _modify_obs(self, obs):
        for i in self.flicker_indices:
            if np.random.rand() < self.p_flicker:
                obs[i] = 0.0
        return obs

@dataclass
class Config:
    # --- 环境与模型基本设置 (Environment & Model Basic Settings) ---
    gid: str = "LunarLanderContinuous-v2"  # 指定要使用的Gymnasium环境ID。
    render_mode:str = None # "human"
    save_interval_seconds: int = 3 * 3600  # 模型自动保存的时间间隔，单位为秒。这里设置为3小时。
    p_flicker: float = 0.1  # [如果关闭自动课程] 固定的观测值“闪烁”概率，即某个维度的观测值有多大几率被置为0。

    # --- 自动课程学习设置 (Automated Curriculum Learning Settings) ---
    use_automated_curriculum: bool = True  # 是否启用自动课程学习。如果为True，环境难度(p_flicker)会根据智能体表现自动调整。
    # 课程难度控制参数 (Curriculum Difficulty Control)
    initial_p_flicker_mean: float = 0.00  # 课程开始时，p_flicker采样的Beta分布的均值，代表初始难度。
    target_p_flicker_mean: float = 0.5  # 课程的最终目标难度，p_flicker采样的Beta分布均值的上限。
    # 难度提升条件 (Difficulty Progression Condition)
    curriculum_reward_threshold: float = -100.0  # 当最近N个轨迹的平均奖励超过此阈值时，就提升难度。
    curriculum_eval_episodes: int = 1600  # 用于评估是否提升难度的历史轨迹数量（即上面的'N'）。
    # 难度提升步长 (Difficulty Increment Step)
    p_flicker_mean_increment: float = 0.02  # 每次提升难度时，p_flicker均值的增加量。
    # Beta分布形状参数 (Beta Distribution Shape)
    beta_concentration: float = 5.0  # Beta分布的浓度参数(ν)。值越大，采样出的p_flicker值越集中在当前均值附近。

    # --- PPO 算法超参数 (PPO Hyperparameters) ---
    # 优势函数与回报计算 (Advantage & Return Calculation)
    use_Vtrace: bool = True  # 是否使用V-trace算法计算优势。优先级最高。
    use_GAE: bool = False  # 是否使用广义优势估计(GAE)。仅在use_Vtrace为False时生效。
    use_Reward_to_Go: bool = False  # 是否使用简单的Reward-to-Go。仅在use_Vtrace和use_GAE都为False时生效。
    gamma: float = 0.99  # 折扣因子，用于计算未来奖励的当前价值。
    lam: float = 0.99  # GAE的lambda参数，用于平衡偏差和方差。
    # 优化器与损失函数 (Optimizer & Loss Function)
    lr: float = 1e-6  # 学习率，用于NAdam优化器。
    clip: float = 0.05  # PPO裁剪范围(ε)，限制策略更新的幅度，防止更新过大。
    rho_clip: float = 1.0  # V-trace中的rho (重要性采样比率)的裁剪上限。
    c_clip: float = 1.0  # V-trace中的c (重要性采样比率)的裁剪上限。
    ent_coef: float = 1e-4  # 熵损失的系数，鼓励策略探索。
    vf_coef: float = 0.5  # 值函数(Critic)损失的系数。
    max_grad_norm: float = 0.5  # 梯度裁剪的最大范数，防止梯度爆炸。
    target_kl: float = 0.01  # 目标KL散度。如果更新后的策略与旧策略的KL散度超过此值，则提前停止更新，以保证稳定性。
    # Actor模型输出分布 (Actor's Output Distribution)
    cov_var_value: float = 0.5  # Actor输出的多元高斯分布的协方差矩阵对角线上的值（方差）。

    # --- 训练流程控制 (Training Process Control) ---
    n_updates_per_iteration: int = 3  # 每次收集到足够数据后，对这些数据进行学习的次数（epoch数）。
    max_cpu_threads: int = 0  # 用于数据采样的CPU并行环境数量。0代表不使用额外的CPU线程。
    max_gpu_threads: int = 0  # 用于数据采样的GPU并行环境数量。0代表不使用额外的GPU线程。
    max_timesteps_per_episode: int = 1200  # 每个轨迹(episode)的最大时间步数。这也是Transformer处理的最大序列长度。
    timesteps_per_batch: int = 16000  # 每个并行环境(worker)一次收集的数据量（时间步数）。
    timesteps_per_mini_batch: int = 1  # 在进行模型更新时，mini_batch时间步的大小，此处设置为1代表我希望只处理1个时间步，但实际上为只处理一个轨迹
    timesteps_all_batch: int = (max_cpu_threads + max_gpu_threads) * timesteps_per_batch  # 所有并行环境收集的总数据量，达到这个量后触发一次模型更新。

    # --- Transformer 模型架构参数 (Transformer Architecture Parameters) ---
    # 核心维度 (Core Dimensions)
    encoder_d_model: int = 512  # Transformer模型内部的主要工作维度 (embedding size)。
    encoder_attention_heads: int = 8  # 多头注意力机制中的头数。
    # 前馈网络 (Feed-Forward Network)
    encoder_d_ff: int = encoder_d_model * 3  # 前馈网络(FFN)的中间层维度。
    # 层数 (Number of Layers)
    backbone_layers: int = 6  # 主干网络中的Transformer Encoder层数。
    actor_layers: int = 3  # Actor头中的Transformer Encoder层数。实际上在模型中还有一个actor_head的模块，所有实际大小为actor_layers+1
    critic_layers: int = 2  # Critic头中的Transformer Encoder层数。实际大小与actor_layers同理

    def __post_init__(self):
        """
        在Config对象创建后自动执行的初始化函数。
        主要用于根据其他参数计算并设置一些衍生属性。
        """
        env = gym.make(self.gid)
        self.obs_dim = env.observation_space.shape[0]  # 从环境中自动获取观测空间的维度。
        self.act_dim = env.action_space.shape[0]  # 从环境中自动获取动作空间的维度。

        # 如果没有设置并行环境，则将总批次大小设置为单个环境的批次大小。
        if self.timesteps_all_batch == 0:
            self.timesteps_all_batch = self.timesteps_per_batch

        # 断言检查，确保模型参数设置合理。
        assert (self.encoder_d_model % self.encoder_attention_heads == 0), "d_model 必须能被 num_heads 整除"
        # 断言检查，确保至少选择了一种优势计算方法。
        assert (self.use_Reward_to_Go or self.use_GAE or self.use_Vtrace), "必须启用至少一种回报计算方法 (RTG, GAE, or V-trace)"

        # 确保三种优势计算方法之间没有逻辑冲突，并设置优先级。
        if self.use_Vtrace:
            self.use_GAE = False
            self.use_Reward_to_Go = False
        elif self.use_GAE:
            self.use_Reward_to_Go = False


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len, base=10000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float) / self.dim))
        t = torch.arange(self.max_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("freqs_cos", emb.cos()[None, None, :, :])
        self.register_buffer("freqs_sin", emb.sin()[None, None, :, :])

    def rotate_half(self, x):
        # x shape: (..., dim)
        x1, x2 = x.chunk(2, dim=-1)  # x1, x2 shape: (..., dim / 2)
        return torch.cat((-x2, x1), dim=-1)  # shape: (..., dim)

    def forward(self, x: torch.Tensor, seq_len: int, seq_len_offset: int = 0):
        # x shape: (batch, n_heads, seq_len, d_k)
        cos = self.freqs_cos[:, :, seq_len_offset: seq_len_offset + seq_len, :]  # shape: (1, 1, seq_len, d_k)
        sin = self.freqs_sin[:, :, seq_len_offset: seq_len_offset + seq_len, :]  # shape: (1, 1, seq_len, d_k)
        # RoPE 旋转的数学等价实现
        # (x * cos) + (rotate_half(x) * sin)
        # 这等价于复数乘法 (x_r + i*x_i) * (cos + i*sin) 的实部和虚部
        rotated_x = (x * cos) + (self.rotate_half(x) * sin)
        return rotated_x.type_as(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)

        self.linear_out = nn.Linear(d_model, d_model)

        self.kv_cache = None

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.normal_(module.bias, mean=0.0, std=0.2)

    def attention(self, query, key, value, mask: torch.Tensor):
        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # 确保 mask 维度匹配 (batch, heads, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value)

    def forward(self, query, key, value, mask, rotary_emb: RotaryEmbedding, use_cache=False):
        batch_size = query.size(0)

        # use_cache 模式下，输入序列长度为 1
        # 否则，为输入的实际序列长度
        seq_len = query.size(1)

        query = self.proj_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        key = self.proj_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        value = self.proj_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 2. 获取缓存的长度
        past_seq_len = 0
        if self.kv_cache is not None:
            # kv_cache[0] is past_key
            past_seq_len = self.kv_cache[0].size(2)
        # 先旋转
        query = rotary_emb(query, seq_len=seq_len, seq_len_offset=past_seq_len)
        key = rotary_emb(key, seq_len=seq_len, seq_len_offset=past_seq_len)
        if use_cache:
            if self.kv_cache is not None:
                past_key, past_value = self.kv_cache
                # 再拼接
                key = torch.cat([past_key, key], dim=2)
                value = torch.cat([past_value, value], dim=2)
                # 将当前的 k, v 保存为缓存，用于下一个时间步
            self.kv_cache = (key, value)
            # Mask 只需要在训练或处理长序列时应用，生成时由于 q_len=1，其实不需要 mask
            mask = None
        else:
            # 将kv_cache清空为训练和下一次收集做准备
            self.kv_cache = None
        # (bs, n_heads, seq_len, d_k)
        x = self.attention(query,key,value,mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.d_k)
        return self.linear_out(x)


class LRNN_StatefulFFN(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_expand: int):
        super(LRNN_StatefulFFN, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_expand = d_expand

        # --- 1. 外部的门控和投影层 (类似Mamba的设计) ---
        self.in_proj = nn.Linear(d_model, 2 * d_expand)
        self.out_proj = nn.Linear(d_expand, d_model)

        # --- 2. 核心状态空间模型 (SSM) 参数的投影层 ---
        # A: 输入无关 (Input-Independent)
        # A 是一个对角复数矩阵，我们存储它的 log。
        # Re(A) 必须为负数以保证稳定性。
        # 形状: (d_expand, d_state)
        A_log_re = torch.log(torch.rand(d_expand, d_state) * 0.5 + 0.5) # 初始化为 U[log(0.5), log(1.0)] -> Re(A) in [-0.693, 0]
        A_log_im = torch.zeros(d_expand, d_state)
        self.A_log = nn.Parameter(torch.complex(A_log_re, A_log_im))

        # 为 B, delta 创建一个统一的投影层
        # B C 为复数
        self.ssm_params_proj = nn.Linear(d_expand, 4 * d_state + 1)

        # D: 残差/跳跃连接
        self.D = nn.Parameter(torch.randn(d_expand))

        self.h_prev = None

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        对模块权重进行特定的数学初始化。
        此函数将通过 `self.apply(self._init_weights)` 应用到每个子模块。
        """
        if isinstance(module, nn.Linear):
            # 对大多数线性层使用标准初始化
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.normal_(module.bias, mean=0.0, std=0.1)

        # --- 对SSM相关参数的投影层进行特殊处理 ---
        if module is self.ssm_params_proj:
            # module.weight 的形状: (d_state * 4 + 1, d_model)
            # module.bias 的形状: (d_state * 4 + 1,)
            with torch.no_grad():
                # 初始化 delta (Δ) 的偏置
                # delta = softplus(proj_out)
                # proj_out = x @ W + b
                # 我们希望初始的 delta 在一个特定范围，例如 ~0.1。
                # softplus(x) ≈ x if x > 20, softplus(x) ≈ exp(x) if x < -20
                # softplus(inv(target)) -> target
                # inv_softplus(y) = log(exp(y) - 1)
                # 设目标 delta_init = 0.1, 则偏置应初始化为 log(exp(0.1) - 1) ≈ -2.25
                target_delta = 0.1
                delta_bias_init = math.log(math.exp(target_delta) - 1)
                torch.nn.init.constant_(module.bias[self.d_state * 4:], delta_bias_init)

                # 4. 初始化 delta (Δ) 的投影权重
                # 我们希望 delta 的初始值主要由偏置决定，而不是输入的 x。
                # 因此，我们将连接到 delta 的投影权重初始化为非常小的值。
                torch.nn.init.constant_(module.weight[self.d_state * 4:, :], 1e-6)

    def discretize(self, A_log: torch.Tensor, delta: torch.Tensor):
        """
        A_log shape: (B, L, d_state) - cfloat
        delta shape: (B, L, 1) - float
        """
        # 让模型自己去学，A_c = A_log
        A = A_log
        delta = delta.unsqueeze(-1)  # -> (B, L, d_expand, 1)
        A = A.unsqueeze(0).unsqueeze(0)  # -> (1, 1, d_expand, d_state)
        delta = delta.to(A.dtype)  # 转换为复数类型，便于计算

        # A_bar
        dA = delta * A
        A_bar = torch.exp(dA)

        # B_bar, 使用更稳健的计算
        # 使用阈值切换公式以保证数值稳定
        # 当 dA 绝对值很小时，使用泰勒展开的一阶近似：(e^x - 1)/x ≈ 1
        # (e^dA - 1)/A = ((e^dA - 1)/dA) * delta
        # 所以当 dA -> 0 时，B_bar -> delta
        abs_dA = torch.abs(dA)
        small_dA_mask = abs_dA < 1e-3  # 阈值可以调整

        # 对 dA != 0 的情况
        epsilon = torch.finfo(dA.dtype).eps
        B_bar_large = delta * (A_bar - 1) / (dA + epsilon)

        # 对 dA ≈ 0 的情况，(A_bar - 1) / dA 趋近于 1，所以 B_bar 趋近于 delta
        # 为了避免计算，直接用 delta
        # B_bar_small = delta.expand_as(B_bar_large)

        # 可以用泰勒展开的更高阶来提高精度
        # B_bar ≈ delta * (1 + dA/2 + dA^2/6 + ...)
        # B_bar_small = delta * (1 + dA / 2) # 二阶近似
        B_bar_small = delta * (1 + dA / 2)

        B_bar = torch.where(small_dA_mask, B_bar_small, B_bar_large)

        return A_bar, B_bar

    def forward(self, x: torch.Tensor, use_h_prev:bool = False):
        """
        x 的形状: (batch_size, seq_len, d_model)
        use_h_prev (可选): 使用上一个序列块的最终隐藏状态, 形状 (batch_size, d_state)
                       这对于处理长序列或在RL中进行单步推理非常有用。当use_h_prev为真是x的形状为(B, 1, d_model)
        返回: 输出张量
        """
        batch_size, seq_len, _ = x.shape

        # --- 外部处理：投影和门控 ---
        gate, x_ssm = self.in_proj(x).chunk(2, dim=-1) # gate, x_ssm: (B, L, d_expand)
        gate = F.silu(gate)

        # --- 核心 SSM 循环 ---

        # 动态生成 SSM 复数 A, B 和 实数delta
        ssm_params = self.ssm_params_proj(x_ssm)
        B_re, B_im, C_re, C_im, delta = ssm_params.split(
            [self.d_state, self.d_state, self.d_state, self.d_state, 1], dim=-1
        )
        B = torch.complex(B_re, B_im) # (B, L, d_state)
        C = torch.complex(C_re, C_im) # (B, L, d_state)

        delta = F.softplus(delta)  # (B, L, 1)

        # --- 3. 离散化 A 和 B ---
        # A_log: (d_expand, d_state), delta: (B, L, d_expand)
        A_bar, B_bar = self.discretize(self.A_log, delta)
        # A_bar: (B, L, d_expand, d_state)
        # B_bar: (B, L, d_expand, d_state)

        # 循环更新状态
        if use_h_prev and self.h_prev is not None:
            h = self.h_prev.to(dtype=torch.cfloat)  # 确保状态是复数 (B, d_expand, d_state)
        else:
            h = torch.zeros(batch_size, self.d_state, device=x.device, dtype=torch.cfloat)

        y_outputs = []

        for t in range(seq_len):
            A_bar_t = A_bar[:, t]       # (B, d_expand, d_state)
            B_bar_t = B_bar[:, t]       # (B, d_expand, d_state)
            B_t = B[:, t]               # (B, d_state)
            C_t = C[:, t]               # (B, d_state)
            delta_t = delta[:, t]       # (B, d_expand)
            x_ssm_t = x_ssm[:, t]       # (B, d_expand)

            # Mamba/S4D 更新公式: h_t = A_bar_t * h_{t-1} + B_bar_t * u_t
            # S4D: u_t = x_t
            # Mamba: u_t = B_t * x_t，但这里的B是输入依赖的，所以更像 u_t = B(x_t)*x_t
            # 简化且有效的实现: h_t = A_bar_t * h_{t-1} + (B_bar_t * delta_t * B_t) * x_ssm_t
            # 我们遵循一个更标准的S4D形式，u_t = x_ssm_t
            # 维度调整： B_bar_t(B,d_exp,d_st) * x_ssm_t(B,d_exp) -> (B,d_exp,d_st)
            h = A_bar_t * h + B_bar_t * x_ssm_t.unsqueeze(-1)

            # 输出方程: y_t = C_t * h_t
            # h: (B, d_expand, d_state), C_t: (B, d_state) -> (B, d_expand)
            # 需要对 C_t 求共轭
            y_t = torch.einsum('bds,bds->bd', h, C_t.unsqueeze(1).conj())
            y_outputs.append(torch.real(y_t))

        y = torch.stack(y_outputs, dim=1)  # (B, L, d_expand)

        # --- 5. 添加残差连接 D 和最终门控/投影 ---
        y = y + x_ssm * self.D
        output = self.out_proj(y * gate)

        if use_h_prev:
            self.h_prev = h
        else:
            self.h_prev = None

        return output


class EncoderLayer(nn.Module):
    def __init__(self, args: Config):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(args.encoder_d_model, args.encoder_attention_heads)
        self.lrnn_ffd = LRNN_StatefulFFN(args.encoder_d_model, args.encoder_d_model, args.encoder_d_ff)
        self.norm1 = nn.LayerNorm(args.encoder_d_model)
        self.norm2 = nn.LayerNorm(args.encoder_d_model)

    def forward(self, x, mask, rotary_emb: RotaryEmbedding, data_collection:bool):
        x2 = self.norm1(x)
        x = x + self.self_attn(x2, x2, x2, mask, rotary_emb, data_collection)
        x2 = self.norm2(x)
        x = x + self.lrnn_ffd(x2, data_collection)
        return x


class Encoder(nn.Module):
    def __init__(self, args:Config):
        super(Encoder, self).__init__()
        self.d_model = args.encoder_d_model
        self.embedding = nn.Linear(args.obs_dim, args.encoder_d_model)
        self.rotary_emb = RotaryEmbedding(
            dim=args.encoder_d_model // args.encoder_attention_heads,
            max_len=args.max_timesteps_per_episode,
        )
        self.backbone = nn.ModuleList([EncoderLayer(args) for _ in range(args.backbone_layers)])
        self.backbone_norm = nn.LayerNorm(args.encoder_d_model)
        self.actor = nn.ModuleList([EncoderLayer(args) for _ in range(args.actor_layers)])
        self.critic = nn.ModuleList([EncoderLayer(args) for _ in range(args.critic_layers)])
        self.actor_norm = nn.LayerNorm(args.encoder_d_model)
        self.actor_end = nn.Linear(args.encoder_d_model, args.act_dim)
        self.critic_norm = nn.LayerNorm(args.encoder_d_model)
        self.critic_end = nn.Linear(args.encoder_d_model, 1)

    def forward(self, src, mask, data_collection:bool):
        src = self.embedding(src) * math.sqrt(self.d_model)
        for layer in self.backbone:
            src = layer(src, mask, self.rotary_emb, data_collection)
        src = self.backbone_norm(src)
        actor = src
        critic = src
        for layer in self.actor:
            actor = layer(actor, mask, self.rotary_emb, data_collection)
        actor = self.actor_norm(actor)
        actor = self.actor_end(actor)
        for layer in self.critic:
            critic = layer(critic, mask, self.rotary_emb, data_collection)
        critic = self.critic_norm(critic)
        critic = self.critic_end(critic)
        return actor, critic


class Transformer(nn.Module):
    def __init__(self, args:Config = Config()):
        super(Transformer, self).__init__()
        self.args = args
        self.encoder = Encoder(self.args)

    def clear_kvCashe_hPrev(self):
        for module in self.modules():
            if isinstance(module, MultiHeadAttention):
                module.kv_cache = None
            elif isinstance(module, LRNN_StatefulFFN):
                module.h_prev = None

    def forward(self, src, mask, data_collection:bool=False):
        actor, critic = self.encoder(src, mask, data_collection)
        return actor, critic


class Environment:
    def __init__(self, id, device, render_mode=None, requires_grad=False, args:Config = Config()):
        self.ppo_instance = None
        self.args = args
        self.id = str(id) + " " + device
        self.device = torch.device(device)
        self.timesteps_per_batch = args.timesteps_per_batch
        self.max_timesteps_per_episode = args.max_timesteps_per_episode
        self.env = gym.make(self.args.gid, render_mode = render_mode) #"human" None
        self.env = LunarLanderPOMDPWrapper(self.env, p_flicker=args.p_flicker)
        self.observation, self.info = self.env.reset()
        self.reward = None
        self.terminated = False
        self.truncated = False
        self.cov_var = torch.full(size=(self.args.act_dim,), fill_value=args.cov_var_value)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)
        self.actor_critic = Transformer().to(self.device)
        self.requires_grad = requires_grad
        for param in self.actor_critic.parameters():
            param.requires_grad = self.requires_grad
        self._init_observation_data()

    def rollout(self):
        self._init_observation_data()
        t = 0
        print(f"环境id：{self.id}开始收集数据")
        while t < self.timesteps_per_batch:
            if self.args.use_automated_curriculum:
                # 从 PPO 主对象获取当前的均值和浓度
                mean = self.ppo_instance.current_p_flicker_mean
                concentration = self.args.beta_concentration
                # Beta分布的参数 a 和 b 可以通过均值(mu)和浓度(nu)计算得出：
                # a = mu * nu
                # b = (1 - mu) * nu
                # 为防止 a或b 为0导致采样出问题，加一个小的epsilon
                alpha = max(mean * concentration, 1e-6)
                beta = max((1.0 - mean) * concentration, 1e-6)
                # 从Beta分布中采样一个p_flicker值
                self.env.p_flicker = min(np.random.beta(alpha, beta), self.args.target_p_flicker_mean)
            else:
                # 如果不使用课程，保持原有逻辑
                self.env.p_flicker = self.args.p_flicker
            ep_obs = []
            ep_acts = []
            ep_log_probs = []
            ep_rews = []
            ep_vals = []
            ep_dones = []
            self.observation, self.info = self.env.reset()
            ep_t = 0
            self.terminated = False
            self.truncated = False
            ep_start_time = time()
            # 把隐藏状态清空
            self.actor_critic.clear_kvCashe_hPrev()
            while ep_t < self.max_timesteps_per_episode and not (self.terminated or self.truncated):
                ep_t += 1
                t += 1
                self.observation = torch.tensor(self.observation, dtype=torch.float).to(self.device)
                ep_obs.append(self.observation.unsqueeze(0))
                action, log_prob, V = self._get_action(self.observation)
                self.observation, self.reward, self.terminated, self.truncated, self.info = self.env.step(action.cpu().numpy())
                ep_acts.append(action.unsqueeze(0))
                ep_log_probs.append(log_prob.unsqueeze(0))
                ep_rews.append(self.reward)
                ep_vals.append(V.item())
                ep_dones.append(self.terminated or self.truncated)

            # 把隐藏状态清空
            self.actor_critic.clear_kvCashe_hPrev()

            self.batch_ep_obs.append(torch.cat(ep_obs, dim=0))
            self.batch_ep_acts.append(torch.cat(ep_acts, dim=0))
            self.batch_ep_log_probs.append(torch.cat(ep_log_probs, dim=0))
            self.batch_ep_rews.append(ep_rews)
            self.batch_lens.append(ep_t)
            self.batch_ep_vals.append(ep_vals)
            self.batch_ep_dones.append(ep_dones)
            self.ep_time_consume.append(time() - ep_start_time)
        print(f"环境id：{self.id}数据收集完毕")
        self._report_info()
        return self.batch_ep_obs, self.batch_ep_acts, self.batch_ep_log_probs, self.batch_ep_rews, self.batch_lens, self.batch_ep_vals, self.batch_ep_dones

    def _get_action(self, observation):
        with torch.no_grad():
            causal_mask = None
            observation = observation.unsqueeze(0).unsqueeze(0) # 1 1 obs_dim
            mean, V = self.actor_critic(observation, mask = causal_mask, data_collection=True)
            dist = MultivariateNormal(mean[0][-1], self.cov_mat)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob, V[0][-1]

    def _report_info(self):
        num_episodes = len(self.batch_ep_rews)
        if num_episodes == 0:
            print(f"--- 环境 {self.id} 报告 (无数据) ---")
            return

        total_timesteps = np.sum(self.batch_lens)
        total_duration = np.sum(self.ep_time_consume)

        # 奖励相关
        ep_rewards = [np.sum(ep_rews) for ep_rews in self.batch_ep_rews]
        avg_reward = np.mean(ep_rewards)
        std_reward = np.std(ep_rewards)
        max_reward = np.max(ep_rewards)
        min_reward = np.min(ep_rewards)

        # 时间步长相关
        avg_len = np.mean(self.batch_lens)
        max_len = np.max(self.batch_lens)
        min_len = np.min(self.batch_lens)

        # 性能/效率相关
        steps_per_second = total_timesteps / total_duration if total_duration > 0 else 0

        # 特定于 LunarLander 的统计信息
        # 假设成功着陆的奖励 > 200, 坠毁的奖励 < -100
        # 这只是一个启发式规则，你可以根据需要调整
        landings = sum(1 for r in ep_rewards if r > 200)
        crashes = sum(1 for r in ep_rewards if r < -100)
        landing_rate = (landings / num_episodes) * 100 if num_episodes > 0 else 0

        # 使用分隔符和标题创建清晰的报告块
        print("\n" + f"--- 环境 {self.id} 采样报告 ---".center(60, "="))

        # 总体概览
        print(f"| 采样概览:")
        print(f"|   - 收集轨迹数量: {num_episodes} 条")
        print(f"|   - 总时间步数:   {total_timesteps} 步")
        print(f"|   - 总耗时:         {total_duration:.2f} 秒")
        print(f"|   - 采样速度:       {steps_per_second:.1f} 步/秒")
        print("-" * 60)
        # 奖励统计
        print(f"| 轨迹奖励 (Reward) 统计:")
        print(f"|   - 平均值 (Mean):  {avg_reward:>8.2f}  |  标准差 (Std): {std_reward:>8.2f}")
        print(f"|   - 最大值 (Max):   {max_reward:>8.2f}  |  最小值 (Min): {min_reward:>8.2f}")
        print("-" * 60)
        # 轨迹长度统计
        print(f"| 轨迹长度 (Episode Length) 统计:")
        print(f"|   - 平均值 (Mean):  {avg_len:>8.1f}  |  最大/小值: {max_len}/{min_len}")
        print("-" * 60)
        # 特定于环境的指标
        print(f"| LunarLander 指标:")
        print(f"|   - 成功着陆率:     {landing_rate:.1f}% ({landings}/{num_episodes})")
        print(f"|   - 坠毁数量:       {crashes} 次")
        print("=" * 60 + "\n")

    def _init_observation_data(self):
        self.batch_ep_obs = []
        self.batch_ep_acts = []
        self.batch_ep_log_probs = []
        self.batch_ep_rews = []
        self.batch_lens = []
        self.batch_ep_vals = []
        self.batch_ep_dones = []

        self.ep_time_consume = []

    def _set_agent(self, agent):
        for p1, p2 in zip(self.actor_critic.parameters(), agent.parameters()):
            p1.data = p2.data.clone().to(self.device)
        print(f"环境id：{self.id}：  模型已更新")


class PPO(Environment):
    def __init__(self, args: Config = Config()):
        self.args = args

        # 这会创建 self.env, self.actor_critic 模型等
        super(PPO, self).__init__(0, "cuda", requires_grad=True, args=self.args, render_mode=args.render_mode)

        if self.args.use_automated_curriculum:
            self.current_p_flicker_mean = self.args.initial_p_flicker_mean
            self.recent_episode_rewards = []
        # else 分支可以省略，因为固定 p_flicker 的情况由 Environment 自身处理

        self.ppo_instance = self

        self.actor_critic_optim = torch.optim.NAdam(self.actor_critic.parameters(), lr=self.args.lr)

        self._init_store_data()
        self._init_compute_data()
        self._init_backup_data()

        self.sub_envs = []
        for i in range(self.args.max_cpu_threads):
            # 确保传递 args
            sub_env = Environment(i + 1, "cpu", args=self.args)
            sub_env.ppo_instance = self
            self.sub_envs.append(sub_env)
        for i in range(self.args.max_gpu_threads):
            # 确保传递 args
            sub_env = Environment(i + 1, "cuda", args=self.args)
            sub_env.ppo_instance = self
            self.sub_envs.append(sub_env)

        self._load()

        if 'last_save_time' not in self.__dict__ or self.last_save_time is None:
            self.last_save_time = time()

        if 'i' not in self.__dict__ or self.i is None:
            self.i = 0

        current_time = time()
        print(f"首次保存将在大约 {(self.args.save_interval_seconds - (current_time - self.last_save_time)) / 3600:.1f} 小时后。")

    def learn(self, simulation_number):
        if len(self.sub_envs):
            print("多线程训练")
            for sub_env in self.sub_envs:
                sub_env._set_agent(self.actor_critic)
            self._multithread_learn(simulation_number)
        else:
            print("单线程训练")
            self._singlethread_learn(simulation_number)

    def _singlethread_learn(self, simulation_number):
        while self.i < simulation_number:
            self._optimizer_to_cpu()  # 在数据收集前移动到CPU
            records = self.rollout()
            cur_timesteps = np.sum(self.batch_lens)
            self._load_data_to_store_data(records)
            print(f"距离模型迭代{cur_timesteps}/{self.args.timesteps_all_batch}组数据")
            if cur_timesteps >= self.args.timesteps_all_batch:
                self.i += 1
                all_ep_rewards = self._collect_rewards_from_rollouts()
                self._optimizer_to_gpu()  # 在模型训练前移回GPU
                self._compute_loss() # 此时优化器和模型都在GPU
                if self.args.use_automated_curriculum:
                    self._update_curriculum(all_ep_rewards)
                self._save()

    def _multithread_learn(self, simulation_number):
        with (ThreadPoolExecutor(max_workers=len(self.sub_envs)) as thread_pool_executor):
            futures = set()
            future_to_env = {}
            cur_timesteps = 0
            self._optimizer_to_cpu()
            for sub_env in self.sub_envs:
                future = thread_pool_executor.submit(sub_env.rollout)
                futures.add(future)
                future_to_env[future] = sub_env
            while self.i < simulation_number:
                for future in as_completed(futures):
                    futures.remove(future)
                    sub_env = future_to_env.pop(future)
                    cur_timesteps += np.sum(sub_env.batch_lens)
                    records = future.result()
                    self._load_data_to_store_data(records)
                    print(f"距离模型迭代{cur_timesteps}/{self.args.timesteps_all_batch}组数据")
                if cur_timesteps >= self.args.timesteps_all_batch:
                    self.i += 1
                    all_ep_rewards = self._collect_rewards_from_rollouts()
                    self._optimizer_to_gpu()  # 在模型训练前移回GPU
                    self._compute_loss()
                    if self.args.use_automated_curriculum:
                        self._update_curriculum(all_ep_rewards)
                    self._save()
                    cur_timesteps = 0
                    self._optimizer_to_cpu()
                    for sub_env in self.sub_envs:
                        sub_env._set_agent(self.actor_critic)
                        new_future = thread_pool_executor.submit(sub_env.rollout)
                        futures.add(new_future)
                        future_to_env[new_future] = sub_env

    def _compute_loss(self):
        total_timesteps_in_batch = np.sum(self.all_batch_lens)
        num_episodes_in_batch = len(self.all_batch_lens)
        print("\n" + "=" * 80)
        print(
            f"| 开始处理第 {self.i} 轮更新... (共 {num_episodes_in_batch} 个轨迹, {total_timesteps_in_batch} 个时间步) |")
        print("=" * 80)

        print("\n--- [阶段 1/2] 计算优势函数和目标值 ---")
        mini_batch_id = 0
        A_k_list = []
        mini_batch_rtgs_list = []

        # 临时存储一些统计数据
        all_V_in_batch = []
        all_rewards_in_batch = [reward for ep_rewards in self.all_batch_rews for reward in ep_rewards]

        while len(self.all_batch_lens):
            mini_batch_id += 1
            self._init_compute_data()
            # 注意：这里的 prepare_compute_data 不应该备份，因为它只用于计算优势
            # 我们需要一个临时的 `prepare_compute_data` 版本或确保它不备份
            self._prepare_compute_data()  # backup = True

            print(
                f"  [优势计算] Mini-batch #{mini_batch_id}: {len(self.mini_batch_lens)} 个轨迹, {np.sum(self.mini_batch_lens)} 个时间步")

            V, curr_log_probs, entropy = self._evaluate()

            # 记录价值函数统计
            all_V_in_batch.extend(V.detach().cpu().numpy())

            if self.args.use_Vtrace:
                A_k = self._calculate_vtrace(curr_log_probs)
                mini_batch_rtgs = A_k + V.detach()
            elif self.args.use_GAE:
                A_k = self._calculate_gae()
                mini_batch_rtgs = A_k + V.detach()
            else:
                mini_batch_rtgs = self._compute_rtgs()
                A_k = mini_batch_rtgs - V.detach()

            # 优势标准化
            A_k_unnormalized_mean, A_k_unnormalized_std = A_k.mean(), A_k.std()
            A_k = (A_k - A_k_unnormalized_mean) / (A_k_unnormalized_std + 1e-10)

            print(f"    - 优势 (GAE/V-trace) [未标准化]: mean={A_k_unnormalized_mean:.4f}, std={A_k_unnormalized_std:.4f}")
            print(f"    - 目标值 (RTG): mean={mini_batch_rtgs.mean():.4f}, std={mini_batch_rtgs.std():.4f}")

            mini_batch_rtgs_list.append(mini_batch_rtgs)
            A_k_list.append(A_k)

        # 打印本轮数据的整体统计
        print("\n  [批次数据统计]")
        print(f"    - 原始奖励 (Rewards): mean={np.mean(all_rewards_in_batch):.4f}, std={np.std(all_rewards_in_batch):.4f}")
        print(f"    - 价值函数 (Value): mean={np.mean(all_V_in_batch):.4f}, std={np.std(all_V_in_batch):.4f}")

        self._load_backup_data_to_store_data()  # 恢复所有数据以进行多轮更新

        print("\n--- [阶段 2/2] 模型参数更新 ---")
        update_model = True
        for j in range(self.args.n_updates_per_iteration):
            print(f"\n  [更新迭代 {j + 1}/{self.args.n_updates_per_iteration}]")
            self.actor_critic_optim.zero_grad()

            # 存储每一轮的损失
            total_actor_loss, total_critic_loss, total_entropy_loss = 0, 0, 0
            processed_timesteps = 0
            mini_batch_idx = -1

            while len(self.all_batch_lens):
                mini_batch_idx += 1
                self._init_compute_data()
                # 最后一个epoch不需要备份，因为数据将被丢弃
                is_last_update_iter = (j == self.args.n_updates_per_iteration - 1)
                self._prepare_compute_data(backup=not is_last_update_iter)

                V, curr_log_probs, entropy = self._evaluate()

                # 计算损失
                entropy_loss = entropy.mean()
                logratios = curr_log_probs - self.mini_batch_log_probs
                ratios = torch.exp(logratios)

                approx_kl = ((ratios - 1) - logratios).mean()

                # 早停机制
                if approx_kl > self.args.target_kl:  # 使用 target_kl 作为硬停止阈值
                    print(f"    ! [早停] Mini-batch #{mini_batch_idx + 1}: KL散度 ({approx_kl.item():.4f}) 显著超过目标 ({self.args.target_kl}). 停止本轮所有更新。")
                    update_model = False
                    break  # 跳出 mini-batch 循环

                surr1 = ratios * A_k_list[mini_batch_idx]
                surr2 = torch.clamp(ratios, 1 - self.args.clip, 1 + self.args.clip) * A_k_list[mini_batch_idx]

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = self.args.vf_coef * nn.MSELoss()(V, mini_batch_rtgs_list[mini_batch_idx])

                # 总损失
                loss = actor_loss + critic_loss - self.args.ent_coef * entropy_loss
                loss.backward()

                # 打印当前 mini-batch 的详细信息
                num_timesteps_in_mini_batch = np.sum(self.mini_batch_lens)
                print(f"    - Mini-batch #{mini_batch_idx + 1} ({num_timesteps_in_mini_batch}步): "
                      f"KL={approx_kl.item():.4f}, "
                      f"PolicyLoss={actor_loss.item():.4f}, "
                      f"ValueLoss={critic_loss.item():.4f}, "
                      f"Entropy={entropy_loss.item():.4f}, "
                      f"RatioMean={ratios.mean().item():.4f}")

                # 累加损失用于计算平均值
                total_actor_loss += actor_loss.item() * num_timesteps_in_mini_batch
                total_critic_loss += critic_loss.item() * num_timesteps_in_mini_batch
                total_entropy_loss += entropy_loss.item() * num_timesteps_in_mini_batch
                processed_timesteps += num_timesteps_in_mini_batch

            if update_model:
                # 梯度裁剪
                grad_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.args.max_grad_norm)

                # 更新模型
                self.actor_critic_optim.step()

                # 打印本轮更新的总结
                avg_actor_loss = total_actor_loss / processed_timesteps
                avg_critic_loss = total_critic_loss / processed_timesteps
                avg_entropy_loss = total_entropy_loss / processed_timesteps
                print(f"  [总结] 更新迭代 {j + 1} 完成: "
                      f"AvgActorLoss={avg_actor_loss:.4f}, "
                      f"AvgValueLoss={avg_critic_loss:.4f}, "
                      f"AvgEntropy={avg_entropy_loss:.4f}, "
                      f"GradNorm={grad_norm:.4f}")

                # 恢复数据进行下一轮更新
                self._load_backup_data_to_store_data()
            else:
                # 如果早停，则清空所有数据，结束更新
                self._clear_store_data()
                self._clear_backup_data()
                print("  [总结] 由于KL散度超标，本轮所有更新已终止。")
                break  # 跳出 n_updates_per_iteration 循环

        self.actor_critic_optim.zero_grad()
        print("=" * 80)
        print(f"| 第 {self.i} 轮更新处理完毕。 |")
        print("=" * 80 + "\n")

    def _evaluate(self):
        causal_masks = self._get_causal_mask()
        mini_batch_obs_padding = self._get_mini_batch_obs_padding()
        mean, V = self.actor_critic(mini_batch_obs_padding, causal_masks)
        mean , V = self._remove_padding_of_action_and_V_then_flatten(mean, V)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(self.mini_batch_acts)
        return V, log_probs, dist.entropy()

    def _compute_rtgs(self):
        mini_batch_rtgs_nested = []
        for ep_rews in self.mini_batch_rews:
            discounted_reward = 0
            ep_rtgs = []
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.args.gamma
                ep_rtgs.append(discounted_reward)
            ep_rtgs.reverse()
            mini_batch_rtgs_nested.append(ep_rtgs)

        # 将嵌套列表展平为一维列表，然后转换为张量
        flattened_rtgs = list(itertools.chain.from_iterable(mini_batch_rtgs_nested))
        return torch.tensor(flattened_rtgs, dtype=torch.float).to(self.device)

    def _calculate_gae(self):
        mini_batch_advantages_nested = []
        for ep_rews, ep_vals, ep_dones, ep_len in zip(self.mini_batch_rews, self.mini_batch_vals, self.mini_batch_dones,
                                                      self.mini_batch_lens):
            advantages = []
            last_advantage = 0
            for t in reversed(range(ep_len)):
                if t + 1 < ep_len:
                    delta = ep_rews[t] + self.args.gamma * ep_vals[t + 1] * (1 - ep_dones[t + 1]) - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]
                advantage = delta + self.args.gamma * self.args.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage
                advantages.insert(0, advantage)
            mini_batch_advantages_nested.append(advantages)

        # 展平并转换
        flattened_advantages = list(itertools.chain.from_iterable(mini_batch_advantages_nested))
        return torch.tensor(flattened_advantages, dtype=torch.float).to(self.device)

    def _calculate_vtrace(self, curr_log_probs):
        mini_batch_advantages_nested = []
        curr_ep_log_probs = []
        batch_ep_log_probs = []
        start_idx = 0
        for ep_len in self.mini_batch_lens:
            curr_ep_log_probs.append(curr_log_probs[start_idx:start_idx + ep_len])
            batch_ep_log_probs.append(self.mini_batch_log_probs[start_idx:start_idx + ep_len])
            start_idx += ep_len
        for ep_rews, ep_vals, ep_dones, ep_log_probs, ep_curr_log_probs, ep_len in zip(
                self.mini_batch_rews, self.mini_batch_vals, self.mini_batch_dones, batch_ep_log_probs, curr_ep_log_probs, self.mini_batch_lens
        ):
            vtrace_values = []
            last_vtrace = 0
            for t in reversed(range(ep_len)):
                if t + 1 < ep_len:
                    delta = ep_rews[t] + self.args.gamma * ep_vals[t + 1] * (1 - ep_dones[t + 1]) - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]
                rho = torch.exp(ep_curr_log_probs[t] - ep_log_probs[t]).clamp(max=self.args.rho_clip).detach()
                c = torch.exp(ep_curr_log_probs[t] - ep_log_probs[t]).clamp(max=self.args.c_clip).detach()
                vtrace_increment = rho * delta + self.args.gamma * c * (1 - ep_dones[t]) * last_vtrace
                last_vtrace = vtrace_increment
                advantage = vtrace_increment
                vtrace_values.insert(0, advantage)
            mini_batch_advantages_nested.append(vtrace_values)

        flattened_advantages = [v.item() for sublist in mini_batch_advantages_nested for v in sublist]
        return torch.tensor(flattened_advantages, dtype=torch.float).to(self.device)

    def _get_causal_mask(self):
        max_timesteps = np.max(self.mini_batch_lens)
        causal_masks = []
        for ep_len in self.mini_batch_lens:
            causal_mask = torch.tril(torch.ones((max_timesteps, max_timesteps))).bool().to(self.device)  # [tgt_len, tgt_len]
            causal_mask = causal_mask.unsqueeze(0) # [1, tgt_len, tgt_len]
            key_mask = torch.zeros(max_timesteps).to(self.device)
            key_mask[:ep_len] = torch.ones(ep_len).to(self.device)
            key_mask = key_mask.bool().unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len]
            tgt_mask = causal_mask & key_mask
            causal_masks.append(tgt_mask)
        causal_masks = torch.stack(causal_masks, dim=0) # [batch_size, 1, tgt_len, tgt_len]
        return causal_masks

    def _get_mini_batch_obs_padding(self):
        max_timesteps = np.max(self.mini_batch_lens)
        mini_batch_obs_padding = []
        for ep_obs, ep_len in zip(self.mini_batch_obs, self.mini_batch_lens):
            expanded_ep_obs = torch.zeros(max_timesteps, self.args.obs_dim).to(self.device)
            expanded_ep_obs[:ep_len] = ep_obs
            mini_batch_obs_padding.append(expanded_ep_obs)
        mini_batch_obs_padding = torch.stack(mini_batch_obs_padding, dim=0)
        return mini_batch_obs_padding

    def _remove_padding_of_action_and_V_then_flatten(self, action, V):
        no_padding_flatten_V = []
        no_padding_flatten_action = []
        # V [batchsize, tgt_len, 1]
        # action [batchsize, tgt_len, act_dim]
        for idx ,ep_len in enumerate(self.mini_batch_lens):
            no_padding_flatten_V.append(V[idx,:ep_len, :])
            no_padding_flatten_action.append(action[idx,:ep_len, :])
        no_padding_flatten_V = torch.cat(no_padding_flatten_V, dim=0).squeeze()
        no_padding_flatten_action = torch.cat(no_padding_flatten_action, dim=0)
        return no_padding_flatten_action, no_padding_flatten_V

    def _update_curriculum(self, new_rewards):
        # 将新收集到的奖励添加到历史记录中
        self.recent_episode_rewards.extend(new_rewards)
        # 只保留最近 N 个奖励
        self.recent_episode_rewards = self.recent_episode_rewards[-self.args.curriculum_eval_episodes:]

        # 检查是否满足提升难度的条件
        if (len(self.recent_episode_rewards) == self.args.curriculum_eval_episodes and
                self.current_p_flicker_mean < self.args.target_p_flicker_mean):

            avg_reward = np.mean(self.recent_episode_rewards)

            if avg_reward > self.args.curriculum_reward_threshold:
                # 性能达标，提升难度
                old_mean = self.current_p_flicker_mean
                self.current_p_flicker_mean = min(
                    self.current_p_flicker_mean + self.args.p_flicker_mean_increment,
                    self.args.target_p_flicker_mean
                )

                print("\n" + "*" * 60)
                print(f"*** 课程学习: 难度提升！ ***")
                print(
                    f"*** 最近 {self.args.curriculum_eval_episodes} 轮平均奖励 {avg_reward:.2f} > 阈值 {self.args.curriculum_reward_threshold} ***")
                print(f"*** p_flicker 均值从 {old_mean:.3f} 增加到 {self.current_p_flicker_mean:.3f} ***")
                print("*" * 60 + "\n")

                # 提升难度后，可以清空奖励历史，以便在新难度下重新评估
                self.recent_episode_rewards = []

    def _collect_rewards_from_rollouts(self):
        rewards = [np.sum(ep_rews) for ep_rews in self.all_batch_rews]
        return rewards

    def _prepare_compute_data(self, backup=True):
        current_timesteps = 0
        while current_timesteps < self.args.timesteps_per_mini_batch and len(self.all_batch_lens):
            ep_obs = self.all_batch_obs.pop()
            self.mini_batch_obs.append(ep_obs)
            ep_acts = self.all_batch_acts.pop()
            self.mini_batch_acts.append(ep_acts)
            ep_log_probs = self.all_batch_log_probs.pop()
            self.mini_batch_log_probs.append(ep_log_probs)
            ep_vals = self.all_batch_vals.pop()
            self.mini_batch_vals.append(ep_vals)
            ep_rews = self.all_batch_rews.pop()
            self.mini_batch_rews.append(ep_rews)
            ep_dones = self.all_batch_dones.pop()
            self.mini_batch_dones.append(ep_dones)
            lens = self.all_batch_lens.pop()
            self.mini_batch_lens.append(lens)

            if backup:
                self.backup_batch_obs.append(ep_obs)
                self.backup_batch_acts.append(ep_acts)
                self.backup_batch_log_probs.append(ep_log_probs)
                self.backup_batch_vals.append(ep_vals)
                self.backup_batch_rews.append(ep_rews)
                self.backup_batch_dones.append(ep_dones)
                self.backup_batch_lens.append(lens)

            current_timesteps += self.mini_batch_lens[-1]

        if backup:
            print(f"当前总批次数据剩余{np.sum(self.all_batch_lens)}组，当前备份数据总数{np.sum(self.backup_batch_lens)}组")

        self.mini_batch_acts = torch.cat(self.mini_batch_acts, dim=0)
        self.mini_batch_log_probs = torch.cat(self.mini_batch_log_probs, dim=0)

    def _load_data_to_store_data(self, records):
        batch_ep_obs, batch_ep_acts, batch_ep_log_probs, batch_ep_rews, batch_lens, batch_ep_vals, batch_ep_dones = records
        for ep_obs, ep_acts, ep_log_probs, ep_rews, lens, ep_vals, ep_dones in zip(
                batch_ep_obs, batch_ep_acts, batch_ep_log_probs, batch_ep_rews, batch_lens, batch_ep_vals,
                batch_ep_dones):
            self.all_batch_obs.append(ep_obs.to(self.device))
            self.all_batch_acts.append(ep_acts.to(self.device))
            self.all_batch_log_probs.append(ep_log_probs.to(self.device))
            self.all_batch_vals.append(ep_vals)
            self.all_batch_rews.append(ep_rews)
            self.all_batch_dones.append(ep_dones)
            self.all_batch_lens.append(lens)

    def _load_backup_data_to_store_data(self):
        while len(self.backup_batch_lens):
            self.all_batch_obs.append(self.backup_batch_obs.pop())
            self.all_batch_acts.append(self.backup_batch_acts.pop())
            self.all_batch_log_probs.append(self.backup_batch_log_probs.pop())
            self.all_batch_vals.append(self.backup_batch_vals.pop())
            self.all_batch_rews.append(self.backup_batch_rews.pop())
            self.all_batch_dones.append(self.backup_batch_dones.pop())
            self.all_batch_lens.append(self.backup_batch_lens.pop())

    def _clear_store_data(self):
        while len(self.all_batch_lens):
            self.all_batch_obs.pop()
            self.all_batch_acts.pop()
            self.all_batch_log_probs.pop()
            self.all_batch_vals.pop()
            self.all_batch_rews.pop()
            self.all_batch_dones.pop()
            self.all_batch_lens.pop()

    def _clear_backup_data(self):
        while len(self.backup_batch_lens):
            self.backup_batch_obs.pop()
            self.backup_batch_acts.pop()
            self.backup_batch_log_probs.pop()
            self.backup_batch_vals.pop()
            self.backup_batch_rews.pop()
            self.backup_batch_dones.pop()
            self.backup_batch_lens.pop()

    def _init_compute_data(self):
        self.mini_batch_obs = []
        self.mini_batch_acts = []
        self.mini_batch_log_probs = []
        self.mini_batch_vals = []
        self.mini_batch_rews = []
        self.mini_batch_dones = []
        self.mini_batch_lens = []

    def _init_store_data(self):
        self.all_batch_obs = []
        self.all_batch_acts = []
        self.all_batch_log_probs = []
        self.all_batch_vals = []
        self.all_batch_rews = []
        self.all_batch_dones = []
        self.all_batch_lens = []

    def _init_backup_data(self):
        self.backup_batch_obs = []
        self.backup_batch_acts = []
        self.backup_batch_log_probs = []
        self.backup_batch_vals = []
        self.backup_batch_rews = []
        self.backup_batch_dones = []
        self.backup_batch_lens = []

    def _move_optimizer_state(self, device):
        """将优化器的所有状态张量移动到指定设备"""
        for state in self.actor_critic_optim.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    def _optimizer_to_cpu(self):
        """将优化器状态移动到CPU，释放显存"""
        if self.device.type == 'cuda':  # 只在GPU模式下执行
            print("\n[优化器管理] 正在将优化器状态移动到 CPU 以进行数据收集...")
            # 在移动前同步，确保所有之前的GPU计算（如上一次的梯度更新）已完成
            torch.cuda.synchronize(self.device)
            self._move_optimizer_state(torch.device('cpu'))
            # 在移动后再次同步，确保数据传输操作本身完成
            torch.cuda.synchronize(self.device)
            # (可选) 清理一下缓存，最大化释放显存的效果
            torch.cuda.empty_cache()
            print("[优化器管理] 移动完成，GPU显存已为数据收集优化。")

    def _optimizer_to_gpu(self):
        """将优化器状态移回GPU，准备训练"""
        if self.device.type == 'cuda':  # 只在GPU模式下执行
            print("\n[优化器管理] 正在将优化器状态移回 GPU 以进行模型更新...")
            # 在移动前同步是好的实践
            torch.cuda.synchronize(self.device)
            self._move_optimizer_state(self.device)  # self.device 是 'cuda'
            # 移动后进行同步，确保优化器状态已经全部到达GPU
            torch.cuda.synchronize(self.device)
            print("[优化器管理] 移动完成，准备开始训练。")

    def _save(self):
        current_time = time()
        if current_time - self.last_save_time >= self.args.save_interval_seconds:
            print(f"\n距离上次保存已过去 {(current_time - self.last_save_time) / 3600:.2f} 小时。正在保存模型...")
            self._optimizer_to_cpu()

            checkpoint = {
                'policy_state_dict': self.actor_critic.state_dict(),
                'optimizer_state_dict': self.actor_critic_optim.state_dict(),
                'last_save_time': current_time,  # <-- 保存当前时间戳
                'iteration': self.i,  # <-- (可选) 同时保存当前迭代次数
                'current_p_flicker_mean': self.current_p_flicker_mean
            }
            torch.save(checkpoint, self.args.gid + '.pth')

            # 更新上次保存时间
            self.last_save_time = current_time
            print(f"保存成功！下一次保存将在大约 {self.args.save_interval_seconds / 3600:.1f} 小时后。")

    def _load(self):
        try:
            # 加载到CPU
            checkpoint = torch.load(self.args.gid + '.pth', map_location='cpu')
            self.actor_critic.load_state_dict(checkpoint['policy_state_dict'])

            # 加载优化器状态到CPU
            self.actor_critic_optim.load_state_dict(checkpoint['optimizer_state_dict'])
            print("模型和优化器状态已加载到CPU。")

            # 现在，将模型移到目标设备（GPU）
            self.actor_critic.to(self.device)
            print(f"模型已移动到目标设备: {self.device}")

            # 注意：优化器状态仍然在CPU上，这正是我们想要的初始状态。
            # 它会在第一次训练前被 _optimizer_to_gpu() 移动到GPU。
            if 'last_save_time' in checkpoint:
                self.last_save_time = checkpoint['last_save_time']
            else:
                self.last_save_time = None
            if self.args.use_automated_curriculum:
                if 'current_p_flicker_mean' in checkpoint:
                    self.current_p_flicker_mean = checkpoint['current_p_flicker_mean']
                    print(f"当前p_flicker_mean为：{self.current_p_flicker_mean}")
                else:
                    self.current_p_flicker_mean = self.args.initial_p_flicker_mean
            if 'iteration' in checkpoint:
                self.i = checkpoint['iteration']
                print(f"从迭代次数 {self.i} 继续训练。")
            else:
                self.i = None
            print(f"模型加载成功。优化器状态当前位于CPU。上次保存时间戳为: {self.last_save_time}")

        except FileNotFoundError:
            print("未找到模型文件，将从头开始训练。")
            self.last_save_time = None
            if self.args.use_automated_curriculum:
                self.current_p_flicker_mean = self.args.initial_p_flicker_mean
            self.i = None
        except Exception as e:
            print(f"加载模型时发生错误: {e}。将从头开始训练。")
            self.last_save_time = None
            if self.args.use_automated_curriculum:
                self.current_p_flicker_mean = self.args.initial_p_flicker_mean
            self.i = None


test = PPO()
test.learn(1000000)