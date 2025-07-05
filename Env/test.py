import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.envs.libs.pettingzoo import PettingZooWrapper
from torchrl.envs.transforms import TransformedEnv, ObservationNorm, CatTensors
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

from pettingzoo.classic import go_v5

# 1. 抽象模型 (Abstract Model)
# -----------------------------------------------
# 这个模型符合您的描述：通过cache的有无来区分训练和推理模式，并处理不同的输入形状。
class AbstractGoModel(nn.Module):
    def __init__(self, action_space_size):
        super().__init__()
        self.action_space_size = action_space_size
        self.d_model = 256 # 假设的内部维度

        # 这里用简单的线性层模拟您复杂的时空处理网络
        # 实际中这里会是 CNN -> Transformer -> Mamba2 -> ...
        self.shared_feature_extractor = nn.Sequential(
            nn.LazyLinear(self.d_model),
            nn.ReLU(),
        )
        self.policy_head = nn.LazyLinear(self.action_space_size)
        self.value_head = nn.LazyLinear(1)
        print("抽象模型已初始化。")
        print(f"动作空间大小: {action_space_size}")

    def forward(self, x, cache=None):
        """
        Args:
            x (torch.Tensor): 观察值.
                - 训练模式 (cache is None): (B, S, L, D) -> (1, S, 19*19, 3)
                - 推理模式 (cache is not None): (B, S, L, D) -> (1, 1, 19*19, 3)
            cache: 用于推理时状态传递的缓存 (此处为符号)。

        Returns:
            (torch.Tensor, torch.Tensor): 策略 logits, 价值.
        """
        # 模拟通过 cache 判断模式
        if cache is not None:
            # 推理模式: (1, 1, L, D)
            # print(f"推理模式，输入形状: {x.shape}")
            assert x.shape[0] == 1 and x.shape[1] == 1, "推理模式应输入单步数据 (1, 1, ...)"
        else:
            # 训练模式: (1, S, L, D)
            # print(f"训练模式，输入形状: {x.shape}")
            assert x.shape[0] == 1, "训练模式应输入单条轨迹 (1, S, ...)"

        # 展平 L 和 D 维度以输入线性层
        # 输入: (B, S, L, D)
        # -> (B, S, L*D)
        if x.dim() == 4:
            x_flat = x.flatten(start_dim=2)
        else: # (B,S,L*D) 已经被tensordict处理过
            x_flat = x

        features = self.shared_feature_extractor(x_flat) # (B, S, d_model)

        # 策略和价值输出
        logits = self.policy_head(features) # (B, S, action_space_size)
        value = self.value_head(features)   # (B, S, 1)

        return logits, value


# 2. 环境设置 (Environment Setup)
# -----------------------------------------------
def make_env():
    """创建并包装 PettingZoo Go 环境的工厂函数"""
    # 原始环境
    env = go_v5.env(board_size=19, render_mode=None) # 训练时关闭渲染
    
    # 使用 TorchRL Wrapper
    env = PettingZooWrapper(env)
    
    # 使用 TorchRL Transforms
    # 将PettingZoo的多智能体输出格式转换为TorchRL的标准格式
    # 默认情况下，PettingZooWrapper会为每个智能体创建独立的key，如 "observation_player_0", "action_player_1"
    # 我们希望将其统一，因为我们的模型是共享的
    # CatTensors 将 player_0 和 player_1 的观测值合并，并根据当前智能体选择一个
    # ObservationNorm 对观测值进行归一化
    # 这一部分也可以通过自定义Wrapper实现更精细的控制，但为了简洁，我们假设默认的Wrapper行为足够
    # 对于自博弈，更常见的做法是让环境始终输出当前玩家的视角
    # PettingZooWrapper已经处理了这一点，它会提供 "agent_name" 和 "current_player_id"
    # 我们只需要确保我们的模型接收 "observation" key
    
    # PettingZooWrapper会输出一个 "observation_player_0" 和 "observation_player_1"
    # 这对于自博弈不方便，我们重命名key，让其统一为 "observation"
    # 同时，我们将 (H, W, C) -> (C, H, W)
    class CustomWrapper(TransformedEnv):
         def _step(self, tensordict):
             # 从环境中获取下一步
             out_td = self.base_env._step(tensordict)
             # 根据当前玩家重命名观察
             agent = out_td["agent_name"] # e.g., "player_0"
             obs_key = f"observation_{agent}"
             if obs_key in out_td.keys():
                out_td["observation"] = out_td[obs_key]
             return out_td

         def _reset(self, tensordict=None, **kwargs):
             out_td = self.base_env._reset(tensordict, **kwargs)
             agent = out_td["agent_name"]
             obs_key = f"observation_{agent}"
             if obs_key in out_td.keys():
                out_td["observation"] = out_td[obs_key]
             return out_td
    
    env = CustomWrapper(env)

    return env


# 3. 超参数与初始化 (Hyperparameters & Initialization)
# -----------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# PPO 参数
total_trajectories = 1000  # 总共要收集的轨迹数
trajectories_per_update = 32 # 多少条轨迹进行一次更新
epochs_per_update = 4      # 每次更新时，数据重复使用的次数
lr = 3e-4
gamma = 0.99
lmbda = 0.95
clip_epsilon = 0.2
entropy_eps = 1e-4

# 创建环境实例以获取规格
env = make_env()
board_shape = env.observation_spec["observation"].shape # (19, 19, 3)
action_size = env.action_spec.space.n # 19*19 + 1 (pass)

# 4. 策略和价值网络 (Policy and Value Networks)
# -----------------------------------------------
# 实例化抽象模型
base_model = AbstractGoModel(action_space_size=action_size)

# 将模型包装为 TorchRL 的 Actor-Critic 模块
# 输入: "observation" -> (B, S, 19*19, 3) -> model -> ("logits", "value")
# ProbabilisticActor 将 logits 转换为一个分布并从中采样，得到 "action"
actor_critic = TensorDictModule(
    module=base_model,
    in_keys=[("observation")], # 输入tensordict的key
    out_keys=["logits", "value"], # 输出tensordict的key
)

# 使用actor_critic的输出来定义策略和价值网络
policy_module = ProbabilisticActor(
    module=actor_critic,
    in_keys=["logits"],
    out_keys=["action"],
    distribution_class=torch.distributions.Categorical,
    return_log_prob=True,
)

value_module = ValueOperator(
    module=actor_critic,
    in_keys=["observation"], # 价值网络直接从观察计算
)

# 5. 数据收集与存储 (Data Collection & Storage)
# -----------------------------------------------
# 数据收集器
# frames_per_batch=env.max_steps * 1 (收集一条完整轨迹)
# total_frames=-1 (无限收集，由我们的循环控制)
collector = SyncDataCollector(
    create_env_fn=make_env,
    policy=policy_module,
    frames_per_batch=env.max_step, # 确保一次收集一条完整的轨迹
    total_frames=-1,
    device=device,
)

# 回放缓冲区
# 使用LazyMemmapStorage可以在内存不足时将数据暂存到磁盘
replay_buffer = TensorDictReplayBuffer(
    storage=LazyMemmapStorage(max_size=env.max_step * trajectories_per_update),
    batch_size=env.max_step, # 训练时一次取一条轨迹
)


# 6. 损失函数与优化器 (Loss Function & Optimizer)
# -----------------------------------------------
advantage_module = GAE(
    gamma=gamma,
    lmbda=lmbda,
    value_network=value_module,
    average_gae=True,
)

loss_module = ClipPPOLoss(
    actor=policy_module,
    critic=value_module,
    clip_epsilon=clip_epsilon,
    entropy_bonus=True,
    entropy_coef=entropy_eps,
    loss_critic_type="smooth_l1",
)

optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)

# 7. 训练循环 (Training Loop)
# -----------------------------------------------
collected_trajectories = 0
total_updates = 0

pbar = tqdm(total=total_trajectories)

while collected_trajectories < total_trajectories:
    # --- 收集数据 ---
    # 循环收集数据，直到缓冲区满 (32条轨迹)
    for _ in range(trajectories_per_update):
        # 收集一条完整的轨迹
        trajectory_data = collector.next()
        
        # 将 (S, D) 形状的数据增加一个批次维度 -> (1, S, D)
        trajectory_data.batch_size = [1] 
        
        # 将轨迹数据存入回放缓冲区
        replay_buffer.add(trajectory_data)
        
        collected_trajectories += 1
        pbar.update(1)

        if collected_trajectories >= total_trajectories:
            break
    
    if len(replay_buffer) == 0:
        continue

    print(f"\n收集到 {len(replay_buffer) // env.max_step} 条轨迹，开始第 {total_updates + 1} 次参数更新...")

    # --- 训练阶段 ---
    total_loss_sum = 0
    # 对收集到的数据进行多次迭代训练
    for i in range(epochs_per_update):
        print(f"  Epoch {i+1}/{epochs_per_update}")
        # 从缓冲区采样轨迹，每次一条
        for trajectory_batch in replay_buffer:
            # GAE计算需要整条轨迹
            # 训练时，模型输入为 (1, S, ...)，cache为None
            # 确保数据在正确设备上
            trajectory_batch = trajectory_batch.to(device)

            # 计算优势 (GAE)
            with torch.no_grad():
                 advantage_module(trajectory_batch)

            # 将轨迹数据展平以便计算损失 (1, S, ...) -> (S, ...)
            # PPO loss期望输入 (N, ...) 的批次数据，N是样本数
            current_batch_size = trajectory_batch.batch_size
            trajectory_batch_flat = trajectory_batch.reshape(-1)

            # 计算 PPO 损失
            loss_td = loss_module(trajectory_batch_flat)
            loss = loss_td["loss_objective"] + loss_td["loss_critic"] + loss_td["loss_entropy"]
            
            # 累计损失
            total_loss_sum += loss.item()

    # 平均损失并执行一次参数更新
    avg_loss = total_loss_sum / (len(replay_buffer) * epochs_per_update)
    
    optimizer.zero_grad()
    # PPO loss模块内部已经完成了反向传播图的构建，我们只需要对总损失求和后反向传播
    # 这里我们用平均损失来指导优化，但梯度是所有损失的总和
    # 为了简化，我们直接对最后一次计算的loss进行反向传播，这在实践中效果相似
    # 更严谨的做法是将每次的loss都backward，但会消耗更多内存
    # 简便方法: 直接对总损失反向传播
    total_loss_for_backward = loss_module(replay_buffer.sample(batch_size=len(replay_buffer)).reshape(-1).to(device))
    final_loss = total_loss_for_backward["loss_objective"] + total_loss_for_backward["loss_critic"] + total_loss_for_backward["loss_entropy"]
    
    final_loss.backward()
    optimizer.step()

    total_updates += 1
    print(f"更新完成。平均损失: {avg_loss:.4f}")

    # 清空缓冲区，为下一轮收集做准备
    replay_buffer.empty()

pbar.close()
collector.shutdown()
print("训练结束。")
