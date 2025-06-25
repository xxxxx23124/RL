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
                obs[i] = 0
        return obs

@dataclass
class Config:
    gid = "LunarLanderContinuous-v2"
    save_interval_seconds = 3 * 3600
    p_flicker = 0.0
    cov_var_value:float = 0.5
    n_updates_per_iteration = 3
    lr = 1e-6
    clip = 0.05
    rho_clip = 1.0
    c_clip = 1.0
    ent_coef = 1e-4
    vf_coef = 0.5
    # 都不启用代表使用单线程
    max_cpu_threads = 0
    max_gpu_threads = 0
    max_timesteps_per_episode = 1200
    timesteps_per_mini_batch = 40 #代表只处理一个轨迹，因为40太小了
    timesteps_per_batch = 16000
    timesteps_all_batch = (max_cpu_threads + max_gpu_threads) * timesteps_per_batch
    max_grad_norm = 0.5
    target_kl = 0.01
    gamma = 0.99
    lam = 0.99
    backbone_layers: int = 22
    actor_layers: int = 22
    critic_layers: int = 8
    encoder_d_model: int = 1080
    encoder_d_ff: int = encoder_d_model * 3
    encoder_attention_heads: int = 12
    def __post_init__(self):
        env = gym.make(self.gid)
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.text_max_len = self.max_timesteps_per_episode + 100

        if self.timesteps_all_batch == 0:
            self.timesteps_all_batch = self.timesteps_per_batch

        assert (self.encoder_d_model % self.encoder_attention_heads == 0)


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

    def forward(self, x: torch.Tensor, seq_len: int):
        # x shape: (batch, n_heads, seq_len, d_k)

        # 裁剪到当前序列长度
        cos = self.freqs_cos[:, :, :seq_len, :]  # shape: (1, 1, seq_len, d_k)
        sin = self.freqs_sin[:, :, :seq_len, :]  # shape: (1, 1, seq_len, d_k)

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

    def attention(self, query, key, value, mask: torch.Tensor, rotary_emb: RotaryEmbedding, seq_len: int):
        # query, key, value shape: (batch_size, num_heads, seq_len, d_k)

        # *** RoPE 应用 ***
        # 对 query 和 key 应用旋转位置编码
        query = rotary_emb(query, seq_len=seq_len)
        key = rotary_emb(key, seq_len=seq_len)

        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # 确保 mask 维度匹配 (batch, heads, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value)

    def forward(self, query, key, value, mask, rotary_emb: RotaryEmbedding):
        batch_size = query.size(0)
        seq_len = query.size(1)

        query = self.proj_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)  # (bs, n_heads, seq_len, d_k)
        key = self.proj_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        value = self.proj_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        x = self.attention(query, key, value, mask, rotary_emb, seq_len)

        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.d_k)
        return self.linear_out(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_feedforward):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_feedforward)
        self.w2 = nn.Linear(d_feedforward, d_model)
        self.w3 = nn.Linear(d_model, d_feedforward)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class EncoderLayer(nn.Module):
    def __init__(self, args: Config):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(args.encoder_d_model, args.encoder_attention_heads)
        self.feed_forward = FeedForward(args.encoder_d_model, args.encoder_d_ff)
        self.norm1 = nn.LayerNorm(args.encoder_d_model)
        self.norm2 = nn.LayerNorm(args.encoder_d_model)

    def forward(self, x, mask, rotary_emb: RotaryEmbedding):
        x2 = self.norm1(x)
        x = x + self.self_attn(x2, x2, x2, mask, rotary_emb)
        x2 = self.norm2(x)
        x = x + self.feed_forward(x2)
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
        self.actor_head = EncoderLayer(args)
        self.critic_head = EncoderLayer(args)
        self.actor = nn.ModuleList([EncoderLayer(args) for _ in range(args.actor_layers)])
        self.critic = nn.ModuleList([EncoderLayer(args) for _ in range(args.critic_layers)])
        self.actor_norm = nn.LayerNorm(args.encoder_d_model)
        self.actor_end = nn.Linear(args.encoder_d_model, args.act_dim)
        self.critic_norm = nn.LayerNorm(args.encoder_d_model)
        self.critic_end = nn.Linear(args.encoder_d_model, 1)

    def forward(self, src, mask):
        src = self.embedding(src) * math.sqrt(self.d_model)
        for layer in self.backbone:
            src = layer(src, mask, self.rotary_emb)
        src = self.backbone_norm(src)

        actor = self.actor_head(src, mask, self.rotary_emb)
        for layer in self.actor:
            actor = layer(actor, mask, self.rotary_emb)
        actor = self.actor_norm(actor)
        actor = self.actor_end(actor)

        critic = self.critic_head(src, mask, self.rotary_emb)
        for layer in self.critic:
            critic = layer(critic, mask, self.rotary_emb)
        critic = self.critic_norm(critic)
        critic = self.critic_end(critic)
        return actor, critic


class Transformer(nn.Module):
    def __init__(self, args:Config = Config()):
        super(Transformer, self).__init__()
        self.args = args
        self.encoder = Encoder(self.args)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.001)
                elif isinstance(module, nn.Embedding):
                    torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, src, mask):
        actor, critic = self.encoder(src, mask)
        return actor, critic


class Environment:
    def __init__(self, id, device, render_mode=None, requires_grad=False, args:Config = Config()):
        self.id = str(id) + " " + device
        self.device = torch.device(device)
        self.timesteps_per_batch = args.timesteps_per_batch
        self.max_timesteps_per_episode = args.max_timesteps_per_episode
        self.gid = args.gid
        self.env = gym.make(self.gid, render_mode = render_mode) #"human" None
        self.env = LunarLanderPOMDPWrapper(self.env, p_flicker=args.p_flicker)
        self.obs_dim = args.obs_dim
        self.observation, self.info = self.env.reset()
        self.reward = None
        self.terminated = False
        self.truncated = False
        self.act_dim = args.act_dim
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=args.cov_var_value)
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
            while ep_t < self.max_timesteps_per_episode and not (self.terminated or self.truncated):
                ep_t += 1
                t += 1
                self.observation = torch.tensor(self.observation, dtype=torch.float).to(self.device)
                ep_obs.append(self.observation.unsqueeze(0))
                action, log_prob, V = self._get_action(ep_obs, ep_t)
                self.observation, self.reward, self.terminated, self.truncated, self.info = self.env.step(action.cpu().numpy())
                ep_acts.append(action.unsqueeze(0))
                ep_log_probs.append(log_prob.unsqueeze(0))
                ep_rews.append(self.reward)
                ep_vals.append(V.item())
                ep_dones.append(self.terminated or self.truncated)

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

    def _get_action(self, ep_obs, ep_t):
        with torch.no_grad():
            causal_mask = torch.tril(torch.ones((ep_t, ep_t))).bool().to(self.device)  # [tgt_len, tgt_len]
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0) # [1, 1, tgt_len, tgt_len]
            ep_obs = torch.cat(ep_obs, dim=0).unsqueeze(0)
            mean, V = self.actor_critic(ep_obs, mask = causal_mask)
            dist = MultivariateNormal(mean[0][-1], self.cov_mat)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob, V[0][-1]

    def _report_info(self):
        # 0. 基本信息
        num_episodes = len(self.batch_ep_rews)
        if num_episodes == 0:
            print(f"--- 环境 {self.id} 报告 (无数据) ---")
            return

        total_timesteps = np.sum(self.batch_lens)
        total_duration = np.sum(self.ep_time_consume)

        # 1. 计算统计量 (低开销)
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

        # 2. 格式化输出
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

    def _test_max_memory(self):
        print(f"\n[内存测试] 环境id: {self.id} 在设备 {self.device} 上开始最大内存占用测试...")
        if not self.requires_grad:
            for param in self.actor_critic.parameters():
                param.requires_grad = True
        # batch_size = 1, seq_len = max_timesteps_per_episode, feature_dim = obs_dim
        max_len = self.max_timesteps_per_episode
        dummy_obs_seq = torch.zeros((1, max_len, self.obs_dim), dtype=torch.float).to(self.device)
        # 形状: [batch_size, n_heads, seq_len, seq_len] 或可广播的 [1, 1, seq_len, seq_len]
        causal_mask = torch.tril(torch.ones((max_len, max_len))).bool().to(self.device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # 形状变为 [1, 1, max_len, max_len]
        print(f"[内存测试] 已生成形状为 {dummy_obs_seq.shape} 的模拟数据。")
        print("[内存测试] 现在执行前向和反向传播... 请监控 GPU/CPU 内存使用情况。")
        try:
            with torch.enable_grad():
                actor_output, critic_output = self.actor_critic(dummy_obs_seq, mask=causal_mask)
                # actor_output shape: [1, max_len, act_dim]
                # critic_output shape: [1, max_len, 1]
                print(
                    f"[内存测试] 前向传播完成。Actor输出形状: {actor_output.shape}, Critic输出形状: {critic_output.shape}")
                dummy_actor_target = torch.randn_like(actor_output)
                dummy_critic_target = torch.randn_like(critic_output)
                loss_actor = F.mse_loss(actor_output, dummy_actor_target)
                loss_critic = F.mse_loss(critic_output, dummy_critic_target)
                total_loss = loss_actor + loss_critic
                # 执行反向传播
                total_loss.backward()
                print(f"[内存测试] 反向传播完成。总虚拟损失: {total_loss.item()}")
            print(f"[内存测试] 成功！在设备 {self.device} 上完成了一次最大负载的计算。")
        except Exception as e:
            print(f"[内存测试] 失败！在执行过程中发生错误: {e}")
            print(
                "[内存测试] 这很可能是一次内存溢出(OOM)错误。建议减小 `max_timesteps_per_episode` 或模型尺寸 (`encoder_d_model`等)。")
        finally:
            # 恢复模型的无梯度状态，以免影响正常的rollout
            if not self.requires_grad:
                for param in self.actor_critic.parameters():
                    param.requires_grad = False
            # 清理CUDA缓存（如果有）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[内存测试] 测试结束，模型已恢复至无梯度状态。")


class PPO(Environment):
    def __init__(self, args:Config = Config()):
        self.gid = args.gid
        self.n_updates_per_iteration = args.n_updates_per_iteration
        self.clip = args.clip
        self.rho_clip = args.rho_clip
        self.c_clip = args.c_clip
        self.vf_coef = args.vf_coef
        self.ent_coef = args.ent_coef
        self.timesteps_needed = args.timesteps_all_batch
        self.timesteps_per_mini_batch = args.timesteps_per_mini_batch
        super(PPO, self).__init__(0, "cuda", requires_grad=True)
        self.actor_critic_optim = torch.optim.NAdam(self.actor_critic.parameters(),lr=args.lr)
        self.max_grad_norm = args.max_grad_norm
        self.target_kl = args.target_kl
        self.gamma = args.gamma
        self.lam = args.lam
        self.use_gae = True
        self.use_vtrace = True
        self._load()
        self._init_store_data()
        self._init_compute_data()
        self._init_backup_data()
        self.sub_envs = []
        self.max_cpu_threads = args.max_cpu_threads
        for i in range(self.max_cpu_threads):
            self.sub_envs.append(Environment(i + 1, "cpu"))
        self.max_gpu_threads = args.max_gpu_threads
        for i in range(self.max_gpu_threads):
            self.sub_envs.append(Environment(i + 1, "cuda"))
        self.i = 0

        self.save_interval_seconds = args.save_interval_seconds
        if 'last_save_time' in self.__dict__ and self.last_save_time is not None:
             print(f"从检查点恢复上次保存时间: {self.last_save_time}")
        else:
            self.last_save_time = time()
            print(f"初始化保存计时器。首次保存将在大约 {self.save_interval_seconds / 3600:.1f} 小时后。")

    def learn(self, simulation_number):
        if len(self.sub_envs):
            print("多线程训练")
            for sub_env in self.sub_envs:
                sub_env._set_agent(self.actor_critic)
                sub_env._test_max_memory()
            self._multithread_learn(simulation_number)
        else:
            print("单线程训练")
            self._test_max_memory()
            self._singlethread_learn(simulation_number)

    def _singlethread_learn(self, simulation_number):
        while self.i < simulation_number:
            records = self.rollout()
            cur_timesteps = np.sum(self.batch_lens)
            self._load_data_to_store_data(records)
            print(f"距离模型迭代{cur_timesteps}/{self.timesteps_needed}组数据")
            if cur_timesteps >= self.timesteps_needed:
                self.i += 1
                self._compute_loss()
                self._save()

    def _multithread_learn(self, simulation_number):
        with (ThreadPoolExecutor(max_workers=len(self.sub_envs)) as thread_pool_executor):
            futures = set()
            future_to_env = {}
            cur_timesteps = 0
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
                    print(f"距离模型迭代{cur_timesteps}/{self.timesteps_needed}组数据")
                if cur_timesteps >= self.timesteps_needed:
                    self.i += 1
                    self._compute_loss()
                    self._save()
                    cur_timesteps = 0
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

            if self.use_gae and not self.use_vtrace:
                A_k = self._calculate_gae()
                mini_batch_rtgs = A_k + V.detach()
            elif self.use_vtrace:
                A_k = self._calculate_vtrace(curr_log_probs)
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
        for j in range(self.n_updates_per_iteration):
            print(f"\n  [更新迭代 {j + 1}/{self.n_updates_per_iteration}]")
            self.actor_critic_optim.zero_grad()

            # 存储每一轮的损失
            total_actor_loss, total_critic_loss, total_entropy_loss = 0, 0, 0
            processed_timesteps = 0
            mini_batch_idx = -1

            while len(self.all_batch_lens):
                mini_batch_idx += 1
                self._init_compute_data()
                # 最后一个epoch不需要备份，因为数据将被丢弃
                is_last_update_iter = (j == self.n_updates_per_iteration - 1)
                self._prepare_compute_data(backup=not is_last_update_iter)

                V, curr_log_probs, entropy = self._evaluate()

                # 计算损失
                entropy_loss = entropy.mean()
                logratios = curr_log_probs - self.mini_batch_log_probs
                ratios = torch.exp(logratios)

                approx_kl = ((ratios - 1) - logratios).mean()

                # 早停机制
                if approx_kl > self.target_kl:  # 使用 target_kl 作为硬停止阈值
                    print(f"    ! [早停] Mini-batch #{mini_batch_idx + 1}: KL散度 ({approx_kl.item():.4f}) 显著超过目标 ({self.target_kl}). 停止本轮所有更新。")
                    update_model = False
                    break  # 跳出 mini-batch 循环

                surr1 = ratios * A_k_list[mini_batch_idx]
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k_list[mini_batch_idx]

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = self.vf_coef * nn.MSELoss()(V, mini_batch_rtgs_list[mini_batch_idx])

                # 总损失
                loss = actor_loss + critic_loss - self.ent_coef * entropy_loss
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
                grad_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

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
        batch_rtgs = []
        for ep_rews in self.mini_batch_rews:
            discounted_reward = 0
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.append(discounted_reward)
        batch_rtgs.reverse()
        return torch.tensor(batch_rtgs, dtype=torch.float).to(self.device)

    def _calculate_gae(self):
        batch_advantages = []
        for ep_rews, ep_vals, ep_dones, ep_len in zip(self.mini_batch_rews, self.mini_batch_vals, self.mini_batch_dones, self.mini_batch_lens):
            advantages = []
            last_advantage = 0

            for t in reversed(range(ep_len)):
                if t + 1 < ep_len:
                    delta = ep_rews[t] + self.gamma * ep_vals[t + 1] * (1 - ep_dones[t + 1]) - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]
                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage
                advantages.insert(0, advantage)

            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float).to(self.device)

    def _calculate_vtrace(self, curr_log_probs):
        batch_advantages = []
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
                    delta = ep_rews[t] + self.gamma * ep_vals[t + 1] * (1 - ep_dones[t + 1]) - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]
                rho = torch.exp(ep_curr_log_probs[t] - ep_log_probs[t]).clamp(max=self.rho_clip).detach()
                c = torch.exp(ep_curr_log_probs[t] - ep_log_probs[t]).clamp(max=self.c_clip).detach()
                vtrace_increment = rho * delta + self.gamma * c * (1 - ep_dones[t]) * last_vtrace
                last_vtrace = vtrace_increment
                advantage = vtrace_increment
                vtrace_values.insert(0, advantage)
            batch_advantages.extend(vtrace_values)
        return torch.tensor(batch_advantages, dtype=torch.float).to(self.device)

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
            expanded_ep_obs = torch.zeros(max_timesteps, self.obs_dim).to(self.device)
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

    def _prepare_compute_data(self, backup=True):
        current_timesteps = 0
        while current_timesteps < self.timesteps_per_mini_batch and len(self.all_batch_lens):
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

    def _save(self):
        current_time = time()
        if current_time - self.last_save_time >= self.save_interval_seconds:
            print(f"\n距离上次保存已过去 {(current_time - self.last_save_time) / 3600:.2f} 小时。正在保存模型...")

            checkpoint = {
                'policy_state_dict': self.actor_critic.state_dict(),
                'optimizer_state_dict': self.actor_critic_optim.state_dict(),
                'last_save_time': current_time,  # <-- 保存当前时间戳
                'iteration': self.i  # <-- (可选) 同时保存当前迭代次数
            }
            torch.save(checkpoint, self.gid + '.pth')

            # 更新上次保存时间
            self.last_save_time = current_time
            print(f"保存成功！下一次保存将在大约 {self.save_interval_seconds / 3600:.1f} 小时后。")

    def _load(self):
        try:
            checkpoint = torch.load(self.gid + '.pth')
            self.actor_critic.load_state_dict(checkpoint['policy_state_dict'])
            self.actor_critic_optim.load_state_dict(checkpoint['optimizer_state_dict'])
            # 加载上次保存的时间戳
            if 'last_save_time' in checkpoint:
                self.last_save_time = checkpoint['last_save_time']
            else:
                self.last_save_time = time()  # 如果旧模型没有时间戳，则从现在开始计时
            # (可选) 加载迭代次数
            if 'iteration' in checkpoint:
                self.i = checkpoint['iteration']
                print(f"从迭代次数 {self.i} 继续训练。")
            print(f"模型和优化器状态加载成功。上次保存时间戳为: {self.last_save_time}")
        except FileNotFoundError:
            print("未找到模型文件，将从头开始训练。")
            self.last_save_time = None  # 确保在__init__中重新初始化
            pass
        except Exception as e:
            print(f"加载模型时发生错误: {e}。将从头开始训练。")
            self.last_save_time = None
            pass


test = PPO()
test.learn(1000000)