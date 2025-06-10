import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from copy import deepcopy
from time import time
import math
from dataclasses import dataclass

@dataclass
class Config:
    gid = "LunarLanderContinuous-v2"
    cov_var_value:float = 0.5
    n_updates_per_iteration = 5
    lr = 1e-4
    clip = 0.2
    rho_clip = 1.0
    c_clip = 1.0
    ent_coef = 0.01
    max_cpu_threads = 2
    max_gpu_threads = 2
    max_timesteps_per_episode = 2000
    timesteps_per_mini_batch = 1600
    timesteps_per_batch = 3 * timesteps_per_mini_batch
    timesteps_all_batch = (max_cpu_threads + max_gpu_threads) * timesteps_per_batch
    max_grad_norm = 0.5
    target_kl = 0.03
    break_kl = 3.0
    gamma = 0.99
    lam = 0.99
    backbone_layers: int = 3
    actor_critic_cross_attn_layers: int = 3
    encoder_d_model: int = 256
    encoder_d_ff: int = 512
    encoder_attention_heads: int = 8
    encoder_num_experts: int = 2
    encoder_top_experts: int = 1
    def __post_init__(self):
        env = gym.make(self.gid)
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.text_max_len = self.max_timesteps_per_episode + 100
        assert (self.encoder_d_model % self.encoder_attention_heads == 0
                and self.encoder_num_experts >= self.encoder_top_experts
                )

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model // 2,)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)，增加 batch 维度
        pe.requires_grad = False
        self.register_buffer('pe', pe, persistent=True)

    def forward(self, x):
        # x 的形状为 (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]  # 取前 seq_len 个位置编码，广播到 batch_size
        return x


class AttentionProjection(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_model)
        self.w2 = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        return F.silu(self.w1(x)) * self.w2(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.proj_q = AttentionProjection(d_model)
        self.proj_k = AttentionProjection(d_model)
        self.proj_v = AttentionProjection(d_model)
        self.linear_out = nn.Linear(d_model, d_model)

    def attention(self, query, key, value, mask:torch.Tensor=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = torch.where(mask == 0, torch.tensor(-1e9, device=scores.device), scores)
        return torch.matmul(F.softmax(scores, dim=-1), value)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query = self.proj_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2).contiguous()
        key = self.proj_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2).contiguous()
        value = self.proj_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2).contiguous()
        x = self.attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.linear_out(x)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_feedforward):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_feedforward)
        self.w2 = nn.Linear(d_feedforward, d_model)
        self.w3 = nn.Linear(d_model, d_feedforward)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class SparseFeedForward(nn.Module):
    def __init__(self, d_model, d_feedforward, num_experts, top_k):
        super(SparseFeedForward, self).__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            FeedForward(d_model, d_feedforward) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-6)
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_k_indices[..., k]  # (batch_size, seq_len)
            probs = top_k_probs[..., k].unsqueeze(-1)  # (batch_size, seq_len, 1)
            for e in range(self.num_experts):
                mask = (expert_idx == e).float().unsqueeze(-1)  # (batch_size, seq_len, 1)
                if mask.sum() > 0:
                    expert_output = self.experts[e](x * mask)
                    output += expert_output * probs * mask
        return output


class EncoderLayer(nn.Module):
    def __init__(self, args: Config):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(args.encoder_d_model, args.encoder_attention_heads)
        self.feed_forward = SparseFeedForward(args.encoder_d_model, args.encoder_d_ff, args.encoder_num_experts,
                                              args.encoder_top_experts)
        self.norm1 = nn.LayerNorm(args.encoder_d_model)
        self.norm2 = nn.LayerNorm(args.encoder_d_model)

    def forward(self, x, mask):
        x2 = self.norm1(x)
        x = x + self.self_attn(x2, x2, x2, mask)
        x2 = self.norm2(x)
        x = x + self.feed_forward(x2)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, args: Config):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(args.encoder_d_model, args.encoder_attention_heads)
        self.cross_attn = MultiHeadAttention(args.encoder_d_model, args.encoder_attention_heads)
        self.feed_forward = SparseFeedForward(args.encoder_d_model, args.encoder_d_ff, args.encoder_num_experts,
                                              args.encoder_top_experts)
        self.norm1 = nn.LayerNorm(args.encoder_d_model)
        self.norm2 = nn.LayerNorm(args.encoder_d_model)
        self.norm3 = nn.LayerNorm(args.encoder_d_model)

    def forward(self, tgt, memory, mask):
        tgt2 = self.norm1(tgt)
        tgt = tgt + self.self_attn(tgt2, tgt2, tgt2, mask)
        tgt2 = self.norm2(tgt)
        tgt = tgt + self.cross_attn(tgt2, memory, memory, mask)
        tgt2 = self.norm3(tgt)
        tgt = tgt + self.feed_forward(tgt2)
        return tgt


class Encoder(nn.Module):
    def __init__(self, args:Config):
        super(Encoder, self).__init__()
        self.d_model = args.encoder_d_model
        self.embedding = nn.Linear(args.obs_dim, args.encoder_d_model)
        self.pos_encoder = PositionalEncoding(args.text_max_len, args.encoder_d_model)
        self.backbone = nn.ModuleList([EncoderLayer(args) for _ in range(args.backbone_layers)])
        self.backbone_norm = nn.LayerNorm(args.encoder_d_model)
        self.actor_head = EncoderLayer(args)
        self.critic_head = EncoderLayer(args)
        self.actor_critic_cross_attn_layers = args.actor_critic_cross_attn_layers
        self.actor = nn.ModuleList([DecoderLayer(args) for _ in range(args.actor_critic_cross_attn_layers)])
        self.critic = nn.ModuleList([DecoderLayer(args) for _ in range(args.actor_critic_cross_attn_layers)])
        self.actor_norm = nn.LayerNorm(args.encoder_d_model)
        self.actor_end = nn.Linear(args.encoder_d_model, args.act_dim)
        self.critic_norm = nn.LayerNorm(args.encoder_d_model)
        self.critic_end = nn.Linear(args.encoder_d_model, 1)

    def forward(self, src, mask):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        for layer in self.backbone:
            src = layer(src, mask)
        src = self.backbone_norm(src)
        actor = self.actor_head(src, mask)
        critic = self.critic_head(src, mask)
        for layer_idx in range(self.actor_critic_cross_attn_layers):
            actor_new = self.actor[layer_idx](actor, critic, mask)
            critic_new = self.critic[layer_idx](critic, actor, mask)
            actor = actor_new
            critic = critic_new
        actor = self.actor_norm(actor)
        actor = self.actor_end(actor)
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
    def __init__(self, id, device, render_mode=None, args:Config = Config()):
        self.id = str(id) + " " + device
        self.device = torch.device(device)
        self.timesteps_per_batch = args.timesteps_per_batch
        self.max_timesteps_per_episode = args.max_timesteps_per_episode
        self.gid = args.gid
        self.env = gym.make(self.gid, render_mode = render_mode) #"human" None
        self.observation, self.info = self.env.reset()
        self.reward = None
        self.terminated = False
        self.truncated = False
        self.act_dim = args.act_dim
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=args.cov_var_value)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)
        self.actor_critic = Transformer().to(self.device)
        for param in self.actor_critic.parameters():
            param.requires_grad = False
        self._init_observation_data()

    def rollout(self):
        self._init_observation_data()
        t = 0
        success = False
        print(f"环境id：{self.id}开始收集数据")
        start_time = time()
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
            ep_end_time = time()
            print(f"环境id：{self.id}：  第{len(self.batch_lens)}局游戏奖励为{np.sum(ep_rews)}，时间步为{ep_t + 1}，当前累计时间步{np.sum(self.batch_lens)}/{self.timesteps_per_batch}，用时{ep_end_time - ep_start_time}秒")
        end_time = time()

        print(f"环境id：{self.id}/运行时间: {end_time - start_time}秒：  平均奖励为{np.average(np.array([np.average(ep) for ep in self.batch_ep_rews]))}，当前累计时间步{np.sum(self.batch_lens)}/{self.timesteps_per_batch}")
        return self.batch_ep_obs, self.batch_ep_acts, self.batch_ep_log_probs, self.batch_ep_rews, self.batch_lens, self.batch_ep_vals, self.batch_ep_dones

    def _init_observation_data(self):
        self.batch_ep_obs = []
        self.batch_ep_acts = []
        self.batch_ep_log_probs = []
        self.batch_ep_rews = []
        self.batch_lens = []
        self.batch_ep_vals = []
        self.batch_ep_dones = []

    def _get_action(self, ep_obs, ep_t):
        with torch.no_grad():
            causal_mask = torch.tril(torch.ones((ep_t, ep_t))).int().to(self.device)  # [tgt_len, tgt_len]
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0) # [1, 1, tgt_len, tgt_len]
            ep_obs = torch.cat(ep_obs, dim=0).unsqueeze(0)
            mean, V = self.actor_critic(ep_obs, mask = causal_mask)
            dist = MultivariateNormal(mean[0][-1], self.cov_mat)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob, V[0][-1]

    def _set_agent(self, agent):
        for p1, p2 in zip(self.actor_critic.parameters(), agent.parameters()):
            p1.data = p2.data.clone().to(self.device)
        print(f"环境id：{self.id}：  模型已更新")


class PPO:
    def __init__(self, args:Config = Config()):
        self.gid = args.gid
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_updates_per_iteration = args.n_updates_per_iteration
        self.clip = args.clip
        self.rho_clip = args.rho_clip
        self.c_clip = args.c_clip
        self.ent_coef = args.ent_coef
        self.timesteps_needed = args.timesteps_all_batch
        self.timesteps_per_mini_batch = args.timesteps_per_mini_batch
        self.act_dim = args.act_dim
        self.obs_dim = args.obs_dim
        self.cov_var = torch.full(size=(args.act_dim,), fill_value=args.cov_var_value)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)
        self.actor_critic = Transformer().to(self.device)
        self.actor_critic_optim = torch.optim.NAdam(self.actor_critic.parameters(),lr=args.lr)
        self.max_grad_norm = args.max_grad_norm
        self.target_kl = args.target_kl
        self.break_kl = args.break_kl
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
        for sub_env in self.sub_envs:
            sub_env._set_agent(self.actor_critic)
        self.i = 0
    def multithread_learn(self, simulation_number):
        with (ThreadPoolExecutor(max_workers=len(self.sub_envs)) as threadpoolexecutor):
            futures = set()
            future_to_env = {}
            cur_timesteps = 0
            for sub_env in self.sub_envs:
                future = threadpoolexecutor.submit(sub_env.rollout)
                futures.add(future)
                future_to_env[future] = sub_env
            while futures and self.i < simulation_number:
                done, _ = wait(futures, return_when="FIRST_COMPLETED")
                for future in done:
                    futures.remove(future)
                    sub_env = future_to_env.pop(future)
                    cur_timesteps += np.sum(sub_env.batch_lens)
                    records = future.result()
                    self._load_data_to_store_data(records)
                    print(f"距离模型迭代{cur_timesteps}/{self.timesteps_needed}组数据")
                if cur_timesteps > self.timesteps_needed:
                    self.i += 1
                    self._compute_loss()
                    print(f"模型已迭代，此轮迭代学习的数据量为{cur_timesteps}组")
                    if (self.i + 1) % 10 == 0:
                        self._save()
                    cur_timesteps = 0
                if self.i < simulation_number:
                    sub_env._set_agent(self.actor_critic)
                    new_future = threadpoolexecutor.submit(sub_env.rollout)
                    futures.add(new_future)
                    future_to_env[new_future] = sub_env

    def _compute_loss(self):
        mini_batch_id = 0
        A_k_list = []
        mini_batch_rtgs_list = []
        while len(self.all_batch_lens):
            mini_batch_id += 1
            self._init_compute_data()
            self._prepare_compute_data()
            V, curr_log_probs, entropy = self._evaluate()
            if self.use_gae and not self.use_vtrace:
                A_k = self._calculate_gae()
                mini_batch_rtgs = A_k + V.detach()
            elif self.use_vtrace:
                A_k = self._calculate_vtrace(curr_log_probs)
                mini_batch_rtgs = A_k + V.detach()
            else:
                mini_batch_rtgs = self._compute_rtgs()
                A_k = mini_batch_rtgs - V.detach()

            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            mini_batch_rtgs_list.append(mini_batch_rtgs)
            A_k_list.append(A_k)
        self._load_backup_data_to_store_data()
        for j in range(self.n_updates_per_iteration):
            self.actor_critic_optim.zero_grad()
            mini_batch_id = 0
            mini_batch_idx = -1
            while len(self.all_batch_lens):
                mini_batch_id += 1
                mini_batch_idx += 1
                self._init_compute_data()
                self._prepare_compute_data(not (j==self.n_updates_per_iteration - 1))
                V, curr_log_probs, entropy = self._evaluate()
                entropy_loss = entropy.mean()
                logratios = curr_log_probs - self.mini_batch_log_probs
                ratios = torch.exp(logratios)
                approx_kl = ((ratios - 1) - logratios).mean()
                if approx_kl > self.break_kl:
                    continue
                if approx_kl > 1.5 * self.target_kl:
                    self.clip = max(self.clip / 1.5, 0.1)
                elif approx_kl < 0.5 * self.target_kl:
                    self.clip = min(self.clip * 1.5, 0.3)
                print(f"调整 clip={self.clip}，目前为第{self.i}轮的第{mini_batch_id}个mini_batch的第{j}轮训练，该mini_batch的样本数为{np.sum(self.mini_batch_lens)}，approx_kl为{approx_kl}")
                surr1 = ratios * A_k_list[mini_batch_idx]
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k_list[mini_batch_idx]
                actor_loss = (-torch.min(surr1, surr2)).mean()
                actor_loss = actor_loss - self.ent_coef * entropy_loss
                critic_loss = nn.MSELoss()(V, mini_batch_rtgs_list[mini_batch_idx])
                actor_loss.backward(retain_graph=True)
                critic_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.actor_critic_optim.step()
            self._load_backup_data_to_store_data()

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
                rho = torch.exp(ep_curr_log_probs[t] - ep_log_probs[t]).clamp(max=self.rho_clip)
                c = torch.exp(ep_curr_log_probs[t] - ep_log_probs[t]).clamp(max=self.c_clip)
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
            causal_mask = torch.tril(torch.ones((max_timesteps, max_timesteps))).int().to(self.device)  # [tgt_len, tgt_len]
            causal_mask = causal_mask.unsqueeze(0) # [1, tgt_len, tgt_len]
            key_mask = torch.zeros(max_timesteps).to(self.device)
            key_mask[:ep_len] = torch.ones(ep_len).to(self.device)
            key_mask = key_mask.int().unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len]
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
        checkpoint = {
            'policy_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.actor_critic_optim.state_dict()
        }
        torch.save(checkpoint, self.gid + '.pth')
        print("保存成功")

    def _load(self):
        try:
            checkpoint = torch.load(self.gid + '.pth')
            self.actor_critic.load_state_dict(checkpoint['policy_state_dict'])
            self.actor_critic_optim.load_state_dict(checkpoint['optimizer_state_dict'])
            print("加载成功")
        except FileNotFoundError:
            pass

test = PPO()
test.multithread_learn(1000000)


