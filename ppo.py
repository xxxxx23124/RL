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
class TransformerConfig:
    in_dim: int
    action_dim: int
    text_max_len: int = 5000
    backbone_layers: int = 3
    actor_critic_cross_attn_layers: int = 3
    encoder_d_model: int = 256
    encoder_d_ff: int = 512
    encoder_attention_heads: int = 4
    encoder_num_experts: int = 2
    encoder_top_experts: int = 1
    def __post_init__(self):
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


class MultiHeadLinearAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadLinearAttention, self).__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.proj_q = AttentionProjection(d_model)
        self.proj_k = AttentionProjection(d_model)
        self.proj_v = AttentionProjection(d_model)
        self.linear_out = nn.Linear(d_model, d_model)

    def linear_attention(self, query, key, value, eps=1e-6):
        # query shape: (batch_size, num_heads, seq_len_q, d_k)
        # key shape:   (batch_size, num_heads, seq_len_kv, d_k)
        # value shape: (batch_size, num_heads, seq_len_kv, d_k)

        phi_query = F.elu(query) + 1  # Shape: (batch_size, num_heads, seq_len_q, d_k)
        phi_key = F.elu(key) + 1  # Shape: (batch_size, num_heads, seq_len_kv, d_k)

        key_value = torch.matmul(phi_key.transpose(-2, -1), value)  # (batch_size, num_heads, d_k, d_k)
        key_sum_vector = phi_key.sum(dim=-2)  # (batch_size, num_heads, d_k)

        q_k_sum = torch.matmul(phi_query, key_sum_vector.unsqueeze(-1)).squeeze(-1)  # (batch_size, num_heads, seq_len_q)
        z_inv = 1.0 / (q_k_sum.unsqueeze(-1) + eps)  # (batch_size, num_heads, seq_len_q, 1)

        numerator = torch.matmul(phi_query, key_value)  # (batch_size, num_heads, seq_len_q, d_k)

        attn_output = numerator * z_inv  # (batch_size, num_heads, seq_len_q, d_k)
        return attn_output

    def forward(self, query, key, value):
        batch_size = query.size(0)

        query = self.proj_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2).contiguous()
        key = self.proj_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2).contiguous()
        value = self.proj_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2).contiguous()

        x = self.linear_attention(query, key, value)

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
    def __init__(self, args: TransformerConfig):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadLinearAttention(args.encoder_d_model, args.encoder_attention_heads)
        self.feed_forward = SparseFeedForward(args.encoder_d_model, args.encoder_d_ff, args.encoder_num_experts,
                                              args.encoder_top_experts)
        self.norm1 = nn.LayerNorm(args.encoder_d_model)
        self.norm2 = nn.LayerNorm(args.encoder_d_model)

    def forward(self, x):
        x2 = self.norm1(x)
        x = x + self.self_attn(x2, x2, x2)
        x2 = self.norm2(x)
        x = x + self.feed_forward(x2)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, args: TransformerConfig):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadLinearAttention(args.encoder_d_model, args.encoder_attention_heads)
        self.cross_attn = MultiHeadLinearAttention(args.encoder_d_model, args.encoder_attention_heads)
        self.feed_forward = SparseFeedForward(args.encoder_d_model, args.encoder_d_ff, args.encoder_num_experts,
                                              args.encoder_top_experts)
        self.norm1 = nn.LayerNorm(args.encoder_d_model)
        self.norm2 = nn.LayerNorm(args.encoder_d_model)
        self.norm3 = nn.LayerNorm(args.encoder_d_model)

    def forward(self, tgt, memory):
        tgt2 = self.norm1(tgt)
        tgt = tgt + self.self_attn(tgt2, tgt2, tgt2)
        tgt2 = self.norm2(tgt)
        tgt = tgt + self.cross_attn(tgt2, memory, memory)
        tgt2 = self.norm3(tgt)
        tgt = tgt + self.feed_forward(tgt2)
        return tgt


class Encoder(nn.Module):
    def __init__(self, args: TransformerConfig):
        super(Encoder, self).__init__()
        self.d_model = args.encoder_d_model
        self.embedding = nn.Linear(args.in_dim, args.encoder_d_model)
        self.pos_encoder = PositionalEncoding(args.text_max_len, args.encoder_d_model)
        self.backbone = nn.ModuleList([EncoderLayer(args) for _ in range(args.backbone_layers)])
        self.backbone_norm = nn.LayerNorm(args.encoder_d_model)
        self.actor_head = EncoderLayer(args)
        self.critic_head = EncoderLayer(args)
        self.actor_critic_cross_attn_layers = args.actor_critic_cross_attn_layers
        self.actor = nn.ModuleList([DecoderLayer(args) for _ in range(args.actor_critic_cross_attn_layers)])
        self.critic = nn.ModuleList([DecoderLayer(args) for _ in range(args.actor_critic_cross_attn_layers)])
        self.actor_norm = nn.LayerNorm(args.encoder_d_model)
        self.actor_end = nn.Linear(args.encoder_d_model, args.action_dim)
        self.critic_norm = nn.LayerNorm(args.encoder_d_model)
        self.critic_end = nn.Linear(args.encoder_d_model, 1)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        for layer in self.backbone:
            src = layer(src)
        src = self.backbone_norm(src)
        actor = self.actor_head(src)
        critic = self.critic_head(src)
        for layer_idx in range(self.actor_critic_cross_attn_layers):
            actor_new = self.actor[layer_idx](actor, critic)
            critic_new = self.critic[layer_idx](critic, actor)
            actor = actor_new
            critic = critic_new
        actor = self.actor_norm(actor)
        actor = self.actor_end(actor)
        critic = self.critic_norm(critic)
        critic = self.critic_end(critic)
        return actor, critic


class Transformer(nn.Module):
    def __init__(self, in_dim, action_dim):
        super(Transformer, self).__init__()
        self.args = TransformerConfig(in_dim=in_dim, action_dim=action_dim)
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

    def forward(self, src):
        actor, critic = self.encoder(src)
        return actor, critic


class Environment:
    def __init__(self, id, gid, render_mode = None, device="cpu", timesteps_per_batch=8000):
        self.id = str(id) + " " + device
        self.device = torch.device(device)
        self.timesteps_per_batch = timesteps_per_batch
        self.max_timesteps_per_episode = 2000
        self.gid = gid

        self.env = gym.make(gid, render_mode = render_mode) #"human" None
        self.observation, self.info = self.env.reset()
        self.reward = None
        self.terminated = False
        self.truncated = False
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.cov_var_value = 0.5
        self.cov_var_value_min = 0.3
        self.cov_var_value_max = 1.0
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=self.cov_var_value)
        self.cov_mat = torch.diag(self.cov_var).to(self.device)
        self.actor_critic = Transformer(self.obs_dim, self.act_dim).to(self.device)
        self.max_success_episode_number = 4

        self._init_observation_data()
        self._init_success_data()

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
                action, log_prob, V = self._get_action(ep_obs)
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
            if np.sum(ep_rews) > 0:
                success = True
                self.batch_success_obs.append(torch.cat(ep_obs, dim=0))
                self.batch_success_acts.append(torch.cat(ep_acts, dim=0))
                self.batch_success_log_probs.append(torch.cat(ep_log_probs, dim=0))
                self.batch_success_vals.append(ep_vals)
                self.batch_success_rews.append(ep_rews)
                self.batch_success_dones.append(ep_dones)
                self.batch_success_lens.append(ep_t)
                self._pop_success_data()
                self.cov_var_value = max(self.cov_var_value - 0.01, self.cov_var_value_min)
                self.cov_var = torch.full(size=(self.act_dim,), fill_value=self.cov_var_value)
                self.cov_mat = torch.diag(self.cov_var).to(self.device)
        if not success:
            print(f"环境id：{self.id}：  当前未成功，添加过往的成功过程，数量为{len(self.batch_success_lens)}")
            for ep_obs, ep_acts, ep_log_probs, ep_rews, ep_vals, ep_dones, success_lens in zip(self.batch_success_obs, self.batch_success_acts, self.batch_success_log_probs,
                                                                                 self.batch_success_rews, self.batch_success_vals, self.batch_success_dones, self.batch_success_lens):
                self.batch_ep_obs.append(ep_obs)
                self.batch_ep_acts.append(ep_acts)
                self.batch_ep_log_probs.append(ep_log_probs)
                self.batch_ep_rews.append(ep_rews)
                self.batch_ep_vals.append(ep_vals)
                self.batch_ep_dones.append(ep_dones)
                self.batch_lens.append(success_lens)
            self.cov_var_value = min(self.cov_var_value + 0.01, self.cov_var_value_max)
            self.cov_var = torch.full(size=(self.act_dim,), fill_value=self.cov_var_value)
            self.cov_mat = torch.diag(self.cov_var).to(self.device)
        end_time = time()

        print(f"环境id：{self.id}/运行时间: {end_time - start_time}秒：  平均奖励为{np.average(np.array([np.average(ep) for ep in self.batch_ep_rews]))}，当前累计时间步{np.sum(self.batch_lens)}/{self.timesteps_per_batch}")
        return self.batch_ep_obs, self.batch_ep_acts, self.batch_ep_log_probs, self.batch_ep_rews, self.batch_lens, self.batch_ep_vals, self.batch_ep_dones, self.cov_mat


    def _init_observation_data(self):
        self.batch_ep_obs = []
        self.batch_ep_acts = []
        self.batch_ep_log_probs = []
        self.batch_ep_rews = []
        self.batch_lens = []
        self.batch_ep_vals = []
        self.batch_ep_dones = []

    def _pop_success_data(self):
        cur = len(self.batch_success_lens)
        while len(self.batch_success_lens) > self.max_success_episode_number:
            self.batch_success_obs.pop()
            self.batch_success_acts.pop()
            self.batch_success_log_probs.pop()
            self.batch_success_vals.pop()
            self.batch_success_rews.pop()
            self.batch_success_dones.pop()
            self.batch_success_lens.pop()
        print(f"环境id：{self.id}：  执行删除，删除了{cur - len(self.batch_success_lens)}，当前成功过程数目{len(self.batch_success_lens)}")

    def _init_success_data(self):
        self.batch_success_obs = []
        self.batch_success_acts = []
        self.batch_success_log_probs = []
        self.batch_success_vals = []
        self.batch_success_rews = []
        self.batch_success_dones = []
        self.batch_success_lens = []

    def _get_action(self, ep_obs):
        with torch.no_grad():
            ep_obs = torch.cat(ep_obs, dim=0).unsqueeze(0)
            mean, V = self.actor_critic(ep_obs)
            dist = MultivariateNormal(mean[0][-1], self.cov_mat)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob, V[0][-1]

    def _set_agent(self, agent):
        self.actor_critic = deepcopy(agent).to(self.device)
        print(f"环境id：{self.id}：  模型已更新")

class PPO:
    def __init__(self, gid="LunarLanderContinuous-v2", render_mode = None):
        self.gid = gid
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_updates_per_iteration = 2
        self.clip = 0.2
        self.rho_clip = 1.0
        self.c_clip = 1.0
        self.ent_coef = 0.01
        env = gym.make(gid, render_mode = render_mode) #"human" None
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.actor_critic = Transformer(obs_dim, act_dim).to(self.device)
        self.actor_critic_optim = torch.optim.NAdam(self.actor_critic.parameters(),lr=5e-5)
        self.max_grad_norm = 0.5
        self.use_gae = True
        self.use_vtrace = True
        self.target_kl = 0.7
        self.approx_kl_break = 4.0
        self.gamma = 0.99
        self.lam = 0.99
        self.timesteps_per_mini_batch = 2400
        self._load()
        self._init_compute_data()
        self.sub_envs = []
        self.max_cpu_threads = 0
        for i in range(self.max_cpu_threads):
            self.sub_envs.append(Environment(i + 1, gid=self.gid,device="cpu", timesteps_per_batch=4800))
        self.max_gpu_threads = 2
        for i in range(self.max_gpu_threads):
            self.sub_envs.append(Environment(i + 1, gid=self.gid, device="cuda", timesteps_per_batch=4800))
        for sub_env in self.sub_envs:
            sub_env._set_agent(self.actor_critic)
        #self.sub_envs.append(Environment(0, gid=self.gid, render_mode="human", device="cuda", timesteps_per_batch=4800))
        self.i = 0
        self._init_store_data()

    def multithread_learn(self, simulation_number):
        with (ThreadPoolExecutor(max_workers=len(self.sub_envs)) as threadpoolexecutor):
            futures = set()
            future_to_env = {}
            for sub_env in self.sub_envs:
                future = threadpoolexecutor.submit(sub_env.rollout)
                futures.add(future)
                future_to_env[future] = sub_env
            while futures and self.i < simulation_number:
                done, _ = wait(futures, return_when="FIRST_COMPLETED")
                for future in done:
                    self.i += 1
                    futures.remove(future)
                    sub_env = future_to_env.pop(future)
                    batch_ep_obs, batch_ep_acts, batch_ep_log_probs, batch_ep_rews, batch_lens, batch_ep_vals, batch_ep_dones, cov_mat = future.result()
                    cov_mat = cov_mat.to(self.device)
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

                print(f"开始训练，当前训练数据一共{np.sum(self.all_batch_lens)}组")
                self._learn(cov_mat)
                if self.i < simulation_number:
                    sub_env._set_agent(self.actor_critic)
                    new_future = threadpoolexecutor.submit(sub_env.rollout)
                    futures.add(new_future)
                    future_to_env[new_future] = sub_env

                if (self.i + 1) % 30 == 0:
                    self._save()

    def _learn(self, cov_mat):
        mini_batch_id = 0
        while len(self.all_batch_lens):
            mini_batch_id += 1
            self._init_compute_data()
            self._prepare_compute_data()
            V, curr_log_probs, entropy = self._evaluate(cov_mat)
            if self.use_gae and not self.use_vtrace:
                self._calculate_gae()
                A_k = self.batch_gae
                self.batch_rtgs = A_k + V.detach()
            elif self.use_vtrace:
                self._calculate_vtrace(curr_log_probs)
                A_k = self.batch_vtrace
                self.batch_rtgs = A_k + V.detach()
            else:
                self._compute_rtgs()
                A_k = self.batch_rtgs - V.detach()

            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for j in range(self.n_updates_per_iteration):
                V, curr_log_probs, entropy = self._evaluate(cov_mat)
                entropy_loss = entropy.mean()
                logratios = curr_log_probs - self.mini_batch_log_probs
                ratios = torch.exp(logratios)
                approx_kl = ((ratios - 1) - logratios).mean()
                if approx_kl > self.approx_kl_break:
                    break
                if approx_kl > 1.5 * self.target_kl:
                    self.clip = max(self.clip / 1.5, 0.1)
                elif approx_kl < 0.5 * self.target_kl:
                    self.clip = min(self.clip * 1.5, 0.3)
                print(
                    f"调整 clip={self.clip}，目前为第{self.i}轮的第{mini_batch_id}个mini_batch的第{j}轮训练，该mini_batch的样本数为{np.sum(self.mini_batch_lens)}，approx_kl为{approx_kl}")
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                actor_loss = (-torch.min(surr1, surr2)).mean()
                actor_loss = actor_loss - self.ent_coef * entropy_loss
                critic_loss = nn.MSELoss()(V, self.batch_rtgs)
                self.actor_critic_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.actor_critic_optim.step()

    def _evaluate(self, cov_mat):
        mean, V = self.actor_critic(self.mini_batch_obs)
        mean = mean.squeeze()
        V = V.squeeze()
        dist = MultivariateNormal(mean, cov_mat)
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
        self.batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(self.device)

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

        self.batch_gae = torch.tensor(batch_advantages, dtype=torch.float).to(self.device)

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
        self.batch_vtrace = torch.tensor(batch_advantages, dtype=torch.float).to(self.device)

    def _prepare_compute_data(self):
        current_timesteps = 0
        while current_timesteps < self.timesteps_per_mini_batch and len(self.all_batch_lens):
            self.mini_batch_obs.append(self.all_batch_obs.pop())
            self.mini_batch_acts.append(self.all_batch_acts.pop())
            self.mini_batch_log_probs.append(self.all_batch_log_probs.pop())
            self.mini_batch_vals.append(self.all_batch_vals.pop())
            self.mini_batch_rews.append(self.all_batch_rews.pop())
            self.mini_batch_dones.append(self.all_batch_dones.pop())
            self.mini_batch_lens.append(self.all_batch_lens.pop())
            current_timesteps += self.mini_batch_lens[-1]

        self.mini_batch_obs = torch.cat(self.mini_batch_obs, dim=0).unsqueeze(0)
        self.mini_batch_acts = torch.cat(self.mini_batch_acts, dim=0)
        self.mini_batch_log_probs = torch.cat(self.mini_batch_log_probs, dim=0)

    def _init_compute_data(self):
        self.mini_batch_obs = []
        self.mini_batch_acts = []
        self.mini_batch_log_probs = []
        self.mini_batch_vals = []
        self.mini_batch_rews = []
        self.mini_batch_dones = []
        self.mini_batch_lens = []

        self.batch_vtrace = []
        self.batch_gae = []
        self.batch_rtgs = []

    def _init_store_data(self):
        self.all_batch_obs = []
        self.all_batch_acts = []
        self.all_batch_log_probs = []
        self.all_batch_vals = []
        self.all_batch_rews = []
        self.all_batch_dones = []
        self.all_batch_lens = []

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


