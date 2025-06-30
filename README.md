# PPO-Transformer-for-LunarLander

一个将 **PPO** 与 **Transformer + SSM(LRNN)** 结合，用来解决 `LunarLanderContinuous-v2` 的强化学习项目。  
代码完全收敛在单文件（`ppo.py`），开箱即用。

---

## 核心特性

### 1. Transformer-SSM 决策网络  
| 组件 | 说明 |
| ---- | ---- |
| 主干 | 多层 **RoPE + Multi-Head Attention** 与 **LRNN-StatefulFFN**（Mamba 风格 SSM），捕捉长时序依赖并节省显存。 |
| Actor / Critic | 共享主干后分叉，各自再堆若干 EncoderLayer 输出 **动作均值** 与 **状态价值**。 |
| KV-Cache & h_prev | 推理时逐步缓存注意力与 SSM 状态，采样效率高。 |

### 2. 强化学习算法  
* **PPO-Clip**，支持 `target_kl` 提前停止。  
* 优势函数三选一，优先级：`V-trace` › `GAE` › `Reward-to-Go`（默认启用 V-trace）。  
* **NAdam** 优化，梯度裁剪、熵正则。  
* **多线程采样**：配置 `max_cpu_threads / max_gpu_threads` 即可在 CPU/GPU 混合并行收集。  
* 采样阶段自动把 **优化器参数搬到 CPU**，训练前再搬回 GPU，极大节省显存。  
* 定时自动 **checkpoint**（含优化器、课程进度），崩溃后无缝恢复。

### 3. 环境与课程学习  
| 功能 | 细节 |
| ---- | ---- |
| POMDP 包装 | `LunarLanderPOMDPWrapper`：以概率 `p_flicker` 将 **6 个关键观测维度随机归零**。 |
| 自动课程 | `p_flicker` 由 Beta 分布采样；当最近 `N` 条轨迹平均奖励 > 阈值时自动提高均值，循序渐进。 |
| 手动模式 | 关闭 `use_automated_curriculum` 即固定 `p_flicker`。 |

---

## 快速开始

```bash
pip install gymnasium gymnasium[box2d] torch numpy
python ppo.py          # 默认单 GPU 训练  
```

常用可调参数（位于 `Config` 类）：  

| 参数 | 功能 |
| ---- | ---- |
| `max_cpu_threads / max_gpu_threads` | 并行环境数量 |
| `use_automated_curriculum` | 开/关课程学习 |
| `encoder_d_model`, `backbone_layers` | Transformer 规模 |
| `timesteps_per_batch`, `timesteps_all_batch` | 采样 / 更新频率 |

---

## TODO

* Web UI 实时监控与调参  
* 实现ViT (Vision Transformer)
* 引入 **PPO-RND** 等稀疏奖励探索策略
