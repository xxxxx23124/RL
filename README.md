## PPO-Transformer-for-LunarLander

这是一个为解决 `LunarLanderContinuous-v2` 环境而构建的强化学习项目，它将 PPO (Proximal Policy Optimization) 算法与 Transformer 架构相结合。代码实现于单个文件中，以保持结构清晰。

### 项目设计与核心功能

#### 1. 模型架构：基于 Transformer 的决策网络

传统的强化学习智能体通常仅依赖当前时刻的观测来做决策。本项目引入 Transformer，旨在让智能体能够理解状态在时间序列上的**上下文关系**。

*   **设计动机**: Transformer 的自注意力机制擅长捕捉序列内的长距离依赖。通过将一次完整的飞行轨迹（episode）视为一个序列，模型不仅能看到当前状态，还能回顾历史状态的变化趋势，从而可能制定出更具远见的策略。
*   **实现细节**:
    *   模型主体采用 Transformer Encoder 架构。
    *   为了使模型理解时间顺序，集成了**旋转位置编码 (RoPE)**。
    *   **Actor (策略网络)** 和 **Critic (价值网络)** 共享一个主干网络，再分别通过各自的 Transformer 层进行细化，最终输出动作和状态价值。

#### 2. 算法实现：功能增强的 PPO

本项目实现了 PPO 算法，并包含以下关键特性：

*   **高级优势函数计算**: 支持使用 **GAE (Generalized Advantage Estimation)** 或 **V-trace** 来计算优势函数，这两种方法旨在优化学习过程中的偏差与方差平衡。
*   **并行数据采样**: 支持配置多个 CPU 或 GPU 线程进行异步数据采样，以提高训练数据收集的效率。主模型在收集到足够数据后进行统一更新。
*   **KL 散度早停**: 为保证策略更新的稳定性，引入了基于 `target_kl` 的早停机制。若单次更新中策略变化过大（近似KL散度超过阈值），则提前中止该轮更新，防止策略崩溃。

#### 3. 环境设定：部分可观测性与课程学习

为了模拟真实世界中信息不完整的情形，本项目将标准环境改造为一个部分可观测马尔可夫决策过程 (POMDP)。

*   **`LunarLanderPOMDPWrapper`**: 该环境包装器会以一定概率 `p_flicker` 随机“遮挡”部分观测信息（如位置、速度等），将其置零，从而增加任务难度。
*   **自动化课程学习 (Automated Curriculum Learning)**: 为了让智能体能逐步适应这种信息缺失的环境，引入了课程学习机制：
    1.  **初始阶段**: 环境接近完全可观测 (`p_flicker` 概率很低)，智能体学习基础控制。
    2.  **难度提升**: 当智能体在当前难度下的平均奖励达到预设阈值 `curriculum_reward_threshold` 时，系统会自动增加 `p_flicker` 的概率。
    3.  **渐进式学习**: 智能体从简单任务开始，逐步过渡到信息稀疏的复杂场景，稳健地掌握着陆技能。

### 如何运行

1.  安装 `gymnasium`, `torch`, `numpy` 及 `box2d` 依赖。
    ```bash
    pip install gymnasium torch numpy
    pip install gymnasium[box2d]
    ```
2.  将代码保存为 `ppo.py` 并直接运行。
    ```bash
    python ppo.py
    ```
3.  所有超参数均可在代码文件顶部的 `Config` 类中进行调整，例如：
    *   `max_cpu_threads`, `max_gpu_threads`: 并行采样线程数。
    *   `use_automated_curriculum`: 启用或禁用课程学习。
    *   `backbone_layers`, `encoder_d_model`: Transformer 模型相关参数。

### 未来探索方向

*   构建一个简单的 Web 交互界面，用于实时监控训练状态和动态调整配置。
*   探索更前沿的算法与模型，例如将 Transformer 替换为基于状态空间模型（如 Mamba）的架构，以处理需要更长记忆的复杂游戏环境。
*   实现 PPO-RND (Random Network Distillation) 等探索算法，以应对奖励稀疏的问题。
