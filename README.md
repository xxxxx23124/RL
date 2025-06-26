# PPO-Transformer-for-LunarLander

你好！这是一个我为解决 `LunarLanderContinuous-v2` 环境而编写的强化学习项目。它将经典的 **PPO (Proximal Policy Optimization)** 算法与 **Transformer** 架构相结合。

作为一名强化学习的初学者，我从 [蘑菇书 EasyRL](https://github.com/datawhalechina/easy-rl) 和 [PPO for Beginners](https://github.com/ericyangyu/PPO-for-Beginners) 等优秀资源中获益良多。这个项目是我学习和实践的产物，代码都在单个文件 `ppo.py` 中，希望能保持简洁。

## 项目亮点与设计思路

### 1. 模型选择：Transformer

在强化学习中，智能体通常根据**当前**的观测（observation）来做出决策。对于像 `LunarLander` 这样的任务，虽然当前观测已经包含了位置、速度等信息，但智能体如果能理解这些状态在**一段时间内的变化趋势**（即上下文），或许能制定出更优的长期策略。

- **为什么选择 Transformer？**
  Transformer 的核心优势在于其自注意力（Self-Attention）机制，它能捕捉序列中各个时间步之间的依赖关系。我希望借助这种能力，让智能体不仅看到当前帧，还能回顾整个轨迹（episode），理解动作与结果之间的长期因果联系。这就像一个玩家在回顾整局游戏录像来学习，而不是只盯着眼前的瞬间。
- **具体实现**
  - 模型使用了一个基于 Transformer Encoder 的架构。
  - 为了让模型理解时间顺序，我引入了 **旋转位置编码 (RoPE)**，这是一种高效的位置编码方法。
  - 演员（Actor）和评论家（Critic）共享一个共同的主干网络（Backbone），之后分别接入各自的 Transformer 层，以学习策略和价值函数。

### 2. 算法实现：PPO (近端策略优化)

我实现了一个功能相对完善的 PPO 算法，并集成了以下特性：

- **GAE & V-trace**：支持使用 GAE (Generalized Advantage Estimation) 或 V-trace 来计算优势函数，这两种方法都能在偏差和方差之间取得更好的平衡。
- **多线程异步采样**：代码支持利用多个 CPU 或 GPU 线程并行与环境交互、收集数据，提升了采样效率(实际测试不是很理想，如果有多块gpu并且cpu与多块gpu通信很快的可以提速，否则瓶颈就是cpu与gpu的通信速度，需要修改代码手动分配到不同gpu上)。当收集到足够多的数据后，主模型会进行统一的更新。
- **KL 散度早停**：为了防止模型更新步子太大导致策略崩溃，并维持多线程环境下子模型策略的一致性，我设置了基于 `target_kl` 的早停机制。当一轮更新中近似 KL 散度超过阈值，该轮更新就会提前终止。

### 3. 环境挑战：部分可观测性 (POMDP) 与课程学习

为了让任务更具挑战性并更好地模拟真实世界中的信息缺失，我将 `LunarLander` 包装成了一个部分可观测马尔可夫决策过程 (POMDP) 环境。

- **`LunarLanderPOMDPWrapper`**：这个包装器会以一定的概率 `p_flicker` 随机“遮挡”掉观测中的某些信息（如位置、速度等），将其置为 0。
- **自动化课程学习 (Automated Curriculum Learning)**：从零开始就面对一个信息大量缺失的环境是非常困难的。因此，我引入了课程学习机制：
  1.  **初始阶段**：`p_flicker` (信息遮挡概率) 非常低，环境接近完全可观测，智能体可以轻松学习基础的降落技巧。
  2.  **评估与提升**：当智能体在当前难度下表现良好时（平均奖励超过预设阈值 `curriculum_reward_threshold`），系统会自动提升难度。
  3.  **逐步增加难度**：难度提升方式为逐渐增加 `p_flicker` 的均值，让环境中的信息变得越来越稀疏。
  
  这种机制让智能体像学生一样，从易到难，循序渐进地掌握在复杂环境中完成任务的能力。

## 如何运行

1.  确保你已经安装了必要的库，例如 `gymnasium`, `torch`, `numpy`。
    ```bash
    pip install gymnasium torch numpy
    pip install gymnasium[box2d] # 安装 LunarLander 环境依赖
    ```
2.  将代码保存为 `ppo.py`。
3.  直接运行脚本即可开始训练：
    ```bash
    python ppo.py
    ```
4.  你可以直接在 `ppo.py` 文件顶部的 `Config` 类中调整超参数，例如：
    -   `max_cpu_threads`, `max_gpu_threads`：设置并行采样的线程数。
    -   `use_automated_curriculum`：启用或禁用课程学习。
    -   模型参数如 `backbone_layers`, `encoder_d_model` 等。
    -   默认配置的模型非常庞大（约8亿参数），如有需要请修改配置。
  
## 一些想法

我很好奇，当一个具备上下文理解能力的模型（如 Transformer）在一个足够复杂的、包含丰富逻辑关系的环境中训练时，是否会像大型语言模型一样，涌现出更高级的策略和对环境背后物理规律的“理解”。这个项目算是一个小小的尝试。  
  
未来可能会为这个项目加一个小交互界面，使用内置服务器让后通过网页访问，可以动态修改一些配置或者看训练状态，未来可能会去实现PPO-RND，或者使用manba2+ttt的结合体代替transformer模型，实现模型可以玩更复杂，需要更多时间步才能完成的游戏。  
  
欢迎任何形式的交流和建议！  
