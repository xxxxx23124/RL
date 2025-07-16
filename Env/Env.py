import numpy as np
from pettingzoo.classic import chess_v6
from ANN.Networks.TimeSpaceChessModel import TimeSpaceChessModel

class BasicAgent:


class TwoPlayerChessEnv:
    """
    一个封装了 PettingZoo aec_api 的国际象棋环境类，
    使其接口更像一个接受两个智能体动作并返回完整回合(rollout)数据的环境。

    - 智能体 'player_0' 代表白方 (White)
    - 智能体 'player_1' 代表黑方 (Black)
    """

    def __init__(self, render_mode=None):
        """
        初始化环境。
        :param render_mode: PettingZoo环境的渲染模式, 如 "human"。
        """
        self._env = chess_v6.env(render_mode=render_mode)
        self.agent_map = {'player_0': 'white', 'player_1': 'black'}
        self.current_agent_name = None
        self.done = True

    def reset(self, seed=None):
        """
        重置环境，并返回白方的初始观察。
        """
        self._env.reset(seed=seed)
        self.done = False

        # 获取第一个智能体（白方）的初始观察
        observation, _, _, _, _ = self._env.last()
        self.current_agent_name = self._env.agent_selection

        # 返回给白方智能体
        return observation

    def play_step(self, white_action, black_action):
        """
        执行一个完整的游戏回合（白方走一步，然后黑方走一步）。

        :param white_action: 白方智能体选择的动作。
        :param black_action: 黑方智能体选择的动作。
        :return: (white_rollout, black_rollout)，如果游戏结束则 rollout 为 None。
                 每个 rollout 的格式是 (observation, action, reward, done, next_observation)。
        """
        if self.done:
            raise RuntimeError("游戏已结束，请先调用 reset() 方法。")

        # --- 白方回合 ---
        if self.current_agent_name != self._env.agents[0]:  # 确保当前是白方在行动
            raise RuntimeError(f"逻辑错误: 期望 {self._env.agents[0]} 行动, 但当前是 {self.current_agent_name}")

        white_obs, _, _, _, _ = self._env.last()
        self._env.step(white_action)

        # 获取白方行动后的状态
        white_reward, white_done, white_trunc, _ = self._env.last()
        black_obs_as_next = self._env.last()[0]  # 黑方的观察就是白方的下一观察
        self.done = white_done or white_trunc

        white_rollout = (white_obs, white_action, white_reward, self.done, black_obs_as_next)

        # 如果白方走完后游戏结束，黑方没有机会行动
        if self.done:
            black_reward = self._env.rewards[self._env.agents[1]]  # 获取黑方的最终奖励
            # 对于黑方来说，它的回合没能开始就结束了
            black_rollout = (black_obs_as_next, None, black_reward, self.done, None)
            return white_rollout, black_rollout

        # --- 黑方回合 ---
        self.current_agent_name = self._env.agent_selection
        black_obs, _, _, _, _ = self._env.last()  # 这应该和上面的 black_obs_as_next 一致
        self._env.step(black_action)

        # 获取黑方行动后的状态
        black_reward, black_done, black_trunc, _ = self._env.last()
        white_obs_as_next = self._env.last()[0]  # 白方的观察就是黑方的下一观察
        self.done = black_done or black_trunc

        black_rollout = (black_obs, black_action, black_reward, self.done, white_obs_as_next)

        # 如果黑方走完后游戏结束，需要更新白方的最终奖励
        if self.done:
            white_reward = self._env.rewards[self._env.agents[0]]
            white_rollout = (white_obs, white_action, white_reward, self.done, black_obs_as_next)

        self.current_agent_name = self._env.agent_selection
        return white_rollout, black_rollout

    def close(self):
        """关闭环境。"""
        self._env.close()

    @property
    def action_space(self):
        """为当前玩家提供动作空间。"""
        return self._env.action_space(self.current_agent_name)


# --- 使用示例 ---

# 假设我们有两个简单的随机策略智能体
def random_policy(observation, action_space):
    """一个简单的随机策略。"""
    mask = observation["action_mask"]
    # 如果没有可用动作（例如游戏结束），返回 None
    if not np.any(mask):
        return None
    return action_space.sample(mask)


# 1. 初始化环境
# env = TwoPlayerChessEnv(render_mode="human") # 如果需要看棋盘
env = TwoPlayerChessEnv()

# 2. 重置环境，获取白方的初始观察
white_observation = env.reset(seed=42)

# 3. 游戏主循环
for i in range(25):  # 最多玩 25 个回合
    print(f"\n--- 回合 {i + 1} ---")

    # 获取白方动作
    white_action_space = env.action_space
    white_action = random_policy(white_observation, white_action_space)

    # 获取黑方动作 (需要先拿到黑方的观察)
    # 在这个封装中，黑方的观察是白方行动后的结果，我们可以在 play_step 返回值中获取
    # 但为了决定黑方动作，我们需要一个临时的观察。这里我们假设可以提前知道，
    # 在实际的 RL 训练中，我们会先执行白方动作，得到黑方观察，再决定黑方动作。
    # 为了简化这个示例，我们假设黑方动作可以和白方同时给出。
    # 从 `play_step` 的返回值 `black_rollout` 中可以获得黑方的真实观察。

    # 一个简单的做法是：先假设一个黑方动作，实际在 play_step 中再获取真实观察。
    # 这里我们用白方的观察来为黑方做一个临时的随机决策，这不影响逻辑，因为最终 play_step 会使用正确的观察。
    temp_black_action_space = env.action_space  # 注意：此时 action_space 仍是白方的
    temp_black_obs_for_policy = white_observation  # 这是一个不精确的假设
    black_action = random_policy(temp_black_obs_for_policy, temp_black_action_space)

    try:
        # 4. 执行一个完整回合
        white_rollout, black_rollout = env.play_step(white_action, black_action)

        # 解包并打印 rollouts
        w_obs, w_act, w_rew, w_done, w_next_obs = white_rollout
        b_obs, b_act, b_rew, b_done, b_next_obs = black_rollout

        print(f"白方: 奖励={w_rew}, 游戏结束={w_done}")
        print(f"黑方: 奖励={b_rew}, 游戏结束={b_done}")

        # 为下一次循环准备白方的观察
        white_observation = w_next_obs

        if env.done:
            print("\n游戏结束!")
            break

    except RuntimeError as e:
        print(e)
        break

# 5. 关闭环境
env.close()