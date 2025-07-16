import torch
from pettingzoo.classic import chess_v6
from Agent.Agent import ChessAgent

class SelfPlayRunner(ChessAgent):
    """
    一个封装了 PettingZoo aec_api 的国际象棋环境类，
    使其接口更像一个接受两个智能体动作并返回完整回合(rollout)数据的环境。

    - 智能体 'player_0' 代表白方 (White)
    - 智能体 'player_1' 代表黑方 (Black)
    """
    def __init__(self, device: torch.device, render_mode: str | None =None):
        self._env = chess_v6.env(render_mode=render_mode)
        # 获取任意一个智能体的观察空间
        # 对于 chess_v6，所有智能体的观察空间都是一样的
        # env.agents[0] 是 'player_0'
        agent = self._env.agents[0]
        obs_space = self._env.observation_space(agent)
        H, W, C = obs_space['observation'].shape
        # ChessAgent ,模型在这个类里
        super().__init__(H=H,W=W,device=device)
        """
        初始化环境。
        :param render_mode: PettingZoo环境的渲染模式, 如 "human"。
        """

        # self.agent_map = {'player_0': 'white', 'player_1': 'black'}
        self.current_agent_name = None
        self.done = True

    def reset(self, seed:int|None=None):
        """
        重置环境，并返回白方的初始观察。
        """
        self._env.reset(seed=seed)
        self.done = False

        # 获取第一个智能体（白方）的初始观察
        observation, reward, termination, truncation, info = self._env.last()
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
        white_action, white_value = self.write_step(white_obs)
        self._env.step(white_action)

        # 获取白方行动后的状态
        white_reward, white_done, white_trunc, _ = self._env.last()
        self.done = white_done or white_trunc

        white_rollout = (white_obs, white_action, white_value, white_reward, self.done)

        # 如果白方走完后游戏结束，黑方没有机会行动
        if self.done:
            black_reward = self._env.rewards[self._env.agents[1]]  # 获取黑方的最终奖励
            black_obs, _, _, _, _ = self._env.last()
            # 对于黑方来说，它的回合没能开始就结束了
            _, black_value = self.black_step(black_obs)
            black_rollout = (black_obs, None, black_value, black_reward, self.done)
            return white_rollout, black_rollout

        # --- 黑方回合 ---
        self.current_agent_name = self._env.agent_selection
        black_obs, _, _, _, _ = self._env.last()
        black_action, black_value = self.black_step(black_obs)
        self._env.step(black_action)

        # 获取黑方行动后的状态
        black_reward, black_done, black_trunc, _ = self._env.last()
        self.done = black_done or black_trunc

        black_rollout = (black_obs, black_action, black_value, black_reward, self.done)

        # 如果黑方走完后游戏结束，需要更新白方的最终奖励
        if self.done:
            white_reward = self._env.rewards[self._env.agents[0]]
            white_rollout = (white_obs, white_action, white_value, white_reward, self.done)

        self.current_agent_name = self._env.agent_selection

        return white_rollout, black_rollout

    def close(self):
        """关闭环境。"""
        self._env.close()

    @property
    def action_space(self):
        """为当前玩家提供动作空间。"""
        return self._env.action_space(self.current_agent_name)