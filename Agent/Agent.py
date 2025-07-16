import torch
import torch.nn.functional as F
from ANN.Networks.TimeSpaceChessModel import TimeSpaceChessModel
from ANN.Layers.Mamba2_layer.InferenceCache import Mamba2InferenceCache

class ChessAgent:
    def __init__(self, H:int, W:int ,device: torch.device):
        self.actor_critic = TimeSpaceChessModel(H, W, device)
        self.device = device
        self.write_cache_list: list[list[Mamba2InferenceCache]] | None = None
        self.black_cache_list: list[list[Mamba2InferenceCache]] | None = None

    def write_step(self, obs) -> tuple[int, float]:
        final_action_int, value, self.write_cache_list = self._step(obs, self.write_cache_list)
        return final_action_int, value

    def black_step(self, obs) -> tuple[int, float]:
        final_action_int, value, self.black_cache_list = self._step(obs, self.black_cache_list)
        return final_action_int, value

    def _step(self, obs, cache_list) -> tuple[int, float, list[list[Mamba2InferenceCache]]]:
        with torch.no_grad:
            core_obs = obs['observation']
            # 1. 将 NumPy 数组转换为 PyTorch 张量
            tensor_obs = torch.from_numpy(core_obs).float().to(self.device)  # 转换为浮点型
            # tensor_obs.shape 是 torch.Size([8, 8, 111])

            # 2. 增加批次和时间步维度
            # 使用 unsqueeze(dim) 在指定维度增加一个大小为1的维度
            final_tensor = tensor_obs.unsqueeze(0).unsqueeze(0)
            # 第一次 unsqueeze(0) -> torch.Size([1, 8, 8, 111])  (增加了批次维度)
            # 第二次 unsqueeze(0) -> torch.Size([1, 1, 8, 8, 111]) (增加了时间步维度)

            # 3. 从模型获取输出
            action_logits_from_model, value, cache_list = self.actor_critic(final_tensor, cache_list)
            # action_logits_from_model (1,1,4672)
            # value (1,1,1)
            # 4. 应用动作掩码
            # action_logits (1,4672)
            action_logits = action_logits_from_model.squeeze(1)
            action_mask = obs['action_mask']
            mask_tensor = torch.from_numpy(action_mask).to(self.device)
            # 将掩码中为 0 的位置 (非法动作) 在 logits 中对应的值设为负无穷
            # 这样 softmax 后的概率会趋近于 0
            # 我们使用一个很大的负数，因为 float('-inf') 有时会产生 NaN
            masked_logits = action_logits.masked_fill(mask_tensor == 0, -1e9)
            action_probs = F.softmax(masked_logits, dim=-1)

            # 5. 从概率分布中选择一个动作

            # 方法A: 采样 (用于训练，增加探索性)
            # torch.multinomial 从分布中抽取一个样本
            final_action = torch.multinomial(action_probs, num_samples=1)
            # final_action 会是一个形状为 (1, 1) 的张量，需要用 .item() 提取数值
            final_action_int = final_action.item()

            # 方法B: 确定性选择 (用于评估/测试，选择最优动作)
            # final_action = torch.argmax(action_probs, dim=-1)
            # final_action_int = final_action.item()

            return final_action_int, value.squeeze().item(), cache_list

    def reset(self):
        self.write_cache_list = None
        self.black_cache_list = None