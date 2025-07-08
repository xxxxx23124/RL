import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from ANN.Layers.Mamba2_layer.InferenceCache import Mamba2InferenceCache
from ANN.Networks.NetworkConfig import NetworkConfig
from ANN.Blocks.TimeSpaceBlock import TimeSpaceBlock
from ANN.Layers.Norm_layer.RMSNorm import RMSNorm
from ANN.Blocks.SpatialFusion_block import SpatialFusion_block
from ANN.Layers.FeedForward_layer.SwiGLUMlp import SwiGLUFeedForward
from ANN.Layers.Transformer_layer.RotaryEmbedding import RotaryEmbedding
from ANN.Layers.FeedForward_layer.FeedForwardConfig import FeedForwardConfig

class TimeSpaceChunk(nn.Module):
    def __init__(self, args: NetworkConfig, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device

        # TimeSpaceBlocks
        if self.args.timespaceblock_num > 0:
            self.timespace_blocks = nn.ModuleList(
                [TimeSpaceBlock(args.block_args, device) for _ in range(args.timespaceblock_num)]
            )
        else:
            self.timespace_blocks = None

    def forward(        
            self,
            x:Tensor,
            H: int,
            W: int,
            rotary_emb: Optional[RotaryEmbedding] = None,
            cache_list:Optional[List[Mamba2InferenceCache]]=None,
            initial_ssm_states_list:Optional[List[Tensor]]=None
            ) -> tuple[Tensor, List[Tensor]]:
        # 这里的输入形状为: B, S, L, D
        ssm_states_list=[]
        # 1. 处理 TimeSpaceBlocks
        if self.timespace_blocks is not None:
            for i, block in enumerate(self.timespace_blocks):
                if cache_list is not None:
                    cache = cache_list[i]
                else:
                    cache = None
                if initial_ssm_states_list is not None:
                    initial_ssm_states = initial_ssm_states_list[i]
                else:
                    initial_ssm_states = None

                x, ssm_states = checkpoint(block,x,H,W,rotary_emb,cache,initial_ssm_states)
                ssm_states_list.append(ssm_states)
        
        return x, ssm_states_list
        
class SpaceChunk(nn.Module):
    def __init__(self, args: NetworkConfig, device: torch.device):
        super().__init__()
        self.args = args
        self.device = device

        # SpaceBlocks
        if self.args.spatialfusion_block_num > 0:
            self.space_blocks = nn.ModuleList(
                [SpatialFusion_block(args.block_args.transformer_args, device) for _ in range(args.spatialfusion_block_num)]
            )
        else:
            self.space_blocks = None

    def forward(        
            self,
            x:Tensor,
            H: int,
            W: int,
            rotary_emb: Optional[RotaryEmbedding] = None,
            ) -> Tensor:
        # 这里的输入形状为: B, S, L, D_low
        # 输出形状为: B, S, L, D_low
        # 处理空间讯息
        if self.space_blocks is not None:
            for i, block in enumerate(self.space_blocks):
                x = checkpoint(block, x, H, W, rotary_emb)
        return x


class TimeSpaceGoModel(nn.Module):
    def __init__(self, H:int, W:int ,device: torch.device):
        super().__init__()
        # 为了方便直接在这里配置超参数了
        self.d_model = 512
        self.value_d_model = 128

        self.H = H
        self.W = W

        # 卷积升维
        self.conv = nn.Conv2d(
            3,
            self.d_model,
            3,
            1,
            'same',
            device=device
        )

        # 配置主干网络
        self.backbone_args = NetworkConfig(self.d_model,
                                           (H*W)*2,
                                           timespaceblock_num=20,
                                           )
        self.backbone = TimeSpaceChunk(self.backbone_args, device=device)

        # 配置策略网络
        self.actorhead_args = NetworkConfig(self.d_model,
                                           (H*W)*2,
                                           timespaceblock_num=20,
                                           )
        self.actorhead = TimeSpaceChunk(self.actorhead_args, device=device)
        self.actorhead_output = nn.Linear(self.d_model, 1, device=device)
        self.action_ffd_args = FeedForwardConfig(d_model=H*W)
        self.pass_action = nn.Sequential(
            RMSNorm(H*W, device=device),
            SwiGLUFeedForward(self.action_ffd_args, device=device),
            nn.Linear(H*W, 1, device=device)
        )
        
        # 配置价值网络
        self.critic_args = NetworkConfig(self.value_d_model,
                                         (H*W)*2,
                                         spatialfusion_block_num=6,
                                         timespaceblock_num=1
                                         )
        self.critichead_begin_args = FeedForwardConfig(d_model=self.d_model,
                                                       d_model_out=self.value_d_model)
        self.critichead_begin = nn.Sequential(
            RMSNorm(self.d_model, device=device),
            SwiGLUFeedForward(self.critichead_begin_args, device=device),
        )
        
        
        self.critichead_timespace = TimeSpaceChunk(self.critic_args, device=device)
        self.critichead_space = SpaceChunk(self.critic_args, device=device)
        self.critichead_output = nn.Linear(self.value_d_model, 1, device=device)

        
        # 配置rotary_emb，rotary_emb是根据headdim来的，headdim在Transformer的配置文件中写死为64了
        self.rotary_emb = RotaryEmbedding(self.backbone_args.block_args.transformer_args, device=device)


    def forward(        
            self,
            x:Tensor,
            cache_list2:Optional[List[List[Mamba2InferenceCache]]]=None,
            initial_ssm_states_list2:Optional[List[List[Tensor]]]=None
            ) -> tuple[Tensor, Tensor, List[List[Tensor]]]:
        B, S, H, W, C = x.shape
        if cache_list2 is not None:
            cache_backbone_list = cache_list2[0]
            cache_policy_list = cache_list2[1]
            cache_value_list = cache_list2[2]
        else:
            cache_backbone_list = None
            cache_policy_list = None
            cache_value_list = None
        
        if initial_ssm_states_list2 is not None:
            initial_ssm_states_backbone_list = initial_ssm_states_list2[0]
            initial_ssm_states_policy_list = initial_ssm_states_list2[1]
            initial_ssm_states_value_list = initial_ssm_states_list2[2]
        else:
            initial_ssm_states_backbone_list = None
            initial_ssm_states_policy_list = None
            initial_ssm_states_value_list = None

        # 这里的输入形状为: x(B, S, H, W, C)
        x = rearrange(x, "b s h w c -> (b s) c h w")
        x = F.gelu(self.conv(x))
        x = rearrange(x, "(b s) d h w -> b s (h w) d", s=S)
        # x形状转变为：B, S, L, D
        ssm_states_list2 = []
        # 主干
        x_backbone, ssm_states_backbone_list = self.backbone(
                                                x,
                                                self.H,
                                                self.W,
                                                self.rotary_emb,
                                                cache_backbone_list,
                                                initial_ssm_states_backbone_list,
                                                )
        ssm_states_list2.append(ssm_states_backbone_list)

        # 策略
        action, ssm_states_policy_list = self.actorhead(
                                                x_backbone,
                                                self.H,
                                                self.W,
                                                self.rotary_emb,
                                                cache_policy_list,
                                                initial_ssm_states_policy_list,
                                                )
        ssm_states_list2.append(ssm_states_policy_list)
        # action B, S, L, D -> B, S, L
        action = self.actorhead_output(action).squeeze(-1)
        # action B, S, L -> B, S, 1
        pass_action = self.pass_action(action)
        policy = torch.cat([action, pass_action], dim=-1)
        # policy B, S, L+1

        # 价值
        # 这一步先降维
        value_input = self.critichead_begin(x_backbone)

        value, ssm_states_value_list = self.critichead_timespace(
                                                value_input,
                                                self.H,
                                                self.W,
                                                self.rotary_emb,
                                                cache_value_list,
                                                initial_ssm_states_value_list,
                                                )
        ssm_states_list2.append(ssm_states_value_list)

        value = self.critichead_space(value,
                                      self.H,
                                      self.W,
                                      self.rotary_emb,
                                      )
        # B, S, L, D_low -> B, S, L
        value = self.critichead_output(value).squeeze(-1)
        # B, S, L -> B, S, 1
        value = torch.mean(value, dim=-1, keepdim=True)
        # map value to [-1, 1]
        value = F.tanh(value)

        return policy, value, ssm_states_list2


def print_model_parameters_by_module(model):
    """
    按模块名称和层级打印参数数量。
    """
    print("--- Model Parameters by Module ---")
    total_params = 0
    for name, module in model.named_modules():
        # 我们只关心那些直接包含参数的模块 (Linear, Conv, LayerNorm, etc.)
        # 并且避免重复计算父模块的参数
        if hasattr(module, 'weight') and isinstance(module.weight, torch.nn.Parameter):
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"{name:<60} | Parameters: {params:,}")
                total_params += params

    # 手动计算可能遗漏的参数（如自定义层的参数）
    # 更稳妥的方式是直接迭代 named_parameters
    print("\n--- Parameters by Named Parameter ---")
    total_params = 0
    for name, param in model.named_parameters():
        print(f"{name:<60} | Shape: {str(list(param.shape)):<20} | Count: {param.numel():,}")
        total_params += param.numel()

    print("-" * 80)
    print(f"Total Trainable Parameters: {total_params:,}")
    print("-" * 80)


def print_model_summary(model, model_name="Model"):
    """
    打印模型的参数总量和可训练参数总量。
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"--- {model_name} Summary ---")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("-" * (len(model_name) + 14))

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("CUDA is not available. Testing on CPU. Memory usage will not be representative.")
        return
    
    H, W = 19, 19
    
    BATCH_SIZE = 1
    SEQUENCE_LENGTH = 8 # 模拟训练时一步占用的显存
    INPUT_CHANNELS = 3

    print(f"--- Model Test ---")
    print(f"Device: {device}")
    print(f"Board size: {H}x{W}")
    print(f"Batch size: {BATCH_SIZE}, Sequence length: {SEQUENCE_LENGTH}")
    
    # 1. 初始化模型
    try:
        model = TimeSpaceGoModel(H, W, device)
        model.to(device)
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Error during model initialization: {e}")
        return
    
    # 2. 创建模拟输入和目标
    # 输入形状: (B, S, H, W, C)
    input_tensor = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, H, W, INPUT_CHANNELS, device=device)

    # 目标策略形状: (B, S, L+1), L = H*W
    target_policy = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, H * W + 1, device=device)

    # 目标价值形状: (B, S, 1)
    target_value = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, 1, device=device)

    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"Target policy shape: {target_policy.shape}")
    print(f"Target value shape: {target_value.shape}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # 4. 运行前向和反向传播，并监控显存
    # 清空缓存以获得准确的初始显存读数
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / 1024**2
    print(f"\nInitial CUDA memory allocated: {initial_memory:.2f} MB")

    try:
        # --- 前向传播 ---
        print("Running forward pass...")
        policy_pred, value_pred, _ = model(input_tensor)
        
        # --- 计算损失 ---
        policy_loss = loss_fn(policy_pred, target_policy)
        value_loss = loss_fn(value_pred, target_value)
        total_loss = policy_loss + value_loss
        print(f"Forward pass successful. Total loss: {total_loss.item():.4f}")

        forward_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"Memory after forward pass: {forward_memory:.2f} MB")

        # --- 反向传播 ---
        print("Running backward pass...")
        optimizer.zero_grad()
        total_loss.backward()
        print("Backward pass successful.")

        backward_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"Memory after backward pass (before optimizer step): {backward_memory:.2f} MB")

        # --- 优化器步骤 ---
        optimizer.step()
        print("Optimizer step successful.")

        # --- 峰值显存 ---
        # max_memory_allocated() 返回的是峰值，更适合衡量最大占用
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\n--- Memory Usage Summary ---")
        print(f"Peak CUDA memory allocated during the process: {peak_memory:.2f} MB")

    except Exception as e:
        import traceback
        print(f"\nAn error occurred during the test run: {e}")
        traceback.print_exc()

    # print_model_parameters_by_module(model)
    print_model_summary(model, "TimeSpaceGoModel")

if __name__ == '__main__':
    test()