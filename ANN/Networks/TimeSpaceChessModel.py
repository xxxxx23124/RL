import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
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
            rotary_emb: RotaryEmbedding | None = None,
            cache_list:list[Mamba2InferenceCache] | None=None,
            ) -> tuple[Tensor, list[Mamba2InferenceCache]]:
        # 这里的输入形状为: B, S, L, D
        new_cache_list=[]
        # 1. 处理 TimeSpaceBlocks
        if self.timespace_blocks is not None:
            for i, block in enumerate(self.timespace_blocks):
                if cache_list is not None:
                    cache = cache_list[i]
                else:
                    cache = None
                x, cache = checkpoint(block,x,H,W,rotary_emb,cache)
                new_cache_list.append(cache)
        
        return x, new_cache_list
        
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
            rotary_emb: RotaryEmbedding | None = None,
            ) -> Tensor:
        # 这里的输入形状为: B, S, L, D_low
        # 输出形状为: B, S, L, D_low
        # 处理空间讯息
        if self.space_blocks is not None:
            for i, block in enumerate(self.space_blocks):
                x = checkpoint(block, x, H, W, rotary_emb)
        return x


class TimeSpaceChessModel(nn.Module):
    def __init__(self, H:int, W:int ,device: torch.device):
        super().__init__()
        # 为了方便直接在这里配置超参数了
        self.d_model = 512
        self.value_d_model = 128

        self.H = H
        self.W = W

        # 卷积升维
        self.conv = nn.Conv2d(
            112,
            self.d_model,
            3,
            1,
            'same',
            device=device
        )

        # 配置主干网络
        self.backbone_args = NetworkConfig(self.d_model,
                                           (H*W)*2,
                                           timespaceblock_num=38,
                                           )
        self.backbone = TimeSpaceChunk(self.backbone_args, device=device)

        # 配置策略网络
        self.actorhead_args = NetworkConfig(self.d_model,
                                           (H*W)*2,
                                           timespaceblock_num=38,
                                           )
        self.actorhead = TimeSpaceChunk(self.actorhead_args, device=device)
        self.actorhead_output = nn.Linear(self.d_model, 73, device=device)
        
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
            cache_list2:list[list[Mamba2InferenceCache]] | None=None,
            ) -> tuple[Tensor, Tensor, list[list[Mamba2InferenceCache]]]:
        B, S, H, W, C = x.shape
        if cache_list2 is not None:
            cache_backbone_list = cache_list2[0]
            cache_policy_list = cache_list2[1]
            cache_value_list = cache_list2[2]
        else:
            cache_backbone_list = None
            cache_policy_list = None
            cache_value_list = None

        # 这里的输入形状为: x(B, S, H, W, C)
        x = rearrange(x, "b s h w c -> (b s) c h w")
        x = F.gelu(self.conv(x))
        x = rearrange(x, "(b s) d h w -> b s (h w) d", s=S)
        # x形状转变为：B, S, L, D
        new_cache_list2 = []
        # 主干
        x_backbone, cache_list1 = self.backbone(
                                                x,
                                                self.H,
                                                self.W,
                                                self.rotary_emb,
                                                cache_backbone_list,
                                                )
        new_cache_list2.append(cache_list1)

        # 策略
        action, cache_list1 = self.actorhead(
                                                x_backbone,
                                                self.H,
                                                self.W,
                                                self.rotary_emb,
                                                cache_policy_list,
                                                )
        new_cache_list2.append(cache_list1)
        # action B, S, L, D -> B, S, L, 73
        action = self.actorhead_output(action)
        action = rearrange(action, "b s l d -> b s (l d)")
        # action B, S, 4672

        # 价值
        # 这一步先降维
        value_input = self.critichead_begin(x_backbone)

        value, cache_list1 = self.critichead_timespace(
                                                value_input,
                                                self.H,
                                                self.W,
                                                self.rotary_emb,
                                                cache_value_list,
                                                )
        new_cache_list2.append(cache_list1)

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

        return action, value, new_cache_list2


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
    """
    测试函数，修改为处理长序列（800），并使用分块、缓存和梯度累积。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("CUDA is not available. Testing on CPU. Memory usage will not be representative.")
        return

    H, W = 8, 8
    BATCH_SIZE = 1
    TOTAL_SEQUENCE_LENGTH = 353  # 总序列长度
    INPUT_CHANNELS = 112

    print(f"--- Model Test for Long Sequence Processing ---")
    print(f"Device: {device}")
    print(f"Board size: {H}x{W}")
    print(f"Batch size: {BATCH_SIZE}, Total sequence length: {TOTAL_SEQUENCE_LENGTH}")
    print(f"Chunking Strategy: Try 16 -> Try 8 -> Fallback to 1")

    # 1. 初始化模型
    try:
        model = TimeSpaceChessModel(H, W, device)
        model.to(device)
        print("Model initialized successfully.")
    except Exception as e:
        print(f"Error during model initialization: {e}")
        return

    # 2. 创建完整的模拟输入和目标
    # 输入形状: (B, S_total, H, W, C)
    full_input_tensor = torch.randn(BATCH_SIZE, TOTAL_SEQUENCE_LENGTH, H, W, INPUT_CHANNELS, device=device)
    # 目标策略形状: (B, S_total, 4672)
    full_target_policy = torch.randn(BATCH_SIZE, TOTAL_SEQUENCE_LENGTH, 4672, device=device)
    # 目标价值形状: (B, S_total, 1)
    full_target_value = torch.randn(BATCH_SIZE, TOTAL_SEQUENCE_LENGTH, 1, device=device)

    print(f"Full input tensor shape: {full_input_tensor.shape}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    # 3. 运行前向和反向传播，并监控显存
    # 清空缓存以获得准确的初始显存读数
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()  # 重置峰值统计
    initial_memory = torch.cuda.memory_allocated() / 1024 ** 2
    print(f"\nInitial CUDA memory allocated: {initial_memory:.2f} MB")

    try:
        # --- 梯度累积和分块处理 ---
        optimizer.zero_grad()  # 在循环开始前清零梯度

        cache_list2 = None  # 初始化缓存为空
        total_loss_accumulated = 0.0

        print("\nStarting chunk processing...")
        processed_len = 0
        while processed_len < TOTAL_SEQUENCE_LENGTH:
            remaining_len = TOTAL_SEQUENCE_LENGTH - processed_len

            # 决定当前块的大小
            if remaining_len >= 64:
                chunk_size = 64
            elif remaining_len >= 32:
                chunk_size = 32
            elif remaining_len >= 16:
                chunk_size = 16
            elif remaining_len >= 8:
                chunk_size = 8
            else:
                chunk_size = 1  # 如果连8都不到，就使用1来处理剩余部分

            # 获取当前块的数据
            chunk_input = full_input_tensor[:, processed_len: processed_len + chunk_size]
            chunk_target_policy = full_target_policy[:, processed_len: processed_len + chunk_size]
            chunk_target_value = full_target_value[:, processed_len: processed_len + chunk_size]

            print(
                f"  - Processing chunk: index {processed_len} to {processed_len + chunk_size - 1} (size: {chunk_size})")

            # --- 前向传播 ---
            # 使用 checkpoint 来节省显存
            # 将上一个块的 cache 传入
            policy_pred, value_pred, cache_list2 = model(chunk_input, cache_list2)

            # --- 计算损失 ---
            policy_loss = loss_fn(policy_pred, chunk_target_policy)
            value_loss = loss_fn(value_pred, chunk_target_value)
            # 为了防止梯度累积时因计算图释放导致损失过大，可以适当缩放
            chunk_loss = (policy_loss + value_loss) * (chunk_size / TOTAL_SEQUENCE_LENGTH)

            total_loss_accumulated += chunk_loss.item() * (TOTAL_SEQUENCE_LENGTH / chunk_size)  # 累加回未缩放的损失值

            # --- 反向传播 (梯度累积) ---
            # 每个块都进行反向传播，梯度会累加到 .grad 属性上
            chunk_loss.backward()

            processed_len += chunk_size

        print("All chunks processed successfully.")

        # --- 优化器步骤 ---
        # 在所有块处理完毕后，进行一次参数更新
        print("Running optimizer step...")
        optimizer.step()
        print("Optimizer step successful.")

        # --- 显存和损失总结 ---
        final_memory = torch.cuda.memory_allocated() / 1024 ** 2
        peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 2  # 整个过程中的峰值

        print(f"\n--- Run Summary ---")
        print(f"Final accumulated loss: {total_loss_accumulated:.4f}")
        print(f"Memory after optimizer step: {final_memory:.2f} MB")
        print(f"Peak CUDA memory allocated during the entire process: {peak_memory:.2f} MB")

    except Exception as e:
        import traceback
        print(f"\nAn error occurred during the test run: {e}")
        traceback.print_exc()

    # 打印模型参数信息
    print_model_summary(model, "TimeSpaceChessModel")


if __name__ == '__main__':
    test()