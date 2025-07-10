重构造中，计划使用ppo训练自博弈下围棋。关于自博弈这一点，搞两个cache就可以分别代表黑棋白棋了。  
  
目前进度 -> 模型设计完成 -> ppo的基础训练流程(在legacy文件夹里)  
未来需要完成 -> 使用torchrl的ppo训练  
  
训练时可以使用checkpoint和ssm_states的机制，只要单片段的计算能够在硬件上承载，整个训练流程的显存占用就是稳定可控的。  
  
使用技术：mamba2，flash attention, RoPE，cnn，SwiGLUMlp，ppo，pettingZoo 
  
模型位置在：ANN -> Networks -> TimespaceGoModel.py里  

TimeSpaceBlock，mamba2处理时间步，flash attention处理空间（图像，每个像素点都当作一个token，其中可能加入cnn辅助注意力提取相邻的空间讯息）  
(B, S, L, D)  定义：B-> batch的大小，S->timestep的大小/时间步，L->输入图像的Location，L也等于H*W，D->每个输入图像的像素点对应的维度，也是cnn空间处理层对应卷积的Channel  
输入: (B, S, L, D)  
空间处理块 (Spatial Processor) SpatialFusion_block(一种使用注意力和卷积相辅相成的结构，结构为并行融合)  
输入: (B*S, L, D)  
输出: (B*S, L, D)  
时间处理块 (Temporal Processor) Mamba2  
输入: (B*L, S, D)  
输出: (B*L, S, D)  
(B, S, L, D)->空间处理层->时间处理层->...交替   
堆叠更多的交替层。通过深层堆叠，后一层的空间处理器可以间接地通过前一层的时间处理器获得时间信息，反之亦然。信息在层与层之间逐渐融合。  
  
模型总体架构：  
输入预处理：一个简单的卷积将(B, S, L, 3) -> (B, S, L, D)  
  
共享主干：n*TimeSpaceBlock -> 共享主干会对当前的全局有一个基础的形势判断。输入: (B, S, L, D) 输出: (B, S, L, D)  
共享主干 -> 策略头, 价值头  
  
策略头：m*TimeSpaceBlock ->负责根据来自共享主干的局势判断，和自身的策略知识做出下一步行动。输入: (B, S, L, D) 输出: (B, S, L+1) -> (B, S, L, D) 在最后一层后可以由一个线性映射为(B, S, L, 1) -> 动作(B, S, L)，最后的这个+1（停一手）来自线性映射得到的(B, S, L)由一个feedforward层映射->(B, S, 1)，然后与拼接动作(B, S, L)拼接为(B, S, L+1)。
  
价值头：i层feedforward层 + 1*TimeSpaceBlock + j个空间处理块 + 1*Linear层-> 不希望使用太多TimeSpaceBlock是因为减轻计算代价。输入: (B, S, L, D) 输出: (B, S, 1)。使用feedforward块把(B, S, L, D)->(B, S, L, D_low)，这一步相对于只根据来自主干的每个点独立的信息（维度D，可能包含很多其他的信息）判断每个点的价值（为D_low）。通过一层TimeSpaceBlock进行浅层提炼。然后再输入j层空间处理层，得到输出 (B, S, L, D_low)，最后再一个线性映射为(B, S, L, 1) -> (B, S, L) ，输入一个Global Average Pooling得到(B, S, 1)+tanh->[-1,1]。  
  
todo：  
(窗口化/局部注意力、分层/金字塔结构、稀疏/低秩近似注意力) -> 可以处理更大的图片，但这个可能不适合下棋，自然图像充满了高度的局部相关性和冗余。围棋棋盘几乎没有冗余。  
添加MCTS -> 增强下棋能力  
搞一个内置server -> 通过浏览器可视化，动态调参甚至对战之类的  
试试其他离散动作空间的强化学习算法  
  
补充一下，这个是Adam优化器下的显存占用，还是推荐Adam。  
  
============================= test session starts =============================  
collecting ... collected 1 item  
  
TimeSpaceGoModel.py::test PASSED [100%]--- Model Test ---  
Device: cuda  
Board size: 19x19  
Batch size: 1, Sequence length: 8  
Model initialized successfully.  
Input tensor shape: torch.Size([1, 8, 19, 19, 3])  
Target policy shape: torch.Size([1, 8, 362])  
Target value shape: torch.Size([1, 8, 1])  
  
Initial CUDA memory allocated: 437.81 MB  
Running forward pass...  
Forward pass successful. Total loss: 5.3176  
Memory after forward pass: 11632.51 MB  
Running backward pass...  
Backward pass successful.  
Memory after backward pass (before optimizer step): 11885.19 MB  
Optimizer step successful.  
  
--- Memory Usage Summary ---  
Peak CUDA memory allocated during the process: 15075.74 MB  
--- TimeSpaceGoModel Summary ---  
Total Parameters: 112,854,169  
Trainable Parameters: 112,854,169  
------------------------------  
  
  
======================== 1 passed, 1 warning in 7.67s =========================  
