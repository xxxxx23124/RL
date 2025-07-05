重构造中，计划未来使用torchrl+gym+godot_rl+godot完成该项目。  
当前进度：mamba2的组件(已实现)，注意力组件(已实现)，旋转编码(已实现)，组装模型(未实现)，torchrl自定义强化学习组件(还不了解)，gym自定义游戏(还不了解)，godot(还不了解)  
计划未来对现有模型做简单的验证测试  
未来将把模型变为（时空分解注意力），mamba2处理时间步，flash attention处理空间（图像，每个像素点都当作一个token，其中可能加入cnn辅助注意力提取相邻的空间讯息）  
输入: (B, S, L, D)  
空间处理层 (Spatial Processor) Transformer的自注意力  
输入: (B*S, L, D)  
输出: (B*S, L, D)  
时间处理层 (Temporal Processor) Mamba2  
输入: (B*L, S, D)  
输出: (B*L, S, D)  
cnn辅助空间处理  
输入: (B, S, L->(H, W), D) -> (B*S, D, H, W)  
输出  (B*S, D, H, W)  