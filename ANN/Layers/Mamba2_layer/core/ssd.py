"""
命名约定:
B -> batch_size
L -> seq_len
H -> num_heads
D -> head_dim/state_dim (SSM状态空间的维度)
c -> num_chunks
l -> chunk_size
s -> s_idx (用于einsum中的序列索引)
p -> p_idx (用于einsum中的head_dim索引)

实现mamba2的ssd, ssd是一种ssm的特殊形式。
ssd接受输入: A,B,C,chunk_size,initial_states, device
x: B,L,H,D
A: B,L,H
B: B,L,H,D
C: B,L,H,D
ssd 返回一组ssm或者说是ssd的观察值y: batch_size,seq_len,num_heads,state_dim 
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

def segsum(
        x: Tensor,
        ) -> Tensor:
    x_cumsum = torch.cumsum(x, dim=-1) # 计算 x 的累积和 [a, a+b, a+b+c, ...]
    """
    类似于下面的计算:
    [[a,       a,       a,      a      ],
    [a+b,     a+b,     a+b,     a+b    ],
    [a+b+c,   a+b+c,   a+b+c,   a+b+c  ],
    [a+b+c+d, a+b+c+d, a+b+c+d, a+b+c+d]]
    ->
    [[0,   -b,    -(b+c),    -(b+c+d)],
    [b,     0,        -c,      -(c+d)],
    [b+c,   c,         0,         -d],
    [b+c+d, c+d,       d,          0]]
    """
    x_segsum = x_cumsum.unsqueeze(-1) - x_cumsum.unsqueeze(-2)
    T = x.size(-1)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
    """
    masked_fill后输出类似于下面的计算:
    [[0,   -inf, -inf, -inf],
    [b,   0,    -inf, -inf],
    [b+c, c,    0,    -inf],
    [b+c+d,c+d, d,    0   ]]
    其实在实际计算中a参数没有被使用, 所以b才是累加和的起点, 这与mamba2的1-ss矩阵的写法有点不同，但实际运行效果一样。
    """
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd(x,
        A, 
        B,
        C, 
        chunk_size=None, 
        initial_states=None, 
        device=None) -> tuple[Tensor, Tensor]:
    # 并行计算diagonal block
    B, L, D, S = x.shape
    """
    c是块的数量, l是chunk_size, B是batch_size, H是head数量, D是特征维度
    x: B,L,H,D -> B,c,l,H,D
    A: B,L,H -> B,c,l,H
    B: B,L,H,D -> B,c,l,H,D
    C: B,L,H,D -> B,c,l,H,D
    """
    x, A, B, C = [
        rearrange(m, "b (c l) ... -> b c l ...", l=chunk_size) for m in (x, A, B, C)
    ]
    # A: B,c,l,H -> B, H, c, l
    A = rearrange(A, "b c l h -> b h c l")
    """
    关注单个块和单个头 (固定 b, c, h):
    einsum("ln, sn, ls, sp -> lp")
    这个公式计算的是: Y_diag[l, p] = sum_{s <= l} ( sum_n C[l, n] * B[s, n] ) * L[l, s] * x[s, p]
    L[l,s] 仅在 l>=s 时非零。所以求和只在 s <= l 的范围内进行。
    """
    # L: B, H, c, l, l
    L = torch.exp(segsum(A, device=device))

    # 每个块内（对角块）的输出
    Y_diag = torch.einsum("bclhn, bcshn, bhcls, bcshp -> bclhp", C, B, L, x)

    # 右半部分（B项）
    A_cumsum = torch.cumsum(A, dim=-1) # 计算 A 的块内累积和 [a, a+b, a+b+c, ...]
    """
    A_cumsum[..., -1:] - A_cumsum
    类似于下面的计算:
    [a+b+c+d, a+b+c+d, a+b+c+d, a+b+c+d]
    ->
    [b+c+d, c+d, d, 0]
    实际上a没有被使用。
    """
    decay_states = torch.exp(A_cumsum[..., -1:] - A_cumsum)
    """
    固定 b, c, h: einsum("ln, l, lp -> pn")
    这个公式计算的是: state = sum_{l} B[l,n] * decay[l] * x[l,p]
    """
    states = torch.einsum("bclhn, bhcl, bclhp -> bchpn", B, decay_states, x)

    # 块间的循环计算 (A 项)
    """
    states 的形状是 (b, c, h, p, n)，我们需要在 c 维度 (dim=1) 前面加 1
    所以对应 (pad_left_for_dim_4, pad_right_for_dim_4, 
              pad_left_for_dim_3, pad_right_for_dim_3, 
              pad_left_for_dim_2, pad_right_for_dim_2,
              pad_left_for_dim_1, pad_right_for_dim_1) -> (0,0, 0,0, 0,0, 1,0)
    states -> (batch, chunk_count+1, head, d_head, d_state)
    """
    if initial_states is None:
        # 在 dim=1 (chunk 维度) 的前面填充 1 个单位，值为 0
        states = F.pad(states, (0, 0, 0, 0, 1, 0), "constant", 0)
    else:
        states = torch.cat([initial_states, states], dim=1)
    """
    A_cumsum 的形状是 (batch, head, chunk, chunk_len)
    A_cumsum[:, :, :, -1] 的形状是 (batch, head, chunk)
    F.pad(A_cumsum[..., -1], (1, 0), "constant", 0) -> [0, S_0, S_1, S_2, ..., S_{C-1}]
    pad后形状为(batch, head, chunk + 1)

    segsum输出为:(batch, head, chunk + 1, chunk + 1)
    [[0,             -inf,      -inf,        -inf],
    [S_0,               0,      -inf,        -inf],
    [S_0+S_1,         S_1,         0,        -inf],
    [S_0+S_1+S_2, S_1+S_2,       S_2,           0]]
    """
    decay_chunk = torch.exp(segsum(F.pad(A_cumsum[..., -1], (1, 0), "constant", 0), device=device))
    """
    decay_chunk 的形状 (batch, head, chunk_count+1, chunk_count+1)
    states 的形状 (batch, chunk_count+1, head, d_head, d_state)
    new_states 的形状 (batch, chunk_count+1, head, d_head, d_state)
    固定 b, h, p, n: einsum("zc, c -> z")
    """
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states, final_state = new_states[:, :-1], new_states[:, -1]
    # 计算状态到输出的转换 (C 项)
    state_decay_out = torch.exp(A_cumsum)
    """
    固定 b, c, h: einsum("ln, pn, l -> lp")
    """
    Y_off = torch.einsum("bclhn, bchpn, bhcl -> bclhp", C, states, state_decay_out)
    # 合并结果 将对角块（块内）和非对角块（块间）的输出相加
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state
