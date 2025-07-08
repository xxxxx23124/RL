from dataclasses import dataclass, field
@dataclass
class TransformerConfig:
    d_model: int  # model dimension (D)
    max_seq_len: int # max_tokens
    max_batch_size: int = 128 # 目前这个值没有使用，因为现在不需要对注意力进行任何缓存
    headdim: int = 64  # head dimension (P)
    base:int=10000 # RotaryEmbedding's base
    cnn_kernel_size: int = 3

    nheads: int = field(init=False)
    def __post_init__(self):
        self.nheads = self.d_model // self.headdim