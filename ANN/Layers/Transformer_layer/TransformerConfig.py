from dataclasses import dataclass
@dataclass
class TransformerConfig:
    max_seq_len: int # max_tokens
    d_model: int  # model dimension (D)
    headdim: int = 64  # head dimension (P)
    base:int=10000 # RotaryEmbedding's base
    def __post_init__(self):
        self.nheads = self.d_model // self.headdim