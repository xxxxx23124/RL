from dataclasses import dataclass, field
@dataclass
class Mamba2Config:
    d_model: int  # model dimension (D)
    d_state: int = 128  # state dimension (N)
    d_conv: int = 4  # convolution kernel size
    expand: int = 2  # expansion factor (E)
    headdim: int = 64  # head dimension (P)
    chunk_size: int = 32  # matrix partition size (Q)

    d_inner: int = field(init=False)
    nheads: int = field(init=False)
    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim