from dataclasses import dataclass, field
@dataclass
class FeedForwardConfig:
    d_model: int  # model dimension (D)
    expand: int = 3  # expansion factor (E)

    d_inner: int = field(init=False)
    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
