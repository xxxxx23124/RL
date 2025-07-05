from dataclasses import dataclass
@dataclass
class FeedForwardConfig:
    d_model: int  # model dimension (D)
    expand: int = 3  # expansion factor (E)
    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
