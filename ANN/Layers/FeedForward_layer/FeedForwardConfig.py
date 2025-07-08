from dataclasses import dataclass, field
@dataclass
class FeedForwardConfig:
    d_model: int  # model dimension (D)
    expand: int = 3  # expansion factor (E)
    d_model_out: int = -1 # output dimension

    d_inner: int = field(init=False)
    def __post_init__(self):
        if self.d_model_out == -1:
            self.d_model_out = self.d_model
        
        self.d_inner = self.expand * self.d_model
