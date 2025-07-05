from dataclasses import dataclass
from ANN.Layers.FeedForward_layer.FeedForwardConfig import FeedForwardConfig
from ANN.Layers.Mamba2_layer.Mamba2Config import Mamba2Config
from ANN.Layers.Transformer_layer.TransformerConfig import TransformerConfig
@dataclass
class BlockConfig:
    d_model: int  # model dimension (D)
    max_seq_len: int # max_tokens
    def __post_init__(self):
        self.mamba2_args = Mamba2Config(self.d_model)
        self.transformer_args = TransformerConfig(self.max_seq_len, self.d_model)
        self.feedforward_args = FeedForwardConfig(self.d_model)
