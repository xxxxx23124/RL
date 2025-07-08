from dataclasses import dataclass, field
from ANN.Layers.FeedForward_layer.FeedForwardConfig import FeedForwardConfig
from ANN.Layers.Mamba2_layer.Mamba2Config import Mamba2Config
from ANN.Layers.Transformer_layer.TransformerConfig import TransformerConfig
@dataclass
class BlockConfig:
    d_model: int  # model dimension (D)
    max_seq_len: int # max_tokens

    mamba2_args: Mamba2Config = field(init=False)
    transformer_args: TransformerConfig = field(init=False)
    feedforward_args: FeedForwardConfig = field(init=False)
    def __post_init__(self):
        self.mamba2_args = Mamba2Config(self.d_model)
        self.transformer_args = TransformerConfig(self.d_model, self.max_seq_len)
        self.feedforward_args = FeedForwardConfig(self.d_model)
