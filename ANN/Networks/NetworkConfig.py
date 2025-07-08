from dataclasses import dataclass, field
from ANN.Blocks.BlockConfig import BlockConfig

@dataclass
class NetworkConfig:
    d_model: int  # model dimension (D)
    max_seq_len: int # max_tokens
    spatialfusion_block_num:int=0
    mamba2_block_num:int=0
    timespaceblock_num:int=0

    block_args: BlockConfig = field(init=False)
    def __post_init__(self):
        self.block_args = BlockConfig(d_model = self.d_model, 
                                      max_seq_len=self.max_seq_len)