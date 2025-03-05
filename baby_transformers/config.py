
from pydantic import BaseModel

class Params(BaseModel):
    vocab_size: int = 50257   # GPT-2's vocabulary size
    max_seq_len: int = 1024
    d_model: int = 768
    n_q: int = 12
    n_kv: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    n_layers: int = 24

class HyperParams(BaseModel):
    per_device_batch_size: int = 8
    global_batch_size: int = 32
    n_epochs: int = 2
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    log_steps: int = 1
    n_rows: int = 2
    output_path: str = "./output"
    max_seq_len: int = 1024