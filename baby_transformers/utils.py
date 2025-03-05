from transformers import GPT2TokenizerFast
import matplotlib.pyplot as plt
from pydantic import BaseModel
import wandb

from baby_transformers.config import HyperParams

def visualize(x: list[float], y: list[float], title: str, x_label: str, y_label: str, save_path: str = None):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

    if save_path:
        plt.savefig(save_path)


def get_tokenizer():
    tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def initialize_wandb(hyperparams: HyperParams):
    run = wandb.init(
        project="baby-transformers",
        name=hyperparams.training_name,
        config=hyperparams.model_dump()
    )
    run.define_metric("#tokens")
    run.define_metric("train/loss", step_metric="#tokens")
    return run
    

def get_num_params(d_model, n_layers, vocab_size, seq_len, include_embedding=True, bias=False):
    bias = bias * 1

    wpe = seq_len * d_model
    # wte shared with head so don't count
    model_ln = d_model + d_model * bias
    head_params = d_model * vocab_size

    # atn_block
    qkv = d_model * 3 * d_model
    w_out = d_model ** 2
    ln = model_ln
    attn_params = qkv + w_out + ln

    # ff
    ff = d_model * d_model * 4
    ln = model_ln
    ff_params = ff * 2 + ln

    params = (attn_params + ff_params) * n_layers + model_ln + head_params
    if include_embedding:
        params += wpe
    return params


def get_activations_mem(d_model, n_layers, vocab_size, n_heads, batch_size, seq_len):
    layer_norm = batch_size * seq_len * d_model * 4  # FP32

    embedding_elements = batch_size * seq_len * d_model   # Number of elements (comes up a lot)

    # attn block
    QKV = embedding_elements * 4  # FP32
    QKT = 2 * embedding_elements * 4  # FP32
    softmax = batch_size * n_heads * seq_len ** 2 * 4  # FP32
    PV = softmax / 2 + embedding_elements * 4  # FP32
    out_proj = embedding_elements * 4  # FP32
    attn_act = layer_norm + QKV + QKT + softmax + PV + out_proj

    # FF block
    ff1 = embedding_elements * 4  # FP32
    gelu = embedding_elements * 4 * 4  # FP32
    ff2 = embedding_elements * 4 * 4  # FP32
    ff_act = layer_norm + ff1 + gelu + ff2

    final_layer = embedding_elements * 4  # FP32
    model_acts = layer_norm + (attn_act + ff_act) * n_layers + final_layer

    # cross_entropy
    cross_entropy1 = batch_size * seq_len * vocab_size * 4  # FP32
    cross_entropy2 = cross_entropy1 * 4  # FP32
    ce = cross_entropy1 + cross_entropy2

    mem = model_acts + ce
    return mem

def total_memory(d_model, n_layers, vocab_size, n_heads, batch_size, seq_len):
    num_params = get_num_params(d_model, n_layers, vocab_size, seq_len)
    model_mem = num_params * 4  # FP32
    activations_mem = get_activations_mem(d_model, n_layers, vocab_size, n_heads, batch_size, seq_len)
    return model_mem + activations_mem

if __name__ == "__main__":
    class Params(BaseModel):
        vocab_size: int = 50257   # GPT-2's vocabulary size
        max_seq_len: int = 1024
        d_model: int = 768
        n_q: int = 12
        n_kv: int = 12
        d_ff: int = 3072
        dropout: float = 0.1
        n_layers: int = 28
        
    params = Params()
    num_params = get_num_params(params.d_model, params.n_layers, params.vocab_size, params.max_seq_len)
    print(f"Number of parameters in the model: {format(num_params, ',')}")

    total_memory = total_memory(params.d_model, params.n_layers, params.vocab_size, params.n_q, 1, params.max_seq_len)
    print(f"Total memory required for the model: {format(total_memory / 1e9, ',')}GB")