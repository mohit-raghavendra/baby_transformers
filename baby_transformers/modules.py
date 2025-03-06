import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func


VERBOSE = False

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        emb = self.embedding(x) # [B, T] -> [B, T, D]
        emb = emb * math.sqrt(self.d_model)
        return emb

class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

    def forward(self, x):
        # x shape: (B, T, D)
        B, T, _ = x.size()
        positions = torch.arange(0, T, device=x.device)
        positions = positions.unsqueeze(0).expand(B, T)
        pos_emb = self.pos_embedding(positions)  # [B, T] -> [B, T, D]
        return pos_emb

class RotaryEmbedding(nn.Module):
    def __init__(self, d, max_seq_len=512):
        super().__init__()
        self.d = d  # per-head dimension, must be even
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d)) # (d/2)
        positions = torch.arange(0, max_seq_len).float() # (T)
        self.thetas = torch.einsum("i,j->ij", positions, inv_freq) # (T, d/2)
        
        self.register_buffer("sin", torch.sin(self.thetas)) # (T, d/2)
        self.register_buffer("cos", torch.cos(self.thetas)) # (T, d/2)

    def forward(self, x):
        # x: (B, h, T, d) where d is even.
        T = x.size(2)
        sin = self.sin[:T, :].unsqueeze(0).unsqueeze(0)  # (1,1,T,d/2)
        cos = self.cos[:T, :].unsqueeze(0).unsqueeze(0)  # (1,1,T,d/2)
        x1, x2 = x.chunk(2, dim=-1)  # each: (B, h, T, d/2)

        if VERBOSE:
            print("Rotary embedding shapes")
            print(f"Input size - {x.size()}")
            print(f"thetas size - {self.thetas.size()}")
            print(f"sin size - {self.sin.size()}")

            print(f"sin size - {sin.size()}")
            print(f"cos size - {cos.size()}")


        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1) # [(B, h, T, d/2), (B, h, T, d/2)] -> (B, h, T, d)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_q, n_kv, dropout=0.0, use_rope=True, max_seq_len=512, use_flash_attn=True):
        super().__init__()
        assert d_model % n_q == 0, "d_model must be divisible by n_heads."

        self.d_model = d_model
        self.n_q = n_q

        self.d_q = d_model // n_q

        self.W_q = nn.Linear(d_model, n_q * self.d_q)
        self.W_k = nn.Linear(d_model, n_q * self.d_q)
        self.W_v = nn.Linear(d_model, n_q * self.d_q)
        self.W_o = nn.Linear(n_q * self.d_q, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout
        self.use_flash_attn = use_flash_attn
        self.use_rope = use_rope
        if self.use_rope:
            self.rope = RotaryEmbedding(d=self.d_q, max_seq_len=max_seq_len)


    def forward(self, x, mask=None):
        """
        x: (B, T, d_model)
        """
        B, T, _ = x.size()

        q_proj = self.W_q(x) # (B, T, d_model)
        k_proj = self.W_k(x) # (B, T, d_model)
        v_proj = self.W_v(x) # (B, T, d_model)

        q_proj = q_proj.reshape(B, T, self.n_q, self.d_q).permute(0, 2, 1, 3) # (B, T, n_q, d_q) -> (B, n_q, T, d_q)
        k_proj = k_proj.reshape(B, T, self.n_q, self.d_q).permute(0, 2, 1, 3) # (B, T, n_q, d_q) -> (B, n_q, T, d_q)
        v_proj = v_proj.reshape(B, T, self.n_q, self.d_q).permute(0, 2, 1, 3) # (B, T, n_q, d_q) -> (B, n_q, T, d_q)
                
        if self.use_rope:
            q_proj = self.rope(q_proj)
            k_proj = self.rope(k_proj)

        if self.use_flash_attn:
            out = flash_attn_func(
                q = q_proj.to(torch.bfloat16),
                k = k_proj.to(torch.bfloat16),
                v = v_proj.to(torch.bfloat16),
                dropout_p=self.dropout_p,
                causal=True,
            ).to(dtype=q_proj.dtype)
        else:
            # Our implementation
            k_proj = k_proj.transpose(-2, -1) # (B, n_q, d_q, T)

            scores = torch.matmul(q_proj, k_proj) / math.sqrt(self.d_q) # (B, n_q, [T, d_q]) * (B, n_kv, [d_q, T])
            scores = scores.masked_fill(mask==0, float("-inf"))

            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v_proj) # (B, n_q, T, T) * (B, n_q, T, d_q) -> (B, n_q, T, d_q)
            out = out.transpose(2, 1)

        out = out.reshape(B, T, self.n_q * self.d_q) # (B, T, n_q, d_kv) -> (B, T, d_model)
        out_proj = self.W_o(out) # (B, T, d_model)

        if VERBOSE:
            print("Attention module shapes")
            print(f"Input size - {x.size()}")

            print(f"q_proj - {q_proj.size()}")
            print(f"k_proj - {k_proj.size()}")
            print(f"v_proj - {v_proj.size()}")

            print(f"QK scores - {scores.size()}") # (B, n_q, T, T)

            print(f"attention matrix - {attn.size()}")
            print(f"out size - {out.size()}")
            print(f"out proj size - {out_proj.size()}")

        return out_proj


class GroupedMultiQueryAttention(nn.Module):
    def __init__(self, d_model, n_q, n_kv, dropout=0.0, use_rope=True, max_seq_len=512, use_flash_attn=True):
        super().__init__()
        assert d_model % n_q == 0, "d_model must be divisible by n_q."
        assert d_model % n_kv == 0, "d_model must be divisible by n_kv."
        assert n_q % n_kv == 0, "n_q must be divisible by n_kv."

        self.d_model = d_model
        self.n_q = n_q
        self.n_kv = n_kv

        self.d_q = d_model // n_q

        self.W_q = nn.Linear(d_model, n_q * self.d_q)
        self.W_k = nn.Linear(d_model, n_kv * self.d_q)
        self.W_v = nn.Linear(d_model, n_kv * self.d_q)
        self.W_o = nn.Linear(n_q * self.d_q, d_model)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)
        self.use_rope = use_rope
        self.use_flash_attn = use_flash_attn
        if self.use_rope:
            self.rope = RotaryEmbedding(d=self.d_q, max_seq_len=max_seq_len)


    def forward(self, x, mask=None):
        """
        x: (B, T, d_model)
        """
        B, T, _ = x.size()

        q_proj = self.W_q(x) # (B, T, d_model)
        k_proj = self.W_k(x) # (B, T, d_model)
        v_proj = self.W_v(x) # (B, T, d_model)

        q_proj = q_proj.reshape(B, T, self.n_q, self.d_q).permute(0, 2, 1, 3) # (B, T, n_q, d_q) -> (B, n_q, T, d_q)
        k_proj = k_proj.reshape(B, T, self.n_kv, self.d_q).permute(0, 2, 1, 3) # (B, T, n_kv, d_q) -> (B, n_kv, T, d_q)
        v_proj = v_proj.reshape(B, T, self.n_kv, self.d_q).permute(0, 2, 1, 3) # (B, T, n_kv, d_q) -> (B, n_kv, T, d_q)

        if self.use_rope:
            q_proj = self.rope(q_proj)
            k_proj = self.rope(k_proj)
            
        if self.use_flash_attn:
            out = flash_attn_func(
                q = q_proj.to(torch.bfloat16),
                k = k_proj.to(torch.bfloat16),
                v = v_proj.to(torch.bfloat16),
                dropout_p=self.dropout_p,
                causal=True,
            ).to(dtype=q_proj.dtype)
        else:
            # Our implementation
            if self.n_q > self.n_kv:
                repeat_factor = self.n_q // self.n_kv
                k_proj = k_proj.repeat(1, repeat_factor, 1, 1)  # (B, n_kv*repeat_factor, T, d_k) which is equal to (B, n_q, T, d_k)
                v_proj = v_proj.repeat(1, repeat_factor, 1, 1)

            k_proj = k_proj.transpose(-2, -1) # (B, n_kv, d_q, T)

            scores = torch.matmul(q_proj, k_proj) / math.sqrt(self.d_q) # (B, n_q, [T, d_q]) * (B, n_kv, [d_q, T])
            scores = scores.masked_fill(mask==0, float("-inf"))


            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v_proj) # (B, n_q, T, T) * (B, n_kv, T, d_q) -> (B, n_q, T, d_q)
            out = out.transpose(2, 1)
        
        out = out.reshape(B, T, self.n_q * self.d_q) # (B, T, n_q, d_kv) -> (B, T, d_model)
        out_proj = self.W_o(out) # (B, T, d_model)

        if VERBOSE:
            print(f"Using Flash Attention: {self.use_flash_attn}")
            print("Attention module shapes")
            print(f"Input size - {x.size()}")

            print(f"q_proj - {q_proj.size()}")
            print(f"k_proj - {k_proj.size()}")
            print(f"v_proj - {v_proj.size()}")

            if not self.use_flash_attn:
                print(f"QK scores - {scores.size()}") # (B, n_q, T, T)
                print(f"attention matrix - {attn.size()}")
                print(f"out size - {out.size()}")
                print(f"out proj size - {out_proj.size()}")

        return out_proj


class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0, activation_type="silu"):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff * 2
        self.activation_type = activation_type

        if self.activation_type == "silu":
            self.w_in = nn.Linear(d_model, d_ff * 2) # Will be split into two parts
        else:
            self.w_in = nn.Linear(d_model, d_ff)
        self.w_out = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        if self.activation_type == "silu":
            x = self.w_in(x)
            x_1, x_2 = x.chunk(2, dim=-1)
            x = F.silu(x_1) * x_2
        else:
            x = self.w_in(x)
            x = F.relu(x)


        x = self.dropout(x)
        x = self.w_out(x)

        if VERBOSE:
            print("MLP module shapes")
            print(f"Input size - {x.size()}")
            print(f"Intermediate size - {x.size()}")
            print(f"Output size - {x.size()}")

        return x
    

class MoEMLP(nn.Module):
    def __init__(self, d_model, d_ff, n_experts=8, top_k=2, dropout=0.0, activation_type="silu"):
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.n_experts = n_experts
        self.top_k = top_k

        self.gating_module = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList(
            [MLP(d_model, d_ff, dropout, activation_type) for _ in range(n_experts)]
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):

        gate_scores = self.gating_module(x) # (B, T, n_experts)
        gate_probs = F.softmax(gate_scores, dim=-1)

        topk_scores, topk_indices = gate_probs.topk(self.top_k, dim=-1) # (B, T, top_k), (B, T, top_k)

        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x) # (B, T, d_model)
            expert_outputs.append(expert_output)

        expert_outputs = torch.stack(expert_outputs, dim=2) # (B, T, n_experts, d_model)

        topk_indices_experts = topk_indices.unsqueeze(-1) # (B, T, top_k, 1)
        topk_indices_experts.expand(-1, -1, -1, self.d_model) # (B, T, top_k, d_model)

        selected_expert_outputs = torch.gather(
            input=expert_outputs, 
            dim=2,
            index=topk_indices_experts
        ) # (B, T, top_k, d_model)

        topk_scores = topk_scores.unsqueeze(-1) # (B, T, top_k, 1)

        output = (selected_expert_outputs * topk_scores).sum(dim=2) # (B, T, d_model)
        output = self.dropout(output)

        if VERBOSE:
            print("MoE module shapes")
            print(f"Input size - {x.size()}")
            print(f"Gate scores size - {gate_scores.size()}")
            print(f"Gate probs size - {gate_probs.size()}")
            print(f"Top k scores size - {topk_scores.size()}")
            print(f"Top k indices size - {topk_indices.size()}")
            print(f"Expert outputs size - {expert_outputs.size()}")
            print(f"Selected expert outputs size - {selected_expert_outputs.size()}")
            print(f"Output size - {output.size()}")

        return output


class TransformerLayer(nn.Module):
    def __init__(self, n_q, n_kv, d_model, d_ff, n_experts=1, top_k=1, dropout=0.0, use_rope=True, max_seq_len=1024, activation="silu", use_flash_attn=True):
        super().__init__()
        
        if n_q == n_kv:
            self.self_attn = MultiHeadAttention(d_model=d_model, n_q=n_q, n_kv=n_kv, dropout=dropout, use_rope=use_rope, max_seq_len=max_seq_len)
        else:
            self.self_attn = GroupedMultiQueryAttention(d_model=d_model, n_q=n_q, n_kv=n_kv, dropout=dropout, use_rope=use_rope, max_seq_len=max_seq_len, use_flash_attn=use_flash_attn)

        if n_experts > 1:
            assert 1 <= top_k <= n_experts, "top_k must be between 1 and n_experts."
            self.mlp = MoEMLP(d_model, d_ff, n_experts, top_k, dropout, activation)
        else:
            self.mlp = MLP(d_model, d_ff, dropout, activation)

        

        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, mask)
        attn_out = x + self.dropout(attn_out)
        h = self.norm1(attn_out)

        mlp_out = self.mlp(h)
        mlp_out = h + self.dropout(mlp_out)
        out = self.norm2(mlp_out)

        return out

class LLM(nn.Module):
    def __init__(self, vocab_size, max_seq_len, n_layers, d_model, d_ff, n_q, n_kv, dropout, use_rope=True, activation="silu", use_flash_attn=True, n_experts=1, top_k=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_q = n_q
        self.n_kv = n_kv
        self.dropout = dropout
        self.use_rope = True
        self.use_flash_attn = use_flash_attn    

        self.token_embedding = InputEmbedding(self.vocab_size, self.d_model)
        self.positional_embedding = PositionalEmbedding(max_seq_len=max_seq_len, d_model=d_model)

        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model = d_model,
                n_q = n_q,
                n_kv = n_kv,
                d_ff = d_ff,
                dropout = dropout,
                use_rope = use_rope,
                max_seq_len = max_seq_len, 
                activation=activation,
                use_flash_attn=use_flash_attn,
                n_experts=n_experts,
                top_k=top_k
            )
        for _ in range(n_layers)])

        self.final_norm = nn.RMSNorm(d_model)
        self.final_head = nn.Linear(d_model, vocab_size, bias=False)


    def forward(self, x):

        token_embedding = self.token_embedding(x)
        positional_embedding = self.positional_embedding(token_embedding)

        h = token_embedding + positional_embedding
        _, n, _ = h.size()

        mask = torch.tril(torch.ones(n, n, device=x.device)).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            h = layer(h, mask)

        h = self.final_norm(h)
        logits = self.final_head(h)

        if VERBOSE:
            print("LLM module shapes")
            print(f"Input size - {x.size()}")
            print(f"Token embedding size - {token_embedding.size()}")
            print(f"Positional embedding size - {positional_embedding.size()}")
            print(f"Input size - {h.size()}")
            print(f"Output size - {logits.size()}")

        return logits
    

if __name__ == "__main__":

    from pydantic import BaseModel

    class Params(BaseModel):
        vocab_size: int = 10000
        max_seq_len: int = 32
        n_layers: int = 6
        d_model: int = 512
        d_ff: int = 2048
        n_q: int = 8
        n_kv: int = 8
        dropout: float = 0.1
        top_k: int = 2
        n_experts: int = 8

    params = Params()
    n_seq = 1
    seq_len = 32


    model = LLM(
        vocab_size = params.vocab_size,
        max_seq_len = params.max_seq_len,
        n_layers = params.n_layers,
        d_model = params.d_model,
        d_ff = params.d_ff,
        n_q = params.n_q,
        n_kv = params.n_kv,
        dropout = params.dropout,
        use_rope = True,
        activation="silu",
        use_flash_attn=True,
        n_experts=1,
        top_k=1
    ).to("cuda")

    x = torch.randint(0, 10000, (n_seq, seq_len)).to("cuda")
    out = model(x)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {format(total_params, ',')}")

    print(out.size())

    # model = LLMMoE(
    #     vocab_size = params.vocab_size,
    #     max_seq_len = params.max_seq_len,
    #     n_layers = params.n_layers,
    #     d_model = params.d_model,
    #     d_ff = params.d_ff,
    #     n_q = params.n_q,
    #     n_kv = params.n_kv,
    #     dropout = params.dropout,
    #     n_experts = params.n_experts,
    #     top_k = params.top_k
    # )

    # x = torch.randint(0, 10000, (n_seq, seq_len))
    # out = model(x)

    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {format(total_params, ',')}")

    # print(out.size())


    # rope = RotaryEmbedding(512)

    # x = torch.randn(1, 8, 32, 512)
    # out = rope(x)
    # print(out.size())



