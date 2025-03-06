import torch
from transformers import GPT2Tokenizer
from baby_transformers.modules import LLM
from baby_transformers.train import TrainerParams
import pathlib

def sample_from_gpt2_model(model, prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.95):

    # Create BOS input
    prompt = prompt.strip()
    if prompt:
        input_ids = tokenizer.encode(prompt)
        input_ids = [tokenizer.bos_token_id] + input_ids
        input_ids = torch.tensor(input_ids, dtype=torch.long, device="cuda").unsqueeze(0)
        print(input_ids.size())
    # Generate text
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            logits = outputs[:, -1, :] / temperature #Corrected line.

            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                min_value = values[:, -1, None]
                logits = torch.where(logits < min_value, torch.full_like(logits, -float("Inf")), logits)

            probs = torch.softmax(logits, dim=-1)

            if top_p > 0 and top_p < 1:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[0, indices_to_remove] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)

            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break # stop generation if EOS token generated.

    # Decode generated sequence
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text


# Example usage

current_dir = pathlib.Path().resolve()
model_path = current_dir / "output/model_vanilla.pth"
params = TrainerParams()
model = LLM(
    vocab_size=params.vocab_size,
    max_seq_len=params.max_seq_len,
    d_model=params.d_model,
    n_q=params.n_q,
    n_kv=params.n_kv,
    d_ff=params.d_ff,
    dropout=params.dropout,
    n_layers=params.n_layers,
    use_rope=params.use_rope,
    activation=params.activation,
    use_flash_attn=params.use_flash_attn,
).to(torch.device("cuda"))
model.load_state_dict(torch.load(model_path))
model.eval()
print("Model loaded successfully!")
# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token # set pad token
print("Tokenizer loaded successfully!")

prompt = "This is a story of the "
generated_text = sample_from_gpt2_model(model, prompt = prompt, max_length=100, temperature=0.7, top_k=50, top_p=0.95)
print(generated_text)