from pydantic import BaseModel
from tqdm import tqdm
import torch
import torch.nn as nn
import pathlib


from baby_transformers.modules import LLMMoE
from baby_transformers.data import get_dataloader
from baby_transformers.utils import get_tokenizer, visualize

class Params(BaseModel):
    vocab_size: int = 50257   # GPT-2's vocabulary size
    max_seq_len: int = 1024
    d_model: int = 768
    n_q: int = 12
    n_kv: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    n_layers: int = 24
    n_experts: int = 3
    top_k: int = 2

class HyperParams(BaseModel):
    batch_size: int = 6
    n_epochs: int = 2
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    log_steps: int = 1
    n_rows: int = 500
    output_path: str = "./output"
    max_seq_len: int = 1024


class TrainerMoE:
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 criterion: nn.Module,
                 device: torch.device,
                 hyperparams: HyperParams):
    
        self.hyperparams = hyperparams
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device

    def _run_step(self, input_ids, target_ids):
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(input_ids)
        loss = self.criterion(logits.view(-1, self.model.vocab_size), target_ids.view(-1))
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()
    
    def _save_results(self, train_losses):
        output_path = self.hyperparams.output_path
        output_path = pathlib.Path(output_path)
        output_path.mkdir(exist_ok=True)
        per_batch_tokens = self.hyperparams.max_seq_len * self.hyperparams.batch_size
        x_vals = [i*per_batch_tokens for i in range(len(train_losses))] 
        y_vals = train_losses
        visualize(x_vals, y_vals, "train loss", "#tokens", "train loss", f"{output_path}/train_loss_moe.png")
        torch.save(self.model.state_dict(), f"./{output_path}/model_moe.pth")


    def train(self, train_loader, n_epochs):
        total_steps = n_epochs * len(train_loader)
        train_losses = []
        with tqdm(total=total_steps) as pbar:
            for epoch in range(n_epochs):
                for batch_idx, (input_ids, target_ids) in enumerate(train_loader, start=1):
                    input_ids = input_ids.to(self.device)
                    target_ids = target_ids.to(self.device)
                    loss = self._run_step(input_ids, target_ids)
                    train_losses.append(loss)

                    if batch_idx % self.hyperparams.log_steps == 0:     
                        message = f"Train loss: {loss:.4f}"      
                        pbar.set_description(message)
                        pbar.update(self.hyperparams.log_steps)

        self._save_results(train_losses)
        return train_losses
    
def main():
    output_path = pathlib.Path("./output")
    output_path.mkdir(exist_ok=True)

    tokenizer = get_tokenizer()
    params = Params()
    hyperparams = HyperParams()
    train_loader = get_dataloader(tokenizer=tokenizer, max_seq_len=params.max_seq_len, batch_size=hyperparams.batch_size, n_rows=hyperparams.n_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LLMMoE(
        vocab_size = params.vocab_size,
        max_seq_len = params.max_seq_len,
        n_layers = params.n_layers,
        d_model = params.d_model,
        d_ff = params.d_ff,
        n_q = params.n_q,
        n_kv = params.n_kv,
        dropout = params.dropout,
        n_experts = params.n_experts,
        top_k = params.top_k
    ).to(device=device)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Total parameters: {format(total_params, ',')}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams.learning_rate, weight_decay=hyperparams.weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=hyperparams.n_epochs * len(train_loader))

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    trainer = TrainerMoE(
        model = model,
        optimizer = optimizer,
        scheduler = scheduler,
        criterion = criterion,
        device = device,
        hyperparams = hyperparams
    )

    train_losses = trainer.train(train_loader, hyperparams.n_epochs)

if __name__ == "__main__":
    main()