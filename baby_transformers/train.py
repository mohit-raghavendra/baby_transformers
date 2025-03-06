from tqdm import tqdm
import torch
import torch.nn as nn
import pathlib

from baby_transformers.modules import LLM
from baby_transformers.data import get_dataloader
from baby_transformers.utils import get_tokenizer, initialize_wandb
from baby_transformers.config import Params, HyperParams   


class TrainerParams(Params):
    vocab_size: int = 50257   
    max_seq_len: int = 1024
    d_model: int = 768
    n_q: int = 12
    n_kv: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    n_layers: int = 10
    use_rope: bool = True
    activation: str = "silu"
    use_flash_attn: bool = False

class TrainerHyperParams(HyperParams):
    per_device_batch_size: int = 16
    global_batch_size: int = 32
    n_epochs: int = 2
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    log_steps: int = 1
    n_rows: int = 1500
    output_path: str = "./output"
    max_seq_len: int = 1024
    training_name: str = "transformer_vanilla"

    
    @property
    def gradient_accumulation_steps(self):
        return self.global_batch_size // self.per_device_batch_size

class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 criterion: nn.Module,
                 device: torch.device,
                 hyperparams: TrainerHyperParams,
                 ):
    
        self.hyperparams = hyperparams
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.wandb_run = initialize_wandb(hyperparams)
        self.scaler = torch.amp.GradScaler()
    

    @torch.autocast(device_type="cuda") # Mixed Precision Training
    def _run_step(self, input_ids, target_ids):
        logits = self.model(input_ids)
        loss = self.criterion(logits.view(-1, self.model.vocab_size), target_ids.view(-1))

        return loss
    
    def _save_results(self):
        output_path = pathlib.Path(self.hyperparams.output_path)
        output_path.mkdir(exist_ok=True)
        torch.save(self.model.state_dict(), output_path / "model_vanilla.pth")


    def train(self, train_loader):
        total_steps = self.hyperparams.n_epochs * len(train_loader)
        self.model.train()
        self.optimizer.zero_grad()
        cumulative_tokens = 0

        with tqdm(total=total_steps) as pbar:
            for epoch in range(1, self.hyperparams.n_epochs+1):
                for batch_idx, (input_ids, target_ids) in enumerate(train_loader, start=1):
                    input_ids = input_ids.to(self.device)
                    target_ids = target_ids.to(self.device)
                    loss = self._run_step(input_ids, target_ids)

                    scaled_loss = loss / self.hyperparams.gradient_accumulation_steps
                    self.scaler.scale(scaled_loss).backward()
                    cumulative_tokens += input_ids.numel()

                    if batch_idx % self.hyperparams.gradient_accumulation_steps == 0 or batch_idx == len(train_loader):
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                    
                    pbar.update(1)

                    if batch_idx % self.hyperparams.log_steps == 0:
                        message = f"Train loss: {loss.item():.4f}"      
                        pbar.set_description(message)
                        self.wandb_run.log({"#tokens": cumulative_tokens, "train/loss": loss})

        print(f"Total tokens processed: {cumulative_tokens}")
        pbar.update(total_steps - pbar.n)
        pbar.close()
        self._save_results()
    
def main():
    output_path = pathlib.Path("./output")
    output_path.mkdir(exist_ok=True)

    tokenizer = get_tokenizer()
    params = TrainerParams()
    hyperparams = TrainerHyperParams()

    print("Params:")
    print(params)
    print("HyperParams:")
    print(hyperparams)
    print(f"Global batch size: {hyperparams.global_batch_size}")
    print(f"Gradient accumulation steps: {hyperparams.gradient_accumulation_steps}")
    train_loader = get_dataloader(tokenizer=tokenizer, max_seq_len=params.max_seq_len, batch_size=hyperparams.per_device_batch_size, n_rows=hyperparams.n_rows)

    print(f"Total tokens to process - {hyperparams.n_epochs * train_loader.dataset.total_tokens}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LLM(
        vocab_size = params.vocab_size,
        max_seq_len = params.max_seq_len,
        n_layers = params.n_layers,
        d_model = params.d_model,
        d_ff = params.d_ff,
        n_q = params.n_q,
        n_kv = params.n_kv,
        dropout = params.dropout,
        use_rope=params.use_rope,
        activation=params.activation,
        use_flash_attn=params.use_flash_attn,
    ).to(device=device)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Total parameters: {format(total_params, ',')}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams.learning_rate, weight_decay=hyperparams.weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=hyperparams.n_epochs * len(train_loader))

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    trainer = Trainer(
        model = model,
        optimizer = optimizer,
        scheduler = scheduler,
        criterion = criterion,
        device = device,
        hyperparams = hyperparams,
    )

    trainer.train(train_loader)

if __name__ == "__main__":
    main()