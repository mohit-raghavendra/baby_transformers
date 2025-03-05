import argparse
import torch
import os
import pathlib
import functools
import sys

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from tqdm import tqdm

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from baby_transformers.modules import LLM
from baby_transformers.data import get_dataloader_distributed
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

class TrainerHyperParams(HyperParams):
    per_device_batch_size: int = 16
    global_batch_size: int = 32
    n_epochs: int = 2
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    log_steps: int = 1
    n_rows: int = 500
    output_path: str = "./output"
    max_seq_len: int = 1024
    world_size: int = 2
    distributed_training_type: str = "ddp"

    @property
    def training_name(self):
        return f"transformer_{self.distributed_training_type}"
    
    ##################################################################
    ############ GRAD ACCUMULATION FOR DDP ############
    ##################################################################
    @property
    def gradient_accumulation_steps(self):
        return self.global_batch_size // (self.per_device_batch_size * self.world_size)

def setup(rank, world_size):

    ##################################################################
    ############ SETUP CODE ############
    ##################################################################

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    ##################################################################
    ############ CLEANUP CODE ############
    ##################################################################

    dist.destroy_process_group()        
        
class DDPTrainer:
    def __init__(self, 
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer, 
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 criterion: nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 hyperparams: TrainerHyperParams,
                 gpu_id: int,
                 world_size: int):

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.gpu_id = gpu_id
        self.world_size = world_size
        self.hyperparams = hyperparams
        self.model = model.to(device=gpu_id)
        self.scaler = torch.amp.GradScaler()

        if self.gpu_id == 0:
            self.wandb_run = initialize_wandb(hyperparams)

    def _print_memory_use(self, stage):
        if self.gpu_id == 0:
            # print(f"Memory allocated at {stage} - {format(torch.cuda.memory_allocated(device=self.gpu_id), ',')} bytes")
            print(f"Max memory allocated at {stage} - {format(torch.cuda.max_memory_allocated(device=self.gpu_id), ',')} bytes")

    @torch.autocast(device_type="cuda") # Mixed Precision Training
    def _run_step(self, input_ids, target_ids):
        logits = self.model(input_ids)
        loss = self.criterion(logits.view(-1, self.model.module.vocab_size),
                              target_ids.view(-1))

        return loss
    
    def _save_results(self):
        ##################################################################
        ############ ONLY RUN SAVES ON MASTER PROCESS ############
        ##################################################################

        if self.gpu_id == 0:
            output_path = pathlib.Path(self.hyperparams.output_path)
            output_path.mkdir(exist_ok=True)
            print("Saving model...")
            torch.save(self.model.module.state_dict(), output_path/ f"model_{self.hyperparams.training_name}.pth")

    
    def train(self):
        total_steps = self.hyperparams.n_epochs * len(self.train_loader)
        self.model.train()
        self.optimizer.zero_grad()
        cumulative_tokens = 0

        with tqdm(total=total_steps) as pbar:
            for epoch in range(1, self.hyperparams.n_epochs+1):
                self.train_loader.sampler.set_epoch(epoch)
                for batch_idx, (input_ids, target_ids) in enumerate(self.train_loader, start=1):

                    input_ids = input_ids.to(self.gpu_id)
                    target_ids = target_ids.to(self.gpu_id)
                    loss = self._run_step(input_ids, target_ids)
                    
                    scaled_loss = loss / self.hyperparams.gradient_accumulation_steps
                    self.scaler.scale(scaled_loss).backward()

                    ##################################################################
                    ####### GRADIENT ACCUMULATION - CUMULATE OVER ALL RANKS ##########
                    ##################################################################``
                    cumulative_tokens += (input_ids.numel() * self.world_size)

                    if batch_idx % self.hyperparams.gradient_accumulation_steps == 0 or batch_idx == len(self.train_loader):
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

                    pbar.update(1)

                    if batch_idx % self.hyperparams.log_steps == 0:     
                        message = f"Train loss: {loss.item():.4f}"      
                        pbar.set_description(message)
                        if self.gpu_id == 0:
                            self.wandb_run.log({"#tokens": cumulative_tokens, "train/loss": loss})

        if self.gpu_id == 0:
            print(f"Total tokens processed: {cumulative_tokens}")
        pbar.update(total_steps - pbar.n)
        pbar.close()
        self._save_results()


def main(rank, world_size, args):

    ##################################################################
    ############ 1. SETUP THE DDP TRAINING  ############
    ############ 2. FETCH A DISTRIBUTED DATA LOADER  ############
    ############ 2. PUT THE MODEL ON THE RIGHT DEVICE  ############
    ##################################################################

    distributed_training_type = args.distributed_training_type
    setup(rank, world_size)
    tokenizer = get_tokenizer()

    params = TrainerParams()
    hyperparams = TrainerHyperParams(world_size=world_size, 
                                      distributed_training_type=distributed_training_type)

    torch.cuda.set_device(rank)

    print("Params:")
    print(params)
    print("HyperParams:")
    print(hyperparams)
    print(f"Global batch size: {hyperparams.global_batch_size}")
    print(f"Gradient accumulation steps: {hyperparams.gradient_accumulation_steps}")
    print(f"World size: {world_size}")
    print(f"Distributed training type: {distributed_training_type}")
    print(f"Run name - {str(hyperparams.training_name)}")
    train_loader = get_dataloader_distributed(
        tokenizer=tokenizer,
        max_seq_len=params.max_seq_len,
        batch_size=hyperparams.per_device_batch_size,
        rank=rank,
        world_size=world_size,
        n_rows=hyperparams.n_rows)
    
    print(f"Total tokens to process - {hyperparams.n_epochs * train_loader.dataset.total_tokens}")
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
    ).to(rank)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {format(total_params, ',')}")


    ##################################################################
    ############ CONVERT MODEL TO DDP / FSDP ############
    ##################################################################

    if hyperparams.distributed_training_type == "ddp":
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    else:
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=1e5,
        )
        torch.cuda.set_device(rank)
        model = FSDP(
            module=model,
            # auto_wrap_policy=auto_wrap_policy,
        )        

    total_params_shard = sum(p.numel() for p in model.parameters())
    print(f"Total parameters after sharding: {format(total_params_shard, ',')}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams.learning_rate, weight_decay=hyperparams.weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=hyperparams.n_epochs * len(train_loader))

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    trainer = DDPTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_loader=train_loader,
        hyperparams=hyperparams,
        gpu_id=rank,
        world_size=world_size
    )

    trainer.train()

    cleanup()

if __name__ == "__main__":

    ##################################################################
    ############ LAUNCH PROCESSES  ############
    ##################################################################


    parser = argparse.ArgumentParser()

    parser.add_argument("--world_size", type=int, default=2, help="Number of GPUs to use")
    parser.add_argument("-d", "--distributed_training_type", type=str, default="ddp", help="Distributed training type - ddp or fsdp")
    args = parser.parse_args()

    mp.spawn(fn=main,
             args=(args.world_size, args),
             nprocs=args.world_size,
             join=True)