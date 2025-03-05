import os
import pathlib
import argparse

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import functools

from transformers import GPT2TokenizerFast
from pydantic import BaseModel
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
    training_name: str = "transformer_dp"
    world_size: int = 2

def setup(rank, world_size):

    ##################################################################
    ############ SETUP CODE ############
    ##################################################################

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12346'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    ##################################################################
    ############ CLEANUP CODE ############
    ##################################################################

    dist.destroy_process_group()        
        
class FSDPTrainer:
    def __init__(self, 
                 model: nn.Module,
                 tokenizer: GPT2TokenizerFast,
                 optimizer: torch.optim.Optimizer, 
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 criterion: nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 hyperparams: HyperParams,
                 gpu_id: int,
                 world_size: int):

        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.train_loader = train_loader
        self.gpu_id = gpu_id
        self.world_size = world_size
        self.hyperparams = hyperparams
        self.model = model
        self.per_batch_num_tokens = self.hyperparams.max_seq_len * self.hyperparams.batch_size

        if self.gpu_id == 0:
            self.wandb_run = initialize_wandb(hyperparams)


    def _print_memory_use(self, stage):
        if self.gpu_id == 0:
            # print(f"Memory allocated at {stage} - {format(torch.cuda.memory_allocated(device=self.gpu_id), ',')} bytes")
            print(f"Max memory allocated at {stage} - {format(torch.cuda.max_memory_allocated(device=self.gpu_id), ',')} bytes")


    def _run_step(self, input_ids, target_ids):
        self.optimizer.zero_grad()
        logits = self.model(input_ids.to(self.gpu_id))
        loss = self.criterion(logits.view(-1, self.tokenizer.vocab_size), target_ids.view(-1).to(self.gpu_id))
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()
    
    def _save_results(self, train_losses):
        ##################################################################
        ############ ONLY RUN SAVES ON MASTER PROCESS ############
        ##################################################################
        cpu_state = self.model.state_dict()

        if self.gpu_id == 0:
            output_path = self.hyperparams.output_path
            output_path = pathlib.Path(output_path)
            output_path.mkdir(exist_ok=True)

            print("Saving model...")
            torch.save(cpu_state, f"./{output_path}/model_{self.hyperparams.training_name}.pth")

    
    def train(self):
        total_steps = self.hyperparams.n_epochs * len(self.train_loader)
        train_losses = []
        self.model.train()
        with tqdm(total=total_steps) as pbar:
            for epoch in range(1, self.hyperparams.n_epochs+1):
                self.train_loader.sampler.set_epoch(epoch)
                for batch_idx, (input_ids, target_ids) in enumerate(self.train_loader, start=1):
                    loss = self._run_step(input_ids, target_ids)
                    train_losses.append(loss)
                    if batch_idx % self.hyperparams.log_steps == 0:     
                        cumulative_steps = (epoch - 1) * len(self.train_loader) + batch_idx
                        num_tokens = self.per_batch_num_tokens * cumulative_steps

                        message = f"Train loss: {loss:.4f}"      
                        pbar.set_description(message)
                        pbar.update(self.hyperparams.log_steps)

                        if self.gpu_id == 0:
                            self.wandb_run.log({"#tokens": num_tokens, "train/loss": loss})

        self._save_results(train_losses)
        return train_losses


def main(rank, world_size):

    ##################################################################
    ############ 1. SETUP THE FSDP TRAINING  ############
    ############ 2. FETCH A DISTRIBUTED DATA LOADER  ############
    ############ 2. PUT THE MODEL ON THE RIGHT DEVICE  ############
    ##################################################################


    setup(rank, world_size)
    tokenizer = get_tokenizer()
    params = Params()
    hyperparams = HyperParams()

    torch.cuda.set_device(rank)

    train_loader = get_dataloader_distributed(
        tokenizer=tokenizer,
        max_seq_len=params.max_seq_len,
        batch_size=hyperparams.batch_size,
        rank=rank,
        world_size=world_size,
        n_rows=hyperparams.n_rows)
    
    model = LLM(
        vocab_size = params.vocab_size,
        max_seq_len = params.max_seq_len,
        n_layers = params.n_layers,
        d_model = params.d_model,
        d_ff = params.d_ff,
        n_q = params.n_q,
        n_kv = params.n_kv,
        dropout = params.dropout,
    ).to(device=rank)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {format(total_params, ',')}")

    
    ##################################################################
    ############ CONVERT MODEL TO FSDP ############
    ##################################################################

    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1e5,
    )
    torch.cuda.set_device(rank)
    model = FSDP(
        module=model,
        # auto_wrap_policy=auto_wrap_policy,
        )
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparams.learning_rate, weight_decay=hyperparams.weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=hyperparams.n_epochs * len(train_loader))

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    trainer = FSDPTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        train_loader=train_loader,
        hyperparams=hyperparams,
        gpu_id=rank,
        world_size=world_size,
    )

    train_losses = trainer.train()

    dist.barrier()  # Ensure all training is completed on all GPUs
    cleanup()

if __name__ == "__main__":

    ##################################################################
    ############ LAUNCH PROCESSES  ############
    ##################################################################

    parser = argparse.ArgumentParser()

    parser.add_argument("--world_size", type=int, default=2, help="Number of GPUs to use")
    args = parser.parse_args()

    mp.spawn(fn=main,
            args=(args.world_size,),
            nprocs=args.world_size,
            join=True
    )