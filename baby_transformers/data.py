import datasets
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler

torch.random.manual_seed(42)

class TextDataset(Dataset):
    def __init__(self, seqs, tokenizer, seq_len):
        tokens = []
        for text in tqdm(seqs):
            tokens.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        
        self.seq_len = seq_len
        self.total_tokens = len(tokens)
        num_chunks = len(tokens) // (seq_len + 1)
        tokens = tokens[:num_chunks * (seq_len + 1)]
        self.examples = [
            tokens[i:i + seq_len + 1] for i in range(0, len(tokens), seq_len + 1)
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_ids = torch.tensor(example[:-1], dtype=torch.long)
        target_ids = torch.tensor(example[1:], dtype=torch.long)
        return input_ids, target_ids

def collate_fn(batch):
    input_ids = torch.stack([x for x, _ in batch])
    target_ids = torch.stack([y for _, y in batch])
    return input_ids, target_ids


def get_dataset(tokenizer, max_seq_len, n_rows=500):
    d = datasets.load_dataset("open-phi/textbooks", split=f"train[:{n_rows}]")
    train_seqs = d["markdown"]

    train_dataset = TextDataset(train_seqs, tokenizer, seq_len=max_seq_len)
    print(f"Dataset has {train_dataset.total_tokens} tokens")

    return train_dataset

def get_dataloader(tokenizer, max_seq_len, batch_size, n_rows=500):
    train_dataset = get_dataset(tokenizer, max_seq_len, n_rows=n_rows)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return train_dataloader


def get_dataloader_distributed(tokenizer, max_seq_len, batch_size, rank, world_size, pin_memory=True, num_workers=0, n_rows=500):
    train_dataset = get_dataset(tokenizer, max_seq_len, n_rows=n_rows)
    distributed_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    dataloader = DataLoader(
        dataset = train_dataset, 
        batch_size = batch_size,
        pin_memory = pin_memory,
        num_workers = num_workers,
        collate_fn = collate_fn,
        shuffle=False,
        sampler=distributed_sampler,
    )

    return dataloader





