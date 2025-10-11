# TokenFreeDecoder
import os
import torch
import numpy as np
import lightning as L
torch.random.manual_seed(0)
from dataclasses import dataclass
from cxt.model import TokenFreeDecoder
from torch.utils.data import DataLoader
from cxt.dataset import LazyDataset, MultiDirLazyDataset
import math
import argparse


def generate_causal_mask(seq_len, full_attention_n=None, device="cpu"):
    full_attention_n = full_attention_n if full_attention_n is not None else 0
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    mask[:full_attention_n, :full_attention_n] = 1  # Full attention for first n tokens
    return mask.bool().unsqueeze(0).unsqueeze(0)

num_gpus = 2
config = {
    'training': {
        'max_steps': 50_000 * 2,
        'max_lr': 3e-4,
        'min_lr': 3e-4 * 0.1,
        'warmup_iters': 10,
        'lr_decay_iters': 50_000 * 3,
        'batch_size': 128,#196, #32
        'grad_accum_steps': 6,#1, 
        'weight_decay': 0.1,
        'betas': (0.9, 0.95),
        'num_workers': 16,#8,
        'prefetch_factor': 2,
    }
}




class LitTokenFreeDecoder(L.LightningModule):
    def __init__(self, gpt_config):
        super().__init__()
        self.model = TokenFreeDecoder(gpt_config)
        self.training_config = config['training']
        self.save_hyperparameters(ignore=['model'])
    def training_step(self, batch, batch_idx):
        x, y = batch
        attn_mask = generate_causal_mask(1001, full_attention_n=501, device='cuda')
        #attn_mask = generate_causal_mask(1001, device='cuda')
        attn_mask = attn_mask.repeat(x.size(0), 1, 1, 1)
        logits, loss = self.model(x, y, attn_mask)
        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        x, y = batch
        attn_mask = generate_causal_mask(1001, full_attention_n=501, device='cuda')
        #attn_mask = generate_causal_mask(1001, device='cuda')
        attn_mask = attn_mask.repeat(x.size(0), 1, 1, 1)
        #with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
        logits, loss = self.model(x, y, attn_mask)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    def configure_optimizers(self):
        optimizer = self.model.configure_optimizers(
            weight_decay=self.training_config['weight_decay'],
            learning_rate=self.training_config['max_lr'],
            betas=self.training_config['betas'],
            device_type=self.device.type
        )
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=self._get_lr_schedule_function
            ),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def _get_lr_schedule_function(self, current_step):
        config = self.training_config
        # Linear warmup
        if current_step < config['warmup_iters']:
            return float(current_step) / float(max(1, config['warmup_iters']))
        # Cosine decay
        if current_step > config['lr_decay_iters']:
            return config['min_lr'] / config['max_lr']
        decay_ratio = (current_step - config['warmup_iters']) / (
            config['lr_decay_iters'] - config['warmup_iters']
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config['min_lr'] / config['max_lr'] + coeff * (1.0 - config['min_lr'] / config['max_lr'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TokenFreeDecoder model")
    parser.add_argument('--dataset_path', type=str, default='/sietch_colab/kkor/cxt/ts/processed', help='Path to the dataset')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2], help='List of GPUs to use')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the checkpoint')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for the optimizer')
    parser.add_argument('--test_batches', type=int, default=100, help='Tiny batch size of 1000')

    args = parser.parse_args()
    dataset_path = args.dataset_path
    gpus = args.gpus
    checkpoint_path = args.checkpoint_path
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    test_batches = args.test_batches
    config['training']['max_lr'] = learning_rate

    # narrow model
    @dataclass
    class TokenFreeDecoderConfig:
        num_samples: int = 50
        sample_scale_embd: int = 2
        output_dim: int = 256+2
        n_embd: int = 400 #768
        combined_dim: int = 1001
        n_layer: int = 6 #12
        bias: bool = False
        dropout: float = 0.1
        n_head: int = 4 #8
        device: str = "cuda"
        batch_size: int = config['training']['batch_size']

    """
    # broad model
    @dataclass
    class TokenFreeDecoderConfig:
        num_samples: int = 50
        sample_scale_embd: int = 2
        output_dim: int = 324+2
        n_embd: int = 400 
        combined_dim: int = 1001
        n_layer: int = 10
        bias: bool = False
        dropout: float = 0.1
        n_head: int = 4
        device: str = "cuda"
        batch_size: int = config['training']['batch_size']
    """

    

    # Check if dataset_path contains multiple subdirectories
    """
    subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    if len(subdirs) > 1:
        print("Using multi directory dataset!")
        train_dataset = MultiDirLazyDataset(root_dir=dataset_path, split='train', test_ratio=0.1)
        test_dataset = MultiDirLazyDataset(root_dir=dataset_path, split='test', test_ratio=0.1)
        print(f"Training samples: {len(train_dataset)}")
        print(f"Testing samples: {len(test_dataset)}")
    else:
        print("Using single directory dataset!")
        train_dataset = LazyDataset(dataset_path, split='train', test_batches=test_batches)
        test_dataset = LazyDataset(dataset_path, split='test', test_batches=test_batches)
        print(f"training dataset {len(train_dataset)} samples")
        print(f"test dataset {len(test_dataset)} samples")
    """

    
    from cxt.dataset2 import PairDataset, ShuffleBufferDataset
    from torch.utils.data import DistributedSampler


    train_dataset = PairDataset(root=dataset_path, split="train", mmap=True)
    #sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)

    #train_dataset.shuffle_files(seed=1234)  # O(#files), no disk I/O
    #train_dataset = ShuffleBufferDataset(train_dataset, buffer_size=4096*8, seed=1234)

    #loader = DataLoader(ds, batch_size=196, shuffle=False, num_workers=32, drop_last=True, prefetch_factor=4)
    test_dataset = PairDataset(root=dataset_path, split="test", mmap=True)
    #test_dataset = ShuffleBufferDataset(test_dataset, buffer_size=4096*8, seed=5678)


    

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        shuffle=True, # taken care by shuffle buffer
        persistent_workers=True,
        prefetch_factor=config['training']['prefetch_factor'],
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        shuffle=False,
        persistent_workers=True,
        prefetch_factor=config['training']['prefetch_factor'],
        drop_last=True
    )
    gpt_config = TokenFreeDecoderConfig()
    

    if checkpoint_path:
        lit_model = LitTokenFreeDecoder.load_from_checkpoint(checkpoint_path, config=gpt_config)
    else:
        lit_model = LitTokenFreeDecoder(gpt_config)
    

    torch.set_float32_matmul_precision('medium')
    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        devices=gpus,
        precision="bf16-mixed",
        strategy="ddp", # not in interactive mode
        #detect_anomaly=True,
        #gradient_clip_val=1.0, # because of fused optim 
        accumulate_grad_batches=config['training']['grad_accum_steps']
    )
    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader
    )