import math
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L

from torcheval.metrics.functional import reciprocal_rank


from .modules import TransformerBlock


class TimeEmbedding(nn.Module):
    def __init__(self, dim, max_time=10000):
        super().__init__()
        self.base_freqs = nn.Parameter(
            torch.randn(1, 1, dim // 2) * (2 * torch.pi / max_time), 
            requires_grad=False
        )
        
    def forward(self, t):
        return torch.cat([torch.sin(self.base_freqs * t.unsqueeze(2)), torch.cos(self.base_freqs * t.unsqueeze(2))], dim=2)

class TGTransformer(L.LightningModule):
    def __init__(
            self, 
            n_vocab,
            d_hidden,
            d_mlp, 
            n_blocks = 2, 
            n_head = 4, 
            dropout=0.,
            lr=1e-3,
            wd=0.01
        ):
        super().__init__()
        self.lr = lr
        self.wd = wd
        self.node_embedding = nn.Embedding(n_vocab, d_hidden)
        self.time_embedding = TimeEmbedding(d_hidden)
        
        blocks = []
        for _ in range(n_blocks):
            blocks += [
                TransformerBlock(
                    d_hidden,
                    d_mlp,
                    n_head, 
                    dropout
                )
            ]
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
            nn.Linear(
                d_hidden, 
                n_vocab
            )
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, t):
        h = self.node_embedding(x) + self.time_embedding(t)
        for i, block in enumerate(self.blocks):
            h = block(h)
        return self.head(h)

    def step(self, batch, mode='train'):
        x, y, t = batch
        logits = self.forward(x, t)
        loss = self.criterion(logits.transpose(1,2), y)
        mrr = reciprocal_rank(logits[:, -1], y[:, -1]).mean()
        
        self.log(f"{mode}_loss", loss.item())
        self.log(f"{mode}_mrr", mrr.item())
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, mode='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, mode='test')
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), betas=(0.9, 0.95), lr=self.lr, weight_decay=self.wd
        )
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
            # optimizer, milestones=[50, 75], gamma=0.1
        # )
        return [optimizer]# [scheduler]