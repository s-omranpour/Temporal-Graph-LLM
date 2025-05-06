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
            requires_grad=True
        )
        
    def forward(self, t):
        return torch.cat([torch.sin(self.base_freqs * t.unsqueeze(2)), torch.cos(self.base_freqs * t.unsqueeze(2))], dim=2)

class TGTransformer(L.LightningModule):
    def __init__(
            self, 
            n_vocab,
            n_feat,
            d_hidden,
            d_mlp, 
            n_blocks = 2, 
            n_head = 4, 
            dropout=0.,
            lr=1e-3,
            wd=0.01,
            neg_sampler=None,
            evaluator=None
        ):
        super().__init__()
        self.lr = lr
        self.wd = wd
        self.neg_sampler = neg_sampler
        self.evaluator = evaluator
        self.feat_embedding = nn.Linear(n_feat, d_hidden)
        self.node_embedding = nn.Embedding(n_vocab, d_hidden)
        self.time_embedding = TimeEmbedding(d_hidden)
        self.proj = nn.Linear(3*d_hidden, d_hidden)
        
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
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
            nn.Linear(
                d_hidden, 
                n_vocab
            )
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x_ind, x_feat, x_time):
        h = self.dropout(
            self.proj(
                torch.cat([
                    self.feat_embedding(x_feat),
                    self.node_embedding(x_ind), 
                    self.time_embedding(x_time)
                ], dim=-1)
            )
        )
        for i, block in enumerate(self.blocks):
            h = block(h)
        return self.head(h)

    def step(self, batch, mode='train'):
        x_ind, x_feat, x_time, y = batch
        logits = self.forward(x_ind, x_feat, x_time)
        loss = self.criterion(logits.transpose(1,2), y)
        
        self.log(f"{mode}_loss", loss.item())

        if mode in ['val', 'test']:
            assert self.evaluator is not None
            assert self.neg_sampler is not None
            metric = []

            src = x_ind[:, -1].cpu()
            dst = y[:,-1].cpu()
            ts = x_time[:, -1].cpu()
            logits = logits[:, -1].cpu().detach()
            neg_batch_list = self.neg_sampler.query_batch(
                src - 1, dst - 1, ts, 
                split_mode=mode
            )
            for idx, neg_batch in enumerate(neg_batch_list):
                neg_batch = torch.tensor(neg_batch) + 1
                input_dict = {
                    "y_pred_pos": logits[idx, dst[idx]],
                    "y_pred_neg": logits[idx, neg_batch.tolist()],
                    "eval_metric": ['mrr'],
                }
                metric += [self.evaluator.eval(input_dict)['mrr']]
            self.log(f"{mode}_MRR", torch.tensor(metric).mean().item())
            
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