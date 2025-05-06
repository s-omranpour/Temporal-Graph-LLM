import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_hidden, d_mlp):
        super().__init__()
        self.w1 = nn.Linear(d_hidden, d_mlp, bias=False)
        self.w2 = nn.Linear(d_mlp, d_hidden, bias=False)
        self.w3 = nn.Linear(d_hidden, d_mlp, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.max_seq_len_cached = 0
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        if self.max_seq_len_cached < seq_len:
            self.max_seq_len_cached = seq_len
            t = torch.arange(
                self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
            )

            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer(
                "cos_cached", emb.cos().to(dtype), persistent=False
            )
            self.register_buffer(
                "sin_cached", emb.sin().to(dtype), persistent=False
            ) # seq_len, dim

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
        
    
    def forward(self, x):
        bs, l, nh, dh = x.shape
        self._set_cos_sin_cache(seq_len=l, device=x.device, dtype=x.dtype)
        cos = self.cos_cached[:l].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:l].unsqueeze(0).unsqueeze(2)
        return (x * cos) + (self.rotate_half(x) * sin)


class EfficientAttention(nn.Module):
    def __init__(
        self, 
        d_model,
        n_head,
        dropout=0.,
    ):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary_emb = RotaryEmbedding(self.d_head, max_position_embeddings=10000)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, causal=False):
        query = self.query_proj(X).unflatten(dim=-1, sizes=(self.n_head, self.d_head)).permute(0,2,1,3) 
        key = self.key_proj(X).unflatten(dim=-1, sizes=(self.n_head, self.d_head)).permute(0,2,1,3) 
        value = self.value_proj(X).unflatten(dim=-1, sizes=(self.n_head, self.d_head)).permute(0,2,1,3) 

        # with sdpa_kernel(SDPBackend.):
        h = F.scaled_dot_product_attention(
            self.rotary_emb(query), self.rotary_emb(key), value,
            is_causal=causal
        )

        return self.out_proj(
            self.dropout(
                h.permute(0,2,1,3).flatten(2)
            )
        )



class TransformerBlock(nn.Module):
    def __init__(
        self, 
        d_hidden, 
        d_mlp,
        n_head, 
        dropout=0.
    ):
        super().__init__()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_hidden)
        self.norm2 = nn.LayerNorm(d_hidden)
        self.attention = EfficientAttention(
            d_hidden, 
            n_head, 
            dropout
        )
        self.feedforward = SwiGLUFeedForward(d_hidden, d_mlp)

    def forward(self, X):
        h = X + self.drop1(self.attention(self.norm1(X), causal=True))
        return h + self.drop2(self.feedforward(self.norm2(h)))