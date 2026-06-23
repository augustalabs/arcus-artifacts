"""Minimal nanoGPT (bias-free, weight-only LayerNorm) matching ode.pt, with
hooks to capture the per-layer residual stream and attention maps."""
import math, torch, torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_attn = nn.Linear(cfg['n_embd'], 3*cfg['n_embd'], bias=cfg['bias'])
        self.c_proj = nn.Linear(cfg['n_embd'], cfg['n_embd'], bias=cfg['bias'])
        self.n_head = cfg['n_head']; self.n_embd = cfg['n_embd']
        self.last_attn = None  # (B, nh, T, T) softmax weights, captured manually

    def forward(self, x, want_attn=False):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        if want_attn:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
            mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            self.last_attn = att.detach()
            y = att @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg['n_embd'], 4*cfg['n_embd'], bias=cfg['bias'])
        self.c_proj = nn.Linear(4*cfg['n_embd'], cfg['n_embd'], bias=cfg['bias'])
    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = LayerNorm(cfg['n_embd'], cfg['bias'])
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = LayerNorm(cfg['n_embd'], cfg['bias'])
        self.mlp = MLP(cfg)
    def forward(self, x, want_attn=False):
        x = x + self.attn(self.ln_1(x), want_attn=want_attn)
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg['vocab_size'], cfg['n_embd']),
            wpe=nn.Embedding(cfg['block_size'], cfg['n_embd']),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg['n_layer'])]),
            ln_f=LayerNorm(cfg['n_embd'], cfg['bias']),
        ))
        self.lm_head = nn.Linear(cfg['n_embd'], cfg['vocab_size'], bias=False)

    def forward(self, idx, want_attn=False, collect_residuals=False):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        residuals = [x.detach().clone()] if collect_residuals else None
        for blk in self.transformer.h:
            x = blk(x, want_attn=want_attn)
            if collect_residuals:
                residuals.append(x.detach().clone())
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits, residuals

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg['block_size']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature == 0.0:
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = float('-inf')
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

def load(path='ode.pt', device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = GPT(ck['model_config'])
    m.load_state_dict(ck['model'])
    m.eval().to(device)
    return m, ck
