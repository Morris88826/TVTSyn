from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .streaming import StreamingModule


# ------------------------ #
# Rotary Positional Embeds #
# ------------------------ #

def build_rope_cache(seq_len: int, dim: int, device, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    assert dim % 2 == 0
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("s,d->sd", t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

def build_rope_cache_offset(start: int, length: int, dim: int, device, base: float = 10000.0):
    assert dim % 2 == 0
    half = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(start, start + length, device=device).float()
    freqs = torch.einsum("l,d->ld", t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

def rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B,H,S,D], cos/sin: [1,1,S,D]; D even
    x1, x2 = x[..., ::2], x[..., 1::2]
    y1 = x1 * cos[..., ::2] - x2 * sin[..., ::2]
    y2 = x1 * sin[..., ::2] + x2 * cos[..., ::2]
    return torch.stack((y1, y2), dim=-1).flatten(-2)

# ---------------- #
# Token Dropout    #
# ---------------- #

def token_dropout_mask(T: int, p: float, device) -> torch.Tensor:
    if p <= 0:
        return torch.ones(T, device=device)
    m = (torch.rand(T, device=device) > p).float()
    if m.sum() == 0:
        m[torch.randint(0, T, (1,), device=device)] = 1.0
    return m

# ------------------------------ #
# Simple Cross-Attention (SDPA)  #
# ------------------------------ #

class CrossAttn(nn.Module):
    def __init__(self, dim_q: int, dim_kv: int, d_model: int, n_heads: int,
                 attn_dropout: float = 0.0, out_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim_q, d_model, bias=False)
        self.k_proj = nn.Linear(dim_kv, d_model, bias=False)
        self.v_proj = nn.Linear(dim_kv, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = attn_dropout
        self.out_dropout = nn.Dropout(out_dropout)

    def _shape(self, x: torch.Tensor, B: int) -> torch.Tensor:
        return x.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self,
        Q_seq: torch.Tensor,     # [B, Sq, dim_q]
        K_seq: torch.Tensor,     # [B, Sk, dim_kv]
        V_seq: torch.Tensor,     # [B, Sk, dim_kv]
        rope_q: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rope_k: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        key_padding_mask: Optional[torch.Tensor] = None   # [B, Sk]
    ) -> torch.Tensor:
        B, Sq, _ = Q_seq.shape
        _,  Sk, _ = K_seq.shape
        q = self._shape(self.q_proj(Q_seq), B)  # [B,H,Sq,d]
        k = self._shape(self.k_proj(K_seq), B)  # [B,H,Sk,d]
        v = self._shape(self.v_proj(V_seq), B)  # [B,H,Sk,d]

        if rope_q is not None:
            q = rope_apply(q, rope_q[0], rope_q[1])
        if rope_k is not None:
            k = rope_apply(k, rope_k[0], rope_k[1])

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,Sq,Sk]
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :].bool(), float("-inf"))
        w = F.softmax(scores, dim=-1)
        if self.attn_dropout and self.training:
            w = F.dropout(w, p=self.attn_dropout)
        out = torch.matmul(w, v).transpose(1, 2).reshape(B, Sq, -1) # [B,Sq,d_model]
        return self.out_dropout(self.out(out))                       # [B,Sq,d_model]

# ------------------------------- #
# Global Timbre Memory (per-utt)  #
# ------------------------------- #

class GlobalTimbreMemory(nn.Module):
    """
    Expand a global timbre vector into per-utterance K/V token sets.
    """
    def __init__(self, g_dim: int = 192, d_kv: int = 128, num_tokens: int = 48, hidden: int = 256):
        super().__init__()
        self.T = num_tokens
        self.d_kv = d_kv
        # Simple MLP -> [B, T*(K+V)]
        self.net = nn.Sequential(
            nn.Linear(g_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * num_tokens * d_kv)
        )
        # Optional per-token prior (learned prototypes) helps stabilize early training
        self.k_prior = nn.Parameter(torch.randn(num_tokens, d_kv) * 0.02)
        self.v_prior = nn.Parameter(torch.randn(num_tokens, d_kv) * 0.02)

    def forward(self, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        g: [B, g_dim]
        returns:
          K_seq, V_seq: [B, T, d_kv]
        """
        B = g.size(0)
        raw = self.net(g)                                 # [B, 2*T*d]
        k, v = torch.split(raw, self.T * self.d_kv, dim=-1)
        k = k.view(B, self.T, self.d_kv) + self.k_prior   # [B,T,d]
        v = v.view(B, self.T, self.d_kv) + self.v_prior   # [B,T,d]
        return k, v


def slerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a_n = F.normalize(a, dim=-1); b_n = F.normalize(b, dim=-1)
    dot = (a_n * b_n).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    omega = torch.acos(dot); sin_omega = torch.sin(omega).clamp_min(eps)
    return (torch.sin((1 - t) * omega) / sin_omega) * a_n + (torch.sin(t * omega) / sin_omega) * b_n


@dataclass
class _TVTimbreState:
    pos_offset: int

    def reset(self):
        self.pos_offset = 0


class StreamingGlobalOnlyTVTimbre(StreamingModule[_TVTimbreState]):
    def __init__(
            self, 
            content_dim: int, 
            g_dim: int = 704, 
            d_kv: int = 128, 
            d_out: int = 192, 
            num_tokens: int = 48,
            n_heads: int = 4, 
            attn_dropout: float = 0.0, 
            gate_hidden: int = 256,
            token_dropout_p: float = 0.1, 
            rope_base: float = 10000.0,
        ):
        super().__init__()
        assert d_kv % n_heads == 0
        self.d_kv = d_kv
        self.d_out = d_out
        self.head_dim = d_kv // n_heads
        self.num_tokens = num_tokens
        self.token_dropout_p = token_dropout_p
        self.rope_base = rope_base

        self.gtm = GlobalTimbreMemory(g_dim=g_dim, d_kv=d_kv, num_tokens=num_tokens)
        
        self.q_proj = nn.Conv1d(content_dim, d_kv, 1, bias=False)
        self.cross = CrossAttn(d_kv, d_kv, d_kv, n_heads, attn_dropout, 0.0)
        self.speaker_proj = nn.Linear(g_dim, d_out)
        self.post = nn.Conv1d(d_kv, d_out, 1)
        self.gate = nn.Sequential(nn.Conv1d(content_dim, gate_hidden, 1), nn.GELU(), nn.Conv1d(gate_hidden, 1, 1), nn.Sigmoid())

    def _init_streaming_state(self, batch_size: int) -> _TVTimbreState:
        return _TVTimbreState(pos_offset=0)

    def forward(self, x: torch.Tensor, global_timbre: torch.Tensor) -> torch.Tensor:
        B, _, T = x.shape
        state = self._streaming_state
        device = x.device
        K_seq, V_seq = self.gtm(global_timbre)  # [B,NT,D]
        if self.training and self.token_dropout_p > 0:
            keep = token_dropout_mask(K_seq.size(1), self.token_dropout_p, device)
            K_seq = K_seq * keep[None, :, None]
            V_seq = V_seq * keep[None, :, None]

        q_new = self.q_proj(x).transpose(1,2)                 # [B,T,D]
        alpha = self.gate(x).transpose(1, 2)                  # [B,T,1]

        if state is None:
            start = 0
        else:
            start = int(state.pos_offset)

        cos_q, sin_q = build_rope_cache_offset(start, T, self.head_dim, device, base=self.rope_base)
        attn = self.cross(q_new, K_seq, V_seq, rope_q=(cos_q, sin_q), rope_k=None) # [B,T,D]

        H = attn.transpose(1, 2)                                     # [B,D,T]
        H192 = F.normalize(self.post(H), dim=1)                      # [B,192,T]
        g = self.speaker_proj(F.normalize(global_timbre, dim=-1)).unsqueeze(-1).expand(B, self.d_out, T)  # [B,192,T]
        tv_embed = slerp(H192.transpose(1, 2), g.transpose(1, 2), alpha).transpose(1, 2)

        if state is not None:
            state.pos_offset += T

        return tv_embed
