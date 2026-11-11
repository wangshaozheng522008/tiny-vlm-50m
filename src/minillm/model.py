"""MiniLLM-87M: a dense decoder-only transformer for text.

Architecture (pre-norm, no biases):

    tokens --[embedding]--> x
    for layer in 16:
        x = x + GQA(RMSNorm(x))        # RoPE, causal, 10 Q heads / 2 KV heads
        x = x + SwiGLU(RMSNorm(x))     # gate * up -> down
    logits = shared_embedding.T @ RMSNorm(x)

The LM head is tied to the token embedding, so its weights are stored exactly
once. RoPE tables are non-persistent buffers: they are derived from the config,
not checkpoint data.

The same `forward` supports three modes:
  * full-sequence training      (past_key_values=None, use_cache=False)
  * prefill for generation      (use_cache=True, returns the cache)
  * single-token decode         (position_offset=len(cache), use_cache=True)
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MiniLLMConfig

# SDPA gained `enable_gqa` in torch 2.5; older builds need the explicit
# `repeat_kv` fallback. Detect once, then remember the answer for the process.
_SDPA_PARAMS = set()
try:  # pragma: no cover - depends on the installed torch
    _SDPA_PARAMS = set(inspect.signature(F.scaled_dot_product_attention).parameters)
except (TypeError, ValueError):  # pragma: no cover
    pass
_HAS_NATIVE_GQA = "enable_gqa" in _SDPA_PARAMS
_NATIVE_GQA_WORKS: bool | None = None if _HAS_NATIVE_GQA else False


class RMSNorm(nn.Module):
    """Root-mean-square normalisation, computed in fp32 for stability."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


def precompute_rope(
    head_dim: int, max_positions: int, theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (max_positions, head_dim) for the rotate-half RoPE."""
    if head_dim % 2:
        raise ValueError("head_dim must be even for RoPE")
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(max_positions, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the head dimension onto the first half."""
    first, second = x.chunk(2, dim=-1)
    return torch.cat([-second, first], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate (b, heads, t, head_dim) queries/keys by their absolute position.

    cos/sin are (t, head_dim); the rotation is applied in fp32 and cast back so
    fp16/bf16 autocast does not lose angle precision.
    """
    dtype = x.dtype
    x32 = x.float()
    out = x32 * cos[None, None] + rotate_half(x32) * sin[None, None]
    return out.to(dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand (b, n_kv, t, hd) to (b, n_kv*n_rep, t, hd) for grouped-query attention."""
    if n_rep == 1:
        return x
    b, n_kv, t, hd = x.shape
    return x[:, :, None].expand(b, n_kv, n_rep, t, hd).reshape(b, n_kv * n_rep, t, hd)


def _sdpa(q, k, v, *, attn_mask, dropout_p: float, is_causal: bool, n_rep: int):
    """SDPA with native GQA when available, explicit KV expansion otherwise."""
    global _NATIVE_GQA_WORKS
    kwargs = dict(attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
    if n_rep == 1:
        return F.scaled_dot_product_attention(q, k, v, **kwargs)
    if _NATIVE_GQA_WORKS is not False:
        try:
            out = F.scaled_dot_product_attention(q, k, v, enable_gqa=True, **kwargs)
            _NATIVE_GQA_WORKS = True
            return out
        except TypeError:  # pragma: no cover - very old torch
            _NATIVE_GQA_WORKS = False
    return F.scaled_dot_product_attention(q, repeat_kv(k, n_rep), repeat_kv(v, n_rep), **kwargs)


class Attention(nn.Module):
    """Grouped-query causal attention with RoPE and an optional KV cache."""

    def __init__(self, cfg: MiniLLMConfig):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dropout = cfg.attention_dropout
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, q_dim, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(q_dim, cfg.hidden_size, bias=cfg.attention_bias)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v) if use_cache else None

        out = _sdpa(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attention_mask is None and t > 1,
            n_rep=self.n_rep,
        )
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.o_proj(out), new_cache


class SwiGLU(nn.Module):
    """SwiGLU FFN: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: MiniLLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=cfg.mlp_bias)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=cfg.mlp_bias)
        self.dropout = cfg.hidden_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.silu(self.gate_proj(x)) * self.up_proj(x)
        y = self.down_proj(y)
        return F.dropout(y, p=self.dropout, training=self.training) if self.dropout else y


class Block(nn.Module):
    def __init__(self, cfg: MiniLLMConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, attention_mask=None, past_key_value=None, use_cache=False):
        h, cache = self.self_attn(
            self.input_layernorm(x), cos, sin,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + h
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, cache


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    loss_sum: torch.Tensor | None = None
    num_loss_tokens: int = 0
    past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None


class MiniLLM(nn.Module):
    def __init__(self, cfg: MiniLLMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(
            cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id
        )
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        cos, sin = precompute_rope(cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta)
        # Derived, not learned: keep them out of the state dict.
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Optional activation checkpointing (training only); set by train.py.
        self.gradient_checkpointing = False
        self._check_budget()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    # ------------------------------------------------------------------ sizes

    def _check_budget(self) -> None:
        """Fail loudly when the instantiated model does not match its config budget."""
        actual = sum(p.numel() for p in self.parameters())  # tied weights counted once
        expected = self.cfg.num_parameters
        if actual != expected:
            raise RuntimeError(
                f"parameter budget mismatch: config says {expected:,}, model has {actual:,}"
            )

    def parameter_breakdown(self) -> dict[str, int]:
        embed = self.embed_tokens.weight.numel()
        attn = sum(
            sum(p.numel() for n, p in layer.self_attn.named_parameters()) for layer in self.layers
        )
        ffn = sum(sum(p.numel() for p in layer.mlp.parameters()) for layer in self.layers)
        norm = sum(
            sum(p.numel() for n, p in layer.named_parameters() if "norm" in n)
            for layer in self.layers
        ) + self.norm.weight.numel()
        return {
            "embedding": embed,
            "attention": attn,
            "ffn": ffn,
            "norm": norm,
            "total": embed + attn + ffn + norm,
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------ mask

    def _build_mask(
        self, t: int, past_len: int, attention_mask: torch.Tensor | None, device
    ) -> torch.Tensor | None:
        """Boolean SDPA mask (True = may attend) for prefill/decode.

        A boolean mask is only materialised when padding is present or when a
        multi-token chunk is decoded against a cache; otherwise SDPA's native
        causal flag is enough.
        """
        if past_len == 0 and attention_mask is None:
            return None
        total = past_len + t
        query_pos = torch.arange(past_len, total, device=device).unsqueeze(-1)
        key_pos = torch.arange(total, device=device).unsqueeze(0)
        allow = key_pos <= query_pos  # (t, total)
        mask = allow[None, None]
        if attention_mask is not None:
            if attention_mask.shape[1] != t:
                raise ValueError("attention_mask must match the number of input tokens")
            mask = mask & attention_mask.to(torch.bool)[:, None, None, :]
        return mask

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        position_offset: int | None = None,
    ) -> ModelOutput:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        b, t = input_ids.shape
        if t == 0:
            raise ValueError("input_ids must not be empty")
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values must have one (k, v) pair per layer")

        past_len = past_key_values[0][0].shape[2] if past_key_values else 0
        offset = past_len if position_offset is None else position_offset
        if offset != past_len:
            raise ValueError("position_offset must match the cached sequence length")
        if offset + t > self.cfg.max_position_embeddings:
            raise ValueError(
                f"sequence length {offset + t} exceeds max_position_embeddings "
                f"{self.cfg.max_position_embeddings}"
            )

        x = self.embed_tokens(input_ids)
        cos = self.rope_cos[offset : offset + t]
        sin = self.rope_sin[offset : offset + t]
        mask = self._build_mask(t, past_len, attention_mask, input_ids.device)

        new_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training and not use_cache:
                x, cache = torch.utils.checkpoint.checkpoint(
                    layer, x, cos, sin, mask, past, use_cache, use_reentrant=False
                )
            else:
                x, cache = layer(x, cos, sin, attention_mask=mask, past_key_value=past, use_cache=use_cache)
            if use_cache:
                new_cache.append(cache)

        logits = self.lm_head(self.norm(x))
        out = ModelOutput(logits=logits, past_key_values=new_cache)
        if labels is not None:
            out.loss_sum, out.num_loss_tokens = self.loss_terms(logits, labels)
            out.loss = out.loss_sum / max(out.num_loss_tokens, 1)
        return out

    def loss_terms(self, logits: torch.Tensor, labels: torch.Tensor):
        """Summed next-token cross entropy and the number of supervised tokens."""
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_sum = F.cross_entropy(
            shift_logits.float().view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.cfg.ignore_index,
            reduction="sum",
        )
        return loss_sum, int((shift_labels != self.cfg.ignore_index).sum())

    # ------------------------------------------------------------------ generate

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 0,
        eos_token_id: int | None = None,
        use_cache: bool = True,
        attention_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        stop_token_ids: list[int] | tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Autoregressive decoding, greedy when temperature <= 0.

        With `use_cache=True` the prompt is prefilled once and each step only
        runs a single token through the decoder; with `use_cache=False` the whole
        prefix is recomputed every step (kept for equivalence tests).

        `stop_token_ids` extends the stop set: generation stops as soon as any
        of those ids (or `eos_token_id`, when given) is sampled. Rows that have
        already finished keep receiving the first stop id, so the return value
        stays a single rectangular tensor.
        """
        self.eval()
        if input_ids.dim() != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape (batch, nonempty_sequence)")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if input_ids.shape[1] + max_new_tokens > self.cfg.max_position_embeddings:
            raise ValueError("prompt plus max_new_tokens exceeds max_position_embeddings")

        stop_ids: list[int] = []
        if eos_token_id is not None:
            stop_ids.append(int(eos_token_id))
        for token_id in stop_token_ids or ():
            if int(token_id) not in stop_ids:
                stop_ids.append(int(token_id))
        fill_id = stop_ids[0] if stop_ids else None

        prompt_len = input_ids.shape[1]
        out = self.forward(input_ids, attention_mask=attention_mask, use_cache=use_cache)
        logits = out.logits[:, -1, :]
        cache = out.past_key_values
        generated = input_ids
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        for step in range(max_new_tokens):
            next_id = self._sample(logits, temperature, top_k, generator)
            if stop_ids:
                next_id = torch.where(
                    finished[:, None], torch.full_like(next_id, fill_id), next_id
                )
                hit = next_id == stop_ids[0]
                for token_id in stop_ids[1:]:
                    hit |= next_id == token_id
                finished |= hit.squeeze(1)
            generated = torch.cat([generated, next_id], dim=1)
            if bool(finished.all()) or step == max_new_tokens - 1:
                break
            if use_cache:
                out = self.forward(next_id, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
            else:
                out = self.forward(generated, use_cache=False)
            logits = out.logits[:, -1, :]
        return generated

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_k: int,
                generator: torch.Generator | None) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        scaled = logits / temperature
        if top_k > 0:
            k = min(top_k, scaled.size(-1))
            threshold = scaled.topk(k, dim=-1).values[:, -1, None]
            scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
        probs = F.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator)


def count_parameters(model: MiniLLM) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_memory_mb(cfg: MiniLLMConfig, batch: int, seq_len: int, bytes_per_param: float = 2.0) -> float:
    """Rough lower bound: weights + one forward's activations (no attention matrix)."""
    weights = cfg.num_parameters * bytes_per_param
    activations = batch * seq_len * cfg.hidden_size * bytes_per_param * 4  # x, qkv-ish, mlp, logits
    return (weights + activations) / 1024**2
