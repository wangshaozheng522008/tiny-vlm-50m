"""TinyVLM: a small decoder-only vision-language model.

Architecture
------------
    image --[ViT encoder]--> patch features --[linear projector]--+
                                                                  |
    text  --[token embedding + position]--------------------------+--> [causal
                                                                  |    decoder]
                                                                  |       |
                                                                  +-----> text

Image tokens are prepended to the text sequence as a prefix. The decoder attends
causally over `[image tokens, text tokens]`, so every text token can see the whole
image while text tokens remain autoregressive.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VLMConfig, VisionConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # (b, h, t, hd)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal and attn_mask is None,
        )
        out = out.transpose(1, 2).reshape(b, t, -1)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = Attention(d_model, num_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), attn_mask, is_causal))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Images -> patch tokens via a strided convolution."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.proj = nn.Conv2d(
            cfg.in_channels, cfg.d_model, kernel_size=cfg.patch_size, stride=cfg.patch_size
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.proj(images)  # (b, d, h/p, w/p)
        return x.flatten(2).transpose(1, 2)  # (b, n_patches, d)


class VisionEncoder(nn.Module):
    """Small ViT-style encoder with a learned position embedding."""

    def __init__(self, cfg: VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.num_patches, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            Block(cfg.d_model, cfg.num_heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.depth)
        )
        self.norm = RMSNorm(cfg.d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images) + self.pos_embed
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, is_causal=False)
        return self.norm(x)


class TinyVLM(nn.Module):
    def __init__(self, cfg: VLMConfig):
        super().__init__()
        self.cfg = cfg
        v, t = cfg.vision, cfg.text

        self.vision = VisionEncoder(v)
        self.projector = nn.Linear(v.d_model, t.d_model, bias=False)
        self.vision_pos = nn.Parameter(torch.zeros(1, v.num_patches, t.d_model))

        self.token_embedding = nn.Embedding(t.vocab_size, t.d_model, padding_idx=cfg.pad_token_id)
        self.text_pos = nn.Embedding(t.max_seq_len, t.d_model)
        self.modality = nn.Embedding(2, t.d_model)

        self.blocks = nn.ModuleList(
            Block(t.d_model, t.num_heads, t.mlp_ratio, t.dropout) for _ in range(t.depth)
        )
        self.norm = RMSNorm(t.d_model)
        self.lm_head = nn.Linear(t.d_model, t.vocab_size, bias=False)
        if t.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.image_token_id = cfg.image_token_id
        self.apply(self._init_weights)
        # `nn.Parameter`s are not visited by `apply`, so init them explicitly.
        for param in (self.vision.pos_embed, self.vision_pos):
            nn.init.normal_(param, mean=0.0, std=0.02)

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
        elif isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------ helpers

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images once so generation can reuse the result.

        Args:
            pixel_values: (b, n_images, c, h, w) or (b, c, h, w)

        Returns:
            (b, n_images * n_patches, text_dim) projected image tokens.
        """
        if pixel_values.dim() == 4:
            pixel_values = pixel_values.unsqueeze(1)
        b, n, c, h, w = pixel_values.shape
        feats = self.vision(pixel_values.reshape(b * n, c, h, w))
        feats = self.projector(feats)
        feats = feats + self.vision_pos[:, : feats.shape[1]].to(feats.dtype)
        return feats.reshape(b, n * feats.shape[1], feats.shape[2])

    def _text_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        if t > self.cfg.text.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len {self.cfg.text.max_seq_len}")
        pos = torch.arange(t, device=input_ids.device)
        return (
            self.token_embedding(input_ids)
            + self.text_pos(pos).unsqueeze(0)
            + self.modality.weight[0]
        )

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        vision_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool = True,
    ):
        """Args:
        input_ids: (b, t) text token ids.
        pixel_values: (b, n_images, c, h, w); ignored when `vision_embeds` is given.
        vision_embeds: precomputed output of :meth:`encode_images`.
        attention_mask: (b, t) with 1 for real text tokens, 0 for padding.
        labels: (b, t) text labels aligned with `input_ids`; -100 to ignore.

        Returns a dict with `logits`, optional `loss`, and `n_image_tokens`.
        """
        text = self._text_embeds(input_ids)

        if vision_embeds is None and pixel_values is not None:
            vision_embeds = self.encode_images(pixel_values)
        if vision_embeds is not None:
            vision_embeds = vision_embeds.to(text.dtype) + self.modality.weight[1]
            n_image_tokens = vision_embeds.shape[1]
            x = torch.cat([vision_embeds, text], dim=1)
        else:
            n_image_tokens = 0
            x = text

        total = x.shape[1]

        # Without padding we only need a causal mask, which SDPA applies natively and
        # cheaply. Padding additionally requires masking out padded keys, which forces a
        # materialised (b, 1, t, t) boolean mask. `True` means "may attend".
        attn_mask = None
        if attention_mask is not None:
            keep = torch.ones(x.shape[0], total, dtype=torch.bool, device=x.device)
            keep[:, n_image_tokens:] = attention_mask.to(torch.bool)
            causal = torch.ones(total, total, dtype=torch.bool, device=x.device).tril()
            attn_mask = keep[:, None, None, :] & causal

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, is_causal=True)
        x = self.norm(x)
        logits = self.lm_head(x[:, n_image_tokens:])

        out = {"logits": logits, "n_image_tokens": n_image_tokens}
        if labels is not None:
            out["loss"] = self.loss(logits, labels)
        return out if return_dict else tuple(out.values())

    def loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Next-token cross entropy over shifted logits (image prefix excluded)."""
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.cfg.ignore_index,
        )

    # ----------------------------------------------------------------- generate

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        vision_embeds: torch.Tensor | None = None,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 50,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Greedy/top-k sampling. Recomputes the prefix each step (no KV cache)."""
        self.eval()
        generated = input_ids
        for _ in range(max_new_tokens):
            if generated.shape[1] > self.cfg.text.max_seq_len:
                break
            logits = self.forward(generated, pixel_values, vision_embeds)["logits"][:, -1, :]
            if temperature <= 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    k = min(top_k, logits.size(-1))
                    threshold = logits.topk(k, dim=-1).values[:, -1, None]
                    logits = logits.masked_fill(logits < threshold, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_id], dim=1)
            if eos_token_id is not None and bool((next_id == eos_token_id).all()):
                break
        return generated


@dataclass
class ParamStats:
    total: int
    vision: int
    text: int

    def size_mb(self, bytes_per_param: float) -> float:
        return self.total * bytes_per_param / (1024**2)


def count_parameters(model: TinyVLM) -> ParamStats:
    vision = sum(p.numel() for p in model.vision.parameters())
    vision += sum(p.numel() for p in model.projector.parameters())
    vision += model.vision_pos.numel()
    total = sum(p.numel() for p in model.parameters())
    return ParamStats(total=total, vision=vision, text=total - vision)
