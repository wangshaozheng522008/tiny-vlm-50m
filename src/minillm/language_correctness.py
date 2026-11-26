"""Numerical checks with small fixtures and the real 87M Pilot weights."""

import copy
import math

import torch

from .config import get_config
from .data import collate_batch
from .language_generation_analysis import verify_attention
from .model import MiniLLM
from .train import TrainingOptions, _train_step


@torch.inference_mode()
def verify_shift(model, ids, mask=None, labels=None):
    device = next(model.parameters()).device
    inputs = torch.tensor([ids], device=device)
    labels = inputs.clone() if labels is None else torch.tensor([labels], device=device)
    mask = None if mask is None else torch.tensor([mask], device=device)
    observed = model(inputs, labels=labels, attention_mask=mask)
    positions = [i for i in range(len(ids) - 1)
                 if int(labels[0, i + 1]) != model.cfg.ignore_index
                 and (mask is None or bool(mask[0, i] and mask[0, i + 1]))]
    logits = observed.logits[0, positions].double()
    targets = labels[0, [i + 1 for i in positions]]
    expected = float((logits.logsumexp(-1) - logits.gather(1, targets[:, None]).squeeze(1)).sum())
    assert observed.num_loss_tokens == len(positions)
    assert math.isclose(float(observed.loss_sum), expected, abs_tol=2e-5, rel_tol=1e-6)
    return {"input_tokens": len(ids), "effective_targets": len(positions),
            "manual_loss_sum": expected, "model_loss_sum": float(observed.loss_sum),
            "mean_loss": float(observed.loss), "status": "PASS"}


def verify_training_fixture():
    """Check uneven micro-batch loss normalization against one full-batch SGD update."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        cfg = get_config("minillm-nano")
        cfg.max_position_embeddings = 16
        model = MiniLLM(cfg)
        reference = copy.deepcopy(model)
        first = {"input_ids": torch.tensor([11, 13, 17, 19]),
                 "labels": torch.tensor([11, 13, -100, 19])}
        second = {"input_ids": torch.tensor([23, 29, 31, 37, 41, 43]),
                  "labels": torch.tensor([23, 29, 31, 37, 41, 43])}
        batch_one, batch_two = collate_batch([first]), collate_batch([second])
        batch_two["attention_mask"][0, 2] = 0
        joined = collate_batch([first, second])
        joined["attention_mask"][1, 2] = 0
        shift = verify_shift(model, second["input_ids"].tolist(), [1, 1, 0, 1, 1, 1],
                             [23, 29, 31, 37, -100, 43])

        opts = TrainingOptions(lr=1e-3, grad_clip=1000)
        initial = model.embed_tokens.weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr)
        result = _train_step(model, [batch_one, batch_two], optimizer, None, torch.device("cpu"), None, opts, opts.lr)
        ref = reference(**joined)
        ref.loss.backward()
        torch.optim.SGD(reference.parameters(), lr=opts.lr).step()
        assert result.tokens == ref.num_loss_tokens == 5
        for param, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(param, expected, atol=2e-7, rtol=2e-5)
        delta = float((model.embed_tokens.weight.detach() - initial).abs().max())
        assert delta > 0
        attention = verify_attention(model, [11, 13, 17, 19, 23, 29], steps=5)
        prompt = torch.tensor([[11, 13, 17]])
        with torch.inference_mode():
            eos = int(model.eval()(prompt).logits[0, -1].argmax())
            actual = model.generate(prompt, max_new_tokens=5, eos_token_id=eos)
        assert actual[0, 3:].tolist() == [eos]
        attention["eos_stop"] = {"status": "PASS", "fixture_eos_id": eos}
        return {"scope": "CPU MiniLLM Nano synthetic token IDs", "shift_mask": shift,
                "accumulation": {"status": "PASS", "effective_targets": result.tokens,
                                 "embedding_max_absolute_update": delta},
                "attention": attention}
