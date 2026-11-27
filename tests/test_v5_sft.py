"""PR #22 regressions for mask boundaries, frozen probes, and interrupted epoch replay."""
import copy
import os
import signal

import pytest
import torch

from minillm import train as trainer
from minillm.checkpoint import load_training_state
from minillm.config import MINILLM_NANO
from minillm.model import MiniLLM
from minillm.quantization_provenance import verify_quantized_tensors
from minillm.quantize import quantize_model
from minillm.tokenizer import END_OF_TURN_ID
from minillm.train import TrainingOptions, fit
from minillm.training_milestones import require_identical_weights
from minillm.v5_sft_data import StrictChatDataset, check_overlap, historical_tasks, write_rows
from minillm.v5_sft_evaluation import knowledge_specs


def example(key, *turns):
    return {"id": key, "topic_category": "日常生活对话",
            "messages": [{"role": ("user", "assistant")[i % 2], "content": content}
                         for i, content in enumerate(turns)]}


def test_strict_sft_masks_prefix_and_stops(tiny_tokenizer):
    row = example("a", "哪里有阳光？", "窗边有阳光。", "我能在那里看书吗？", "可以在窗边看书。")
    data = StrictChatDataset("unused", tiny_tokenizer, 512, rows=[row])
    ids, labels = (data[0][key].tolist() for key in ("input_ids", "labels"))
    expected = [-100] * len(ids)
    for turn in (1, 3):
        prefix = tiny_tokenizer.apply_chat_template(row["messages"][:turn], add_generation_prompt=True)
        assert ids[:len(prefix)] == prefix
        end = ids.index(END_OF_TURN_ID, len(prefix))
        expected[len(prefix):end + 1] = ids[len(prefix):end + 1]
    assert labels == expected
    with pytest.raises(ValueError, match="truncation"):
        StrictChatDataset("unused", tiny_tokenizer, len(ids) - 1, rows=[row])


def test_historical_sources_and_near_overlap():
    tasks, world = historical_tasks()
    assert len(tasks) == 300
    assert len(list(knowledge_specs(world))) == 97
    history = [{"id": "history", "prompt": "秋天的早晨为什么容易见到露水？"}]
    almost = example("a", "在秋天的早晨为什么容易见到露水？", "早晨温度低。")
    assert check_overlap([almost], history) == [["historical:history", "a"]]


@pytest.mark.parametrize("location", ("codes", "scales", "fp16"))
def test_quantization_provenance_rejects_tensor_changes(location):
    model = MiniLLM(MINILLM_NANO)
    payload = quantize_model(model)
    verify_quantized_tensors(payload, model)
    corrupted = copy.deepcopy(payload)
    if location == "fp16":
        tensor = next(iter(corrupted["fp16"].values()))
    else:
        tensor = next(iter(corrupted["quantized"].values()))[location]
    flat = tensor.flatten()
    flat[0] = int(flat[0]) ^ 1 if location == "codes" else float(flat[0]) + 1
    with pytest.raises(ValueError, match="quantized source mismatch"):
        verify_quantized_tensors(corrupted, model)


def test_strict_sft_epoch_tail_and_resume(tmp_path, tiny_tokenizer, monkeypatch):
    rows = [example("a", "天空是什么颜色？", "天空是蓝色的。"),
            example("b", "窗户要关吗？", "下雨了，先关窗吧。"),
            example("c", "春天到了吗？", "院子里的花开了。")]
    path = tmp_path / "data.jsonl"
    write_rows(path, rows)
    dataset = StrictChatDataset(path, tiny_tokenizer, 128)
    cfg = MINILLM_NANO.copy()
    cfg.vocab_size = tiny_tokenizer.vocab_size
    torch.manual_seed(42)
    initial = MiniLLM(cfg)

    def run(name, resume=None):
        out = tmp_path / name
        model = copy.deepcopy(initial)
        opts = TrainingOptions(out=str(out), epochs=2, batch_size=1, grad_accum=2,
                               save_every=1, val_every=2, log_every=1, device="cpu",
                               precision="fp32", threads=1, grad_checkpointing=True,
                               milestone_steps=(0, 2, 4), validate_before_milestone=True, cosine_final_update=True,
                               provenance={"test": "sft full replay"},
                               resume=str(out / resume) if resume else None)
        return fit(opts, model, tiny_tokenizer, dataset, dataset, mode="sft")

    run("continuous")
    original = trainer._train_step
    count = 0
    def stop_at_epoch(*args, **kwargs):
        nonlocal count
        result = original(*args, **kwargs)
        count += 1
        if count == 2:
            os.kill(os.getpid(), signal.SIGTERM)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(trainer, "_train_step", stop_at_epoch)
        assert run("resumed")["stop_reason"] == "SIGTERM"
    assert run("resumed", "state.prev.pt")["stop_reason"] == "target_reached"
    uninterrupted = load_training_state(tmp_path / "continuous/state.pt")
    resumed = load_training_state(tmp_path / "resumed/state.pt")
    assert resumed["samples_seen"] == 6
    assert resumed["tokens_seen"] == 2 * dataset.stats["supervised_tokens"]
    assert resumed["history"] == uninterrupted["history"]
    require_identical_weights(uninterrupted["model"], resumed["model"], "SFT recovery")
    assert resumed["optimizer"]["param_groups"] == uninterrupted["optimizer"]["param_groups"]
    assert resumed["optimizer"]["param_groups"][0]["lr"] == resumed["manifest"]["training"]["min_learning_rate"]
    for key, slot in resumed["optimizer"]["state"].items():
        require_identical_weights(uninterrupted["optimizer"]["state"][key], slot, "SFT AdamW recovery")
    for name in ("python",):
        assert resumed["rng"][name] == uninterrupted["rng"][name]
    assert torch.equal(resumed["rng"]["torch"], uninterrupted["rng"]["torch"])
