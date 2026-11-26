"""Regression checks for the generation and evaluation paths used by PR #17."""

import json
import math
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from minillm.checkpoint import CheckpointError
from minillm.data import PackedPretrainDataset
from minillm.evaluation import EvaluationConfig
from minillm.evaluation_contamination import contamination_audit
from minillm.evaluation_windows import evaluate_indexed_windows, representative_windows
from minillm.language_correctness import verify_training_fixture
from minillm.language_diagnostics import verify_historical_greedy
from minillm.language_memorization import original_corpus, run_memorization
from minillm.language_token_analysis import expansion_budget
from minillm.model import MiniLLM
from minillm.train import DIAGNOSTIC_ONLY, load_model_checkpoint, save_model_checkpoint


def test_top_p_keeps_crossing_token_and_composes_with_top_k(monkeypatch):
    def inspect(probabilities, num_samples, generator):
        assert num_samples == 1
        return probabilities
    monkeypatch.setattr(torch, "multinomial", inspect)
    logits = torch.tensor([[2., 4., 1., 3.]]).log()
    got = MiniLLM._sample(logits, 1., 0, None, top_p=.6)
    torch.testing.assert_close(got, torch.tensor([[0., 4/7, 0., 3/7]]))
    got = MiniLLM._sample(logits, 1., 0, None, top_p=.01)
    assert got.tolist() == [[0., 1., 0., 0.]]
    got = MiniLLM._sample(logits, 1., 2, None, top_p=.55)
    assert got.tolist() == [[0., 1., 0., 0.]]


def test_repetition_penalty_is_sign_aware_once_per_visible_token():
    scores = torch.tensor([[2., -2., 3., -3.]])
    history = torch.tensor([[0, 0, 1, 2]])
    mask = torch.tensor([[1, 1, 1, 0]])
    penalized = MiniLLM._penalize_repetition(scores, history, 1.2, mask)
    torch.testing.assert_close(penalized, torch.tensor([[2/1.2, -2*1.2, 3., -3.]]))
    assert scores.tolist() == [[2., -2., 3., -3.]]


def test_sampling_is_seeded_and_default_greedy_matches_argmax(nano_cfg):
    torch.manual_seed(12)
    model = MiniLLM(nano_cfg).eval()
    inputs = torch.tensor([[24, 25, 26]])
    samples = []
    for seed in (42, 43, 44):
        def generate():
            return model.generate(inputs, max_new_tokens=10, temperature=.8, top_p=.9,
                                  repetition_penalty=1.1, generator=torch.Generator().manual_seed(seed))
        first = generate()
        assert torch.equal(first, generate())
        samples.append(tuple(first[0].tolist()))
    assert len(set(samples)) == 3

    expected = inputs
    with torch.inference_mode():
        for _ in range(5):
            expected = torch.cat([expected, model(expected).logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    assert torch.equal(model.generate(inputs, max_new_tokens=5), expected)
    assert torch.equal(model.generate(inputs, max_new_tokens=5, top_p=1., repetition_penalty=1.), expected)


def test_masked_prompt_keys_remain_hidden_during_cached_and_uncached_decode(nano_cfg):
    torch.manual_seed(33)
    model = MiniLLM(nano_cfg).eval()
    inputs = torch.tensor([[11, 12, 13, 14]])
    changed = torch.tensor([[11, 21, 13, 22]])
    mask = torch.tensor([[1, 0, 1, 0]])
    with torch.inference_mode():
        initial = model(inputs, use_cache=True, attention_mask=mask)
        token = torch.tensor([[17]])
        total_mask = torch.cat([mask, torch.ones_like(token)], dim=1)
        cached = model(token, past_key_values=initial.past_key_values, attention_mask=total_mask).logits
        full = model(torch.cat([changed, token], dim=1), attention_mask=total_mask).logits[:, -1:]
        torch.testing.assert_close(cached, full, atol=1e-6, rtol=1e-5)
    outputs = [model.generate(prompt, max_new_tokens=5, attention_mask=mask, use_cache=cache)[0, 4:]
               for prompt in (inputs, changed) for cache in (True, False)]
    assert all(torch.equal(outputs[0], row) for row in outputs[1:])
    assert outputs[0][0] == initial.logits[0, 2].argmax()


@pytest.mark.parametrize("use_cache", [True, False])
def test_padding_keeps_every_generation_step_at_the_same_rotary_position(nano_cfg, monkeypatch, use_cache):
    torch.manual_seed(42)
    model = MiniLLM(nano_cfg).eval()
    captured = []
    def greedy(logits, temperature, top_k, generator):
        captured.append(logits.clone())
        return logits.argmax(dim=-1, keepdim=True)
    monkeypatch.setattr(model, "_sample", greedy)

    def decode(ids, mask=None):
        captured.clear()
        result = model.generate(torch.tensor(ids), attention_mask=None if mask is None else torch.tensor(mask),
                                max_new_tokens=5, use_cache=use_cache)
        return result[:, len(ids[0]):], torch.stack(captured)

    first, first_logits = decode([[11, 13, 17]])
    second, second_logits = decode([[19, 23]])
    expected = torch.cat([first, second])
    expected_logits = torch.cat([first_logits, second_logits], dim=1)
    for ids, masks in (([[0, 0, 11, 13, 17], [0, 0, 0, 19, 23]], [[0, 0, 1, 1, 1], [0, 0, 0, 1, 1]]),
                       ([[11, 13, 17, 0, 0], [19, 23, 0, 0, 0]], [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])):
        padded, padded_logits = decode(ids, masks)
        torch.testing.assert_close(padded_logits, expected_logits, atol=1e-6, rtol=1e-5)
        assert torch.equal(padded, expected)


@pytest.mark.parametrize("change", [None, "id", "duplicate", "generated_token_ids", "stop_reason",
                                    "stop_token_id", "generation_budget", "seed", "top_p", "penalty"])
def test_historical_greedy_gate_requires_identical_tasks_policy_and_stops(change):
    config = EvaluationConfig()
    manifest = {"generation": asdict(config), "comparison_policy": {"config": asdict(config)},
                "files": {"predictions.jsonl": {"sha256": "historical", "bytes": 123}}}
    task = {"id": "first", "generation_mode": "raw", "text_category": "zh", "prompt_token_ids": [11],
            "generation_budget": 64, "stop_ids": [2], "generated_token_ids": [13, 2],
            "stop_reason": "eos", "stop_token_id": 2}
    tasks = [task, {**task, "id": "second"}]
    current = [row.copy() for row in reversed(tasks)]
    strategy, penalty = {"temperature": 0.0, "top_k": 0, "top_p": 1.0}, 1.0
    if change == "seed":
        config = replace(config, seed=43)
    elif change == "top_p":
        strategy[change] = 0.9
    elif change == "penalty":
        penalty = 1.1
    elif change == "duplicate":
        current[1]["id"] = current[0]["id"]
    elif change is not None:
        current[0][change] = None if change != "id" else "unexpected"

    if change is None:
        result = verify_historical_greedy(tasks, tasks, current, manifest, config, strategy, penalty)
        assert result["status"] == "PASS" and result["matched_tasks"] == 2
    else:
        with pytest.raises(ValueError, match="Greedy"):
            verify_historical_greedy(tasks, tasks, current, manifest, config, strategy, penalty)


def test_cross_entropy_matches_known_values_and_full_1024_window(tmp_path, nano_cfg):
    model = MiniLLM(nano_cfg)
    path = tmp_path / "train.bin"
    np.resize(np.array([0, 1, 2, 3], dtype=np.uint16), 1025).tofile(path)
    sample = PackedPretrainDataset(path, 1024)[0]
    assert sample["input_ids"].shape == (1024,)
    loss, count = model.loss_terms(torch.zeros(1, 1024, 4), sample["labels"][None])
    assert count == 1023
    assert float(loss) == pytest.approx(1023 * math.log(4), rel=1e-6)
    logits = torch.tensor([[[1., 6., 2., 1.], [10., 1., 4., 2.], [0., 0., 0., 0.]]]).log()
    unequal, count = model.loss_terms(logits, torch.tensor([[3, 1, 2]]))
    assert count == 2
    assert float(unequal) == pytest.approx(-math.log(.6) - math.log(4/17), abs=2e-7)


def test_manual_loss_grad_accumulation_causal_rope_cache_and_eos():
    result = verify_training_fixture()
    assert result["shift_mask"]["effective_targets"] == 2
    assert result["accumulation"]["effective_targets"] == 5
    assert result["attention"]["future_token_max_delta"] <= 3e-5
    assert result["attention"]["cache_greedy_equal"]
    assert result["attention"]["eos_stop"]["status"] == "PASS"
    assert result["attention"]["context_limit_checks"] == "PASS"


def test_representative_windows_attribute_cross_category_labels(tmp_path, nano_cfg):
    path = tmp_path / "train.bin"
    np.arange(11, 24, dtype=np.uint16).tofile(path)
    data = PackedPretrainDataset(path, 4)
    documents = [{"category": "zh_general", "tokens": 2},
                 {"category": "zh_encyclopedic", "tokens": 3},
                 {"category": "en_general", "tokens": 8}]
    plan = representative_windows(documents, data, seed=42, token_budget=6)
    assert plan == representative_windows(documents, data, seed=42, token_budget=6)
    assert plan["window_indices"][0] == 0 and len(plan["window_indices"]) == 2
    assert plan["by_window"][0] == {"zh_general": 1, "zh_encyclopedic": 2}
    assert plan["sampled_targets_by_category"] == {"zh_general": 1, "zh_encyclopedic": 2, "en_general": 3}
    assert len(plan["window_indices_sha256"]) == 64
    with pytest.raises(ValueError, match="did not cover"):
        representative_windows(documents, data, seed=42, token_budget=3)
    model = MiniLLM(nano_cfg).eval()
    result = evaluate_indexed_windows(model, data, documents, plan, torch.device("cpu"), 2)
    with torch.inference_mode():
        inputs = data[0]["input_ids"].unsqueeze(0)
        losses = F.cross_entropy(model(inputs).logits[0, :-1], inputs[0, 1:], reduction="none")
    assert result["by_category"]["zh_general"]["loss"] == pytest.approx(float(losses[0]), abs=1e-6)
    assert result["by_category"]["zh_encyclopedic"]["loss"] == pytest.approx(float(losses[1:].mean()), abs=1e-6)
    assert result["loss_sum"] == pytest.approx(sum(row["loss_sum"] for row in result["by_category"].values()))


def test_prefix_observations_keep_formal_audit_and_have_no_gold_answer(tmp_path):
    prefix, exact = "清晨风很大，", "观察记录已经送到仓库。"
    for split in ("train", "val", "test"):
        rows = [{"text": "当时" + prefix + "船还留在岸边。"}, {"text": exact}] if split == "train" else []
        (tmp_path / f"{split}.text.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (tmp_path / "train.selected.jsonl").write_text("", encoding="utf-8")
    report = contamination_audit([{"id": "formal", "prompt": exact}], dataset=tmp_path,
                                 fixed_prompts=[{"id": "fixed", "prompt": prefix}])
    summary = report["sources"][0]
    assert summary["exact_task_ids"] == ["formal"]
    assert summary["fixed_prefix_observations"] == [
        {"id": "fixed", "prefix_occurrence_records": 1, "full_record_equals_prefix": 0}]
    assert report["fixed_prefix_policy"]["answer_leakage"] == "UNASSESSED_BY_PREFIX_OVERLAP"
    assert report["sources"][-1]["kind"] == "pretrain/train_selected"


def test_diagnostic_corpus_and_checkpoints_are_separate(tmp_path, tiny_tokenizer, nano_cfg):
    rows = original_corpus()
    assert rows == original_corpus()
    assert len({row["prefix"] for row in rows}) == len(rows) == 120
    assert sum(row["kind"] == "passage" for row in rows) == 30
    occupied = tmp_path / "run"
    occupied.mkdir()
    with pytest.raises(ValueError, match="fresh"):
        run_memorization(tmp_path / "unused.json", tmp_path / "report", occupied, device=torch.device("cpu"))
    checkpoint = tmp_path / "probe.pt"
    save_model_checkpoint(checkpoint, MiniLLM(nano_cfg), tiny_tokenizer, step=0, tokens=0,
                          val_loss=None, extra={"scope": DIAGNOSTIC_ONLY})
    with pytest.raises(CheckpointError, match=DIAGNOSTIC_ONLY):
        load_model_checkpoint(checkpoint, torch.device("cpu"))


def test_expansion_preview_respects_category_ceilings_and_existing_approval():
    ratios = {"zh_general": .45, "zh_encyclopedic": .30, "zh_education": .15,
              "en_general": .10, "math_logic_code": 0.0}
    order = [{"document_id": f"{category}-{doc}", "category": category, "tokens": 5}
             for category in list(ratios)[:4] for doc in range(20)]
    result = expansion_budget(order, ratios, 8, batch_size=1, grad_accum=2, seed=42, tolerance=.07)
    assert result["limiting_category"] == "zh_general"
    preview = result["max_feasible_seeded_full_update_preview"]
    assert preview["planned_effective_tokens"] <= result["absolute_raw_supply_upper_bound"]
    assert preview["planned_effective_tokens"] == preview["sampled_training_mix"]["effective_tokens"]
    assert result["strict_exact_ratio_bounds"]["full_update_raw_upper_bound"] == 140
    assert preview["planned_effective_tokens"] == 182
    assert result["requests"]["2000000"]["preview"]["status"] == "PACKING_INFEASIBLE"
    assert not result["approved_trainable_extension_tokens"]


def test_expansion_budget_preserves_sub_percent_ratios_and_integral_limits():
    ratios = {"zh_general": .4525, "zh_encyclopedic": .30, "zh_education": .145,
              "en_general": .10, "math_logic_code": .0025}
    order = [{"document_id": category, "category": category, "tokens": 100} for category in ratios]
    report = expansion_budget(order, ratios, 8, batch_size=1, grad_accum=2, seed=42, tolerance=0)
    rounded = report["requests"]["2000000"]
    assert report["exact_allocation"]["math_logic_code"] == "1/400"
    assert report["absolute_raw_supply_upper_bound"] == 220
    assert rounded["target_by_category"]["math_logic_code"] == (rounded["rounded_effective_tokens"] + 399) // 400
    assert report["strict_exact_ratio_bounds"]["minimum_integral_ratio_full_update_tokens"] == 2800
    assert report["strict_exact_ratio_bounds"]["full_update_raw_upper_bound"] == 0
    assert report["max_feasible_seeded_full_update_preview"] is None
