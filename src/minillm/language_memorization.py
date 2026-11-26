"""Isolated 87M training-set memorization probe with original Chinese text."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from .config import get_config
from .data import EpochBatchSampler, collate_batch
from .evaluation import repetition
from .model import MiniLLM
from .tokenizer import EOS_ID, MiniLLMTokenizer
from .train import DIAGNOSTIC_ONLY, TrainingOptions, _train_step, lr_at, save_model_checkpoint
from .training_manifest import write_json
from .utils import set_seed
from .assessment_io import environment, publish_manifest, report_directory
from .corpus_utils import identity

SITES = (
    ("雾灯岛", "蓝羽站", "白瓷铃", "三号木柜", "斑尾鸥"),
    ("栗潮镇", "青瓦馆", "折纸船", "西侧竹架", "银脚鸭"),
    ("沙针湾", "石榴亭", "红铜尺", "东边书桌", "花背猫"),
    ("苔钟谷", "暖溪屋", "绿玻璃杯", "门边藤篮", "灰耳兔"),
    ("槐星村", "望雨楼", "松木盒", "二楼窗台", "青翅雀"),
    ("云碗岭", "渡月台", "橙色灯罩", "圆形矮桌", "黄尾犬"),
    ("竹帆港", "听潮所", "旧帆布包", "靠墙长凳", "黑鼻獭"),
    ("盐雀滩", "拾贝室", "锡制茶罐", "入口小台", "紫颈鸽"),
    ("芦镜坡", "叶纹铺", "绘图夹板", "高脚方凳", "长腿鹭"),
    ("柚霜原", "晨雾廊", "陶土水壶", "后门石阶", "短角鹿"),
)
SENTENCES = (
    ("{place}的值班表贴在入口处，", "周三清晨由修表师傅负责开门。"),
    ("管理员整理完{station}的工具，", "就把{object}放回{shelf}。"),
    ("{animal}听见院门轻轻一响，", "便钻进花盆后面的阴影里。"),
    ("雨停后大家走出{station}，", "沿着铺了碎石的小路去查看水位。"),
    ("客人询问{place}的路线，", "柜台上的地图画着一座窄桥和两排榆树。"),
    ("旧标签被风吹掉以后，", "她在{object}上系了一条紫色细绳。"),
    ("{station}的窗户朝着菜园，", "午后的阳光刚好照到一排刚发芽的豆苗。"),
    ("傍晚清点物品的时候，", "志愿者发现{object}仍然留在{shelf}上。"),
    ("{place}把旧车棚改成了等候区，", "墙上新装了三盏灯，坐在长椅上的人终于能看清车票了。"),
)
PASSAGES = (
    ("{station}每周记录一次院子里的降雨量。",
     "工作人员把一只直筒容器安放在远离屋檐的地方。记录员先查看水面高度，再把数字写进纸质表格。遇到刮风的日子，他们会检查底座，防止容器倾倒。收好表格之后，值班员把{object}放回{shelf}，并把当天读数抄在走廊的木板上。"),
    ("{place}制作了一份供游客使用的散步指南。",
     "从路口出发，沿水渠走到第一座石桥，再顺着矮墙往北转。休息处的长椅旁边留有一块空地，行人可以在那里系鞋带或整理背包。见到{animal}时可以站在远处观察，走完这一段路就能找到{station}。"),
    ("{station}开放了一间小小的修理室。",
     "居民可以携带松脱的拉链和掉了把手的抽屉来登记。接待员给物品挂上号码牌，然后核对需要修理的部位。当天修好后，工作人员会把取件时间写在卡片上。用于示范的{object}摆在{shelf}，方便新来的居民辨认材料。"),
)
SCOPE = DIAGNOSTIC_ONLY


def original_corpus():
    """90 short sentences and 30 explanatory passages with observed prefixes."""
    rows = []
    for index, values in enumerate(SITES):
        fields = dict(zip(("place", "station", "object", "shelf", "animal"), values))
        for kind, templates in (("sentence", SENTENCES), ("passage", PASSAGES)):
            for number, (prefix, suffix) in enumerate(templates):
                # Some template prefixes repeat across sites. Include the place
                # in those prompts so each measured suffix is unambiguous.
                if kind == "sentence" and number in (5, 7):
                    prefix = "在{place}，" + prefix
                rows.append({"id": f"{kind}-{index:02d}-{number}", "kind": kind,
                             "prefix": prefix.format(**fields), "suffix": suffix.format(**fields)})
    return rows


def empirical_accuracy_ceiling(items):
    """Identical causal prefixes can have different valid training targets."""
    contexts = defaultdict(Counter)
    for item in items:
        ids = item["input_ids"].tolist()
        for position in range(1, len(ids)):
            contexts[tuple(ids[:position])][ids[position]] += 1
    correct = sum(max(targets.values()) for targets in contexts.values())
    total = sum(sum(targets.values()) for targets in contexts.values())
    return {"maximum_correct": correct, "targets": total, "maximum_accuracy": correct / total}


@torch.inference_mode()
def measure(model, tokenizer, rows, items, device, batch_size):
    model.eval()
    loss, tokens, correct = 0.0, 0, 0
    for start in range(0, len(items), batch_size):
        batch = {key: value.to(device) for key, value in collate_batch(items[start:start + batch_size]).items()}
        result = model(**batch)
        labels = batch["labels"][:, 1:]
        correct += int((result.logits[:, :-1].argmax(-1) == labels).sum())
        loss += float(result.loss_sum)
        tokens += result.num_loss_tokens
    generations = []
    for row in rows:
        prompt, target = row["prompt_ids"], row["target_ids"]
        generated = model.generate(torch.tensor([prompt], device=device), max_new_tokens=len(target)+1,
                                   eos_token_id=EOS_ID)[0, len(prompt):].tolist()
        kept = generated[:-1] if generated[-1] == EOS_ID else generated
        text = tokenizer.decode(kept)
        punctuation_ids = [token for token in kept if token in row["punctuation_ids"]]
        generations.append({"id": row["id"], "prefix": row["prefix"], "target": row["suffix"],
                            "target_ids": target, "prompt_ids": prompt, "generated_ids": generated,
                            "text": text, "exact_text": text == row["suffix"],
                            "exact_tokens": kept == target, "ended_with_eos": generated[-1] == EOS_ID,
                            "punctuation_tokens": len(punctuation_ids),
                            "repeated_punctuation_tokens": sum(
                                left == right and left in row["punctuation_ids"]
                                for left, right in zip(kept, kept[1:])),
                            **repetition(text, kept)})
    punctuation_total = sum(row["punctuation_tokens"] for row in generations)
    return {"train_loss": loss / tokens, "teacher_forced_accuracy": correct / tokens,
            "teacher_forced_tokens": tokens,
            "exact_text_fraction": sum(row["exact_text"] for row in generations) / len(generations),
            "exact_token_fraction": sum(row["exact_tokens"] for row in generations) / len(generations),
            "adjacent_repeated_punctuation_fraction": (
                sum(row["repeated_punctuation_tokens"] for row in generations) / max(punctuation_total, 1)),
            "mean_token_4gram_repeat_ratio": sum(row["token_4gram_repeat_ratio"] for row in generations) / len(generations),
            "generations": generations}


def run_memorization(tokenizer_path, out, run_dir, *, device, steps=500):
    from .language_token_analysis import vocabulary

    if not 1 <= steps <= 500:
        raise ValueError("diagnostic updates must be between 1 and 500")
    out, run_dir = Path(out).resolve(), Path(run_dir).resolve()
    if out == run_dir or out in run_dir.parents or run_dir in out.parents:
        raise ValueError("use separate directories for diagnostic weights and reports")
    if run_dir.exists() or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
        raise ValueError("choose fresh diagnostic run and report directories")
    tokenizer = MiniLLMTokenizer.load(tokenizer_path)
    cfg = get_config("minillm-87m")
    tokenizer.assert_vocab_size(cfg.vocab_size)
    corpus = original_corpus()
    items, probes = [], []
    probe_ids = {f"{kind}-{site:02d}-{template}" for site, (kind, template) in enumerate(
        [("sentence", 0), ("passage", 2), ("sentence", 3), ("passage", 1), ("sentence", 4),
         ("passage", 2), ("sentence", 6), ("sentence", 5), ("passage", 1), ("sentence", 8)])}
    punctuation = {row["id"] for row in vocabulary(tokenizer) if row["class"] == "punctuation"}
    for row in corpus:
        tokens = tokenizer.encode_document(row["prefix"] + row["suffix"], category="zh")
        if not 2 <= len(tokens) <= 128 or any(token in tokenizer.special_ids.values() for token in tokens[:-1]):
            raise ValueError("invalid diagnostic document length or unexpected special token")
        ids = torch.tensor(tokens)
        items.append({"input_ids": ids, "labels": ids.clone()})
        if row["id"] in probe_ids:
            prompt = tokenizer.encode(row["prefix"], category="zh")
            target = tokens[len(prompt):-1]
            if tokens[:len(prompt)] != prompt or tokenizer.decode(target) != row["suffix"]:
                raise ValueError(f"{row['id']}: prefix is not a joint BPE boundary")
            probes.append({**row, "prompt_ids": prompt, "target_ids": target, "punctuation_ids": punctuation})
    if len({row["prefix"] for row in corpus}) != len(corpus):
        raise ValueError("diagnostic prefixes must be unique")
    if {row["id"] for row in probes} != probe_ids:
        raise ValueError("diagnostic probe IDs do not match the corpus")

    opts = TrainingOptions(batch_size=8, grad_accum=1, lr=3e-4, min_lr=3e-5,
                           seed=42, precision="fp32", max_steps=steps)
    run_dir.mkdir(parents=True, exist_ok=False)
    set_seed(opts.seed)
    model = MiniLLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr, betas=(0.9, 0.95),
                                 weight_decay=opts.weight_decay)
    sampler = EpochBatchSampler(len(items), opts.batch_size, opts.seed)
    metadata = {"scope": SCOPE, "seed": opts.seed, "model_config": cfg.to_dict(),
                "tokenizer_fingerprint": tokenizer.fingerprint, "corpus_sha256": identity(corpus),
                "documents": len(corpus), "raw_tokens": sum(len(item["input_ids"]) for item in items),
                "sequence_length_cap": 128, "batch_size": opts.batch_size, "max_updates": steps,
                "lr_peak": opts.lr, "lr_min": opts.min_lr, "warmup_steps": 10,
                "optimizer": "AdamW", "precision": opts.precision,
                "sampling": "repeat the original diagnostic corpus in seeded shuffled epochs",
                "scope_of_metrics": "all training sequences; fixed prompts are in-sample",
                "teacher_forced_accuracy_ceiling": empirical_accuracy_ceiling(items),
                "environment": environment(device)}
    write_json(run_dir / "DIAGNOSTIC_ONLY.json", metadata)
    tokens_seen, step, epoch = 0, 0, 0

    def observe():
        result = {"step": step, "tokens_seen": tokens_seen,
                  **measure(model, tokenizer, probes, items, device, opts.batch_size)}
        print(f"{SCOPE} step={step} tokens={tokens_seen} loss={result['train_loss']:.4f} "
              f"accuracy={result['teacher_forced_accuracy']:.4f} exact={result['exact_text_fraction']:.4f}", flush=True)
        with (run_dir / "measurements.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
        return result

    history = [observe()]
    while step < steps:
        for indices in sampler.batches(epoch):
            model.train()
            batch = collate_batch([items[i] for i in indices])
            lr = lr_at(step, steps, opts.lr, opts.min_lr, min(10, steps))
            for group in optimizer.param_groups:
                group["lr"] = lr
            update = _train_step(model, [batch], optimizer, None, device, None, opts, lr)
            tokens_seen += update.tokens
            step += 1
            if step % 25 == 0 or step == steps:
                history.append(observe())
            if step == steps:
                break
        epoch += 1
    save_model_checkpoint(run_dir / "diagnostic-model.pt", model, tokenizer,
                          step=step, tokens=tokens_seen, val_loss=None,
                          extra={"mode": SCOPE, "scope": SCOPE, "diagnostic_corpus_sha256": identity(corpus)})
    with report_directory(out) as directory:
        write_json(directory / "memorization.json", {"config": metadata, "history": history})
        write_json(directory / "original-corpus.json", corpus)
        publish_manifest(directory, "manifest.json", {"scope": SCOPE, "schema_version": 1})
    return history[-1]
