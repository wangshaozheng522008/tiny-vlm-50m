#!/usr/bin/env python3
"""Independent model review, semantic screening and quota selection for local SFT."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from generate_v5_sft_local import TEXT, UNITS_PER_BATCH, inference, table, until_valid
from minillm.assessment_io import read_json, records, report_directory
from minillm.corpus_utils import file_info, identity, verify_file
from minillm.training_manifest import write_json
from minillm.v5_sft_data import (CHECKS, CONFIG, FAMILIES, QUOTAS, ROOT, check_overlap, freeze,
                                historical_tasks, normalized, write_rows)
from minillm.v5_sft_experiment import base_assets

REVIEWER = "unsloth/Qwen3.5-4B-GGUF:Q4_K_M@e87f176479d0855a907a41277aca2f8ee7a09523"
REVIEWER_SHA256 = "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4"
REVIEWER_SOURCE = "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/e87f176479d0855a907a41277aca2f8ee7a09523"
REVIEW_SYSTEM = """审查简体中文聊天训练数据。逐条独立判断。dialogue 的第1句是用户，第2句是助手，后两句若有则为下一轮。
不要替模型改写原句。reason 用25字左右指出原句的关键细节和问题；确实答得好就指出实际完成了什么，而不是说“符合逻辑”。
检查顺序：1. 用户实际说的像聊天吗？“某人想提议……”这种写作说明是 style。2. 两句里的“我”是不同的人。
先分别判断哪一方借出、借入或执行动作；本该用户做、答案却说“我来还给你”，这种是 role。
3. 答案中的线路、楼栋标志、实时状态和其他人能否食用等，用户没提供就不能断言或猜测；这些是 fabrication。
同屋可以答应帮忙拿东西，但不能无证据说明陌生地址的具体门牌。天气、配料等缺失应明确说不确定。
4. 用户要求写一句，答案须实际给出正文；缺正文是 instruction。事实错是 factual，答错问题是 relevance。
多轮才检查上一句的指代和条件是否对应，错则 context。上述问题都没有，才填 none；答案不必复述用户每个词。
好例：用户“帮我把门打开一下？”助手“行，我去开。”正确承诺可以做的动作，issue=none。
坏例：用户“我把你借给我的笔放前台还你，行吗？”助手“可以，我会把它还你。”助手说自己还，issue=role。
坏例：用户“我能带这包给室友吃吗？不清楚里面有什么。”助手“他应该没事，带去吧。”无配料信息猜他人能吃，issue=fabrication。
fact_family、question_family、dialogue_family 各给一个具体简短的标签，近义问法对相同事实采用相同标签。"""
PAIR_SYSTEM = """盲审同一中文用户提问下两个模型的回答。参考答案是语义或指令参考，允许另一种正确的自然说法。
分别判断回答是否相关、语义正确、尊重格式条件、对信息不足处理得当。明显复读、乱答、角色错位或没有答到点都不通过。
若两者各有得失，可判平。给出两边具体理由。输出每题的编号、A和B是否达标以及偏好。"""
SIMILARITY_SYSTEM = """检查两条中文问题是否是同一件事的改写或询问同一具体事实。如果只共享宽泛主题，实际请求、
事实、对象或所需条件有实质区别，就不是同一问题。逐对返回重复判断和简短的区别理由。"""
BGE = "https://huggingface.co/BAAI/bge-small-zh-v1.5/tree/7999e1d3359715c523056ef9478215996d62a620"


def collect(generation, previous, work):
    output = work / "collected"
    if output.exists():
        receipt = read_json(output / "manifest.json")
        verify_file(output / "candidates.jsonl", receipt["candidates"])
        return output / "candidates.jsonl"
    profile = read_json(generation / "generation-profile.json")
    gathered, units = [], []
    for batch in profile["batch_plan"]:
        for unit in range(UNITS_PER_BATCH):
            path = generation / batch["id"] / f"{unit:02}.json"
            document = read_json(path)
            units.append(file_info(path))
            gathered.extend(document["rows"])
    prior = []
    for folder in sorted(previous.glob("*/manifest.json")):
        manifest = read_json(folder)
        path = folder.parent / "candidates.jsonl"
        verify_file(path, manifest["candidates"])
        prior.append({"batch": manifest["batch"], "manifest": file_info(folder)})
        gathered.extend(records(path))
    if not 8000 <= len(gathered) <= 10000 or len({r["id"] for r in gathered}) != len(gathered):
        raise ValueError(f"expected 8000–10000 uniquely identified candidates, found {len(gathered)}")
    usage = Counter()
    unknown = 0
    for path in sorted(generation.glob("*/*.*-*.json")):
        raw = read_json(path)
        if not raw.get("usage"):
            unknown += 1
            continue
        usage.update({key: raw["usage"][key] for key in ("prompt_tokens", "completion_tokens")})
        usage["requests_with_usage"] += 1
    with report_directory(output) as staged:
        write_rows(staged / "candidates.jsonl", gathered)
        write_json(staged / "manifest.json", {"generation": file_info(generation / "generation-profile.json"),
                   "units_sha256": identity(units), "prior_batches": prior, "new_usage": dict(usage),
                   "new_receipts_without_usage": unknown, "external_api_calls": 0,
                   "counts": dict(Counter(r["split"] for r in gathered)), "rows": len(gathered),
                   "candidates": file_info(staged / "candidates.jsonl")})
    return output / "candidates.jsonl"


def audit_group(directory, index, group, port):
    path = directory / f"{index:05}.json"
    batch_id = identity([(row["id"], row["message_sha256"]) for row in group])
    if path.exists():
        existing = read_json(path)
        if existing["binding"] != batch_id:
            raise ValueError("independent review group changed")
        return existing["decisions"]
    payload = [{"id": i, "category": r["topic_category"],
                "dialogue": [m["content"] for m in r["messages"]]} for i, r in enumerate(group, 1)]
    issues = ("none", "fabrication", "role", "instruction", "factual", "relevance", "style", "context")
    fields = {"id": {"type": "integer", "minimum": 1, "maximum": len(group)},
              "reason": TEXT, "issue": {"enum": list(issues)}, **{key: TEXT for key in FAMILIES}}
    def parse(receipt):
        results = json.loads(receipt["content"])["rows"]
        if len(results) != len(group) or {r["id"] for r in results} != set(range(1, len(group)+1)):
            raise ValueError("wrong review coverage")
        if any(not r["reason"].strip() or r["issue"] not in issues for r in results):
            raise ValueError("unsupported judgment")
        return sorted(results, key=lambda r: r["id"])

    verdicts = until_valid(path, path, lambda n: inference(port, REVIEW_SYSTEM, json.dumps(payload, ensure_ascii=False),
                           table(fields, len(group)), seed=42000+index*5+n, limit=750, temperature=.15), parse)
    decisions = [] if verdicts is None else [dict(r, candidate=row["id"], accept=r["issue"] == "none")
                                             for r, row in zip(verdicts, group, strict=True)]
    write_json(path, {"binding": batch_id, "model": REVIEWER, "decisions": decisions})
    return decisions


def independent_review(rows, directory, port, workers, weights):
    directory.mkdir(parents=True, exist_ok=True)
    weight_info = file_info(weights)
    if weight_info["sha256"] != REVIEWER_SHA256:
        raise ValueError("independent reviewer weights differ")
    profile = {"model": REVIEWER, "weights": weight_info, "system": REVIEW_SYSTEM,
               "model_source": REVIEWER_SOURCE,
               "input": identity([(r["id"], r["message_sha256"]) for r in rows]), "batch_size": 2}
    profile_path = directory / "profile.json"
    if profile_path.exists() and read_json(profile_path) != profile:
        raise ValueError("review profile changed")
    write_json(profile_path, profile)
    groups = [rows[i:i+2] for i in range(0, len(rows), 2)]
    decisions = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(lambda args: audit_group(directory, *args, port), enumerate(groups)), 1):
            decisions.update({row["candidate"]: row for row in result})
            if i % 20 == 0 or i == len(groups):
                progress = {"groups": i, "planned_groups": len(groups), "reviewed": len(decisions),
                            "accepted": sum(row["accept"] for row in decisions.values())}
                write_json(directory / "progress.json", progress)
                print(progress, flush=True)
    return decisions


def calibrate_reviewer(previous, work, port, workers, weights):
    source = previous / "qwen-train-001/candidates.jsonl"
    # These labels come from the recorded independent review of the actual pilot.
    expected = {2: True, 4: False, 6: False, 10: True, 17: False,
                21: True, 35: True, 39: False, 54: True, 55: False}
    prefix = "qwen-train-001-"
    expected = {f"{prefix}{key:03}": value for key, value in expected.items()}
    rows = [row for row in records(source) if row["id"] in expected]
    if len(rows) != len(expected):
        raise ValueError("reviewer calibration controls are missing")
    directory = work / "review-calibration"
    judgments = independent_review(rows, directory, port, workers, weights)
    correct = Counter()
    for key, gold in expected.items():
        if key in judgments and judgments[key]["accept"] == gold:
            correct["pass" if gold else "reject"] += 1
    ready = correct["pass"] >= 4 and correct["reject"] >= 4
    receipt = {"status": "READY" if ready else "REJECTED", "controls": file_info(source),
               "weights": file_info(weights), "system_sha256": identity(REVIEW_SYSTEM),
               "expected": expected, "correct": dict(correct), "judgments": judgments}
    write_json(directory / "result.json", receipt)
    if not ready:
        raise RuntimeError("independent reviewer did not pass calibration controls")


def semantic_screen(rows, history, model_dir, out):
    import torch
    from transformers import AutoModel, AutoTokenizer

    checkpoints = {p.name: file_info(p) for p in model_dir.iterdir() if p.is_file()}
    texts = [h["prompt"] for h in history]
    texts.extend(" ".join(m["content"] for m in row["messages"] if m["role"] == "user") for row in rows)
    binding = {"source": BGE, "files": checkpoints, "input": identity(texts),
               "historical_threshold": .91, "cross_split_threshold": .93, "same_split_threshold": .962,
               "historic_review_min": .75, "cross_split_review_min": .79}
    if out.exists():
        prior = read_json(out)
        if prior["binding"] != binding:
            raise ValueError("semantic screening identity changed")
        return prior
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    vectors = []
    with torch.inference_mode():
        for i in range(0, len(texts), 64):
            inputs = tokenizer(texts[i:i+64], padding=True, truncation=True, max_length=128, return_tensors="pt")
            output = model(**inputs.to(device)).last_hidden_state[:, 0]
            vectors.append(torch.nn.functional.normalize(output, dim=-1))
        embedding = torch.cat(vectors)
        candidates = embedding[len(history):]
        historical = (candidates @ embedding[:len(history)].T).max(dim=1)
        removed = {rows[i]["id"]: [history[j]["id"], score] for i, (score, j) in enumerate(
            zip(historical.values.tolist(), historical.indices.tolist())) if score >= binding["historical_threshold"]}
        borderline = [[f"historical:{history[j]['id']}", rows[i]["id"], score]
                      for i, (score, j) in enumerate(zip(historical.values.tolist(), historical.indices.tolist()))
                      if binding["historic_review_min"] <= score < binding["historical_threshold"]]
        edges = []
        split_codes = torch.tensor([("train", "val", "test").index(row["split"]) for row in rows], device=device)
        for start in range(0, len(rows), 128):
            block = candidates[start:start+128] @ candidates.T
            pairs = (block >= binding["cross_split_threshold"]).nonzero().tolist()
            for i, j in pairs:
                left = start + i
                if left >= j:
                    continue
                cross = rows[left]["split"] != rows[j]["split"]
                score = float(block[i, j])
                if cross or score >= binding["same_split_threshold"]:
                    edges.append([rows[left]["id"], rows[j]["id"], score])
            cross = block.masked_fill(split_codes[start:start+len(block), None] == split_codes[None, :], -1)
            nearest = cross.max(dim=1)
            for i, (score, j) in enumerate(zip(nearest.values.tolist(), nearest.indices.tolist()), start):
                if rows[i]["split"] != "train" and binding["cross_split_review_min"] <= score < binding["cross_split_threshold"]:
                    borderline.append([rows[i]["id"], rows[j]["id"], score])
    result = {"binding": binding, "historical_hits": removed, "edges": edges, "borderline": borderline}
    write_json(out, result)
    return result


def resolve_similarity(work, rows, history, semantic, port, workers):
    directory = work / "similarity-review"
    directory.mkdir(exist_ok=True)
    texts = {f"historical:{row['id']}": row["prompt"] for row in history}
    texts.update({row["id"]: " / ".join(m["content"] for m in row["messages"] if m["role"] == "user")
                  for row in rows})
    pairs = semantic["borderline"]
    write_json(directory / "profile.json", {"model": REVIEWER, "system": SIMILARITY_SYSTEM,
                                             "semantic": file_info(work / "semantic-screen.json")})

    def group_job(item):
        index, group = item
        path = directory / f"{index:04}.json"
        if path.exists():
            return read_json(path)["decisions"]
        query = [{"id": i, "first": texts[a], "second": texts[b]} for i, (a, b, _) in enumerate(group, 1)]
        fields = {"id": {"type": "integer"}, "duplicate": {"type": "boolean"}, "reason": TEXT}
        def parse(receipt):
            parsed = json.loads(receipt["content"])["rows"]
            if len(parsed) != len(group) or {v["id"] for v in parsed} != set(range(1, len(group)+1)):
                raise ValueError("incomplete semantic-pair coverage")
            return sorted(parsed, key=lambda v: v["id"])
        result = until_valid(path, path, lambda n: inference(port, SIMILARITY_SYSTEM,
                   json.dumps(query, ensure_ascii=False), table(fields, len(group)), seed=27000+index*5+n,
                   limit=360, temperature=.10), parse)
        decisions = [{"pair": raw, "judgment": judgment} for raw, judgment in zip(
                     group, result or [None] * len(group), strict=True)]
        write_json(path, {"decisions": decisions})
        return decisions

    groups = [pairs[i:i+6] for i in range(0, len(pairs), 6)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        decisions = [row for group in pool.map(group_job, enumerate(groups)) for row in group]
    write_json(work / "semantic-review.json",
               {"pairs": len(pairs), "reviewed": sum(d["judgment"] is not None for d in decisions),
                "unresolved_or_duplicate": [d["pair"] for d in decisions
                    if d["judgment"] is None or d["judgment"]["duplicate"]]})


def pick(rows, decisions, lexical, semantic):
    forbidden = {rid for rid, result in decisions.items() if not result["accept"]}
    forbidden.update(semantic["historical_hits"])
    edges = defaultdict(set)
    for a, b, *_ in [*lexical, *semantic["edges"], *semantic["resolved_exclusions"]]:
        if a.startswith("historical:"):
            forbidden.add(b)
        elif a == b:
            forbidden.add(a)
        else:
            edges[a].add(b)
            edges[b].add(a)
    by_split = defaultdict(list)
    for row in rows:
        rid = row["id"]
        if rid in decisions and rid not in forbidden:
            review = decisions[rid]
            canonical = {key: normalized(review[key]) for key in FAMILIES}
            if all(canonical.values()):
                by_split[row["split"]].append((row, canonical))
    owners, used, selected = {}, set(), {}
    remaining = {}
    for split in ("test", "val", "train"):
        pool = by_split[split]
        # Stable round robin across the batches preserves topic diversity.
        batch_counts = Counter()
        ranked = []
        for row, family in sorted(pool, key=lambda pair: (pair[0]["generation"]["batch"], identity(pair[0]["id"]))):
            batch = row["generation"]["batch"]
            batch_counts[batch] += 1
            ranked.append((batch_counts[batch], identity(row["id"]), row, family))
        ranked.sort(key=lambda entry: entry[:2])
        target = QUOTAS[split]
        budget = {(category, flag): round(n * .10) if flag else n - round(n * .10)
                  for category, n in target.items() for flag in (False, True)}
        for _, _, row, family in ranked:
            rid = row["id"]
            key = row["topic_category"], len(row["messages"]) > 2
            if (budget[key] <= 0 or edges[rid] & used
                    or any(owners.get((k, v), split) != split for k, v in family.items())):
                continue
            used.add(rid)
            selected[rid] = family
            budget[key] -= 1
            owners.update({(k, v): split for k, v in family.items()})
        remaining[split] = {f"{category}|{'multi' if flag else 'single'}": count
                            for (category, flag), count in budget.items() if count}
    return selected, remaining


def publish(work, raw, rows, decisions, semantic, frozen):
    historical, _ = historical_tasks()
    knowledge = read_json(ROOT / "configs/v5-sft-knowledge-probes.json")
    history = [*historical, *({"id": r["id"]+"-cloze", "prompt": r["cloze"]} for r in knowledge)]
    lexical = check_overlap(rows, history)
    selected, missing = pick(rows, decisions, lexical, semantic)
    audit = {"source_manifest": file_info(raw.parent / "manifest.json"),
             "review_profile": file_info(work / "review/profile.json"),
             "semantic_audit": file_info(work / "semantic-screen.json"),
             "semantic_pair_review": file_info(work / "semantic-review.json"),
             "human_review": "PENDING", "review_method": "distinct model judgment with archived evidence",
             "prefreeze_lexical_flags": len(lexical), "selected": len(selected), "unfilled": missing}
    write_json(work / "selection-status.json", audit)
    if any(missing.values()):
        raise ValueError(f"insufficient independently admitted examples; see {work / 'selection-status.json'}")
    evidence = "independent-model review receipt and lexical/BGE audit in the campaign"
    reviews = []
    for row in rows:
        rid = row["id"]
        if rid not in decisions:
            continue
        review = decisions[rid]
        if rid in selected:
            status = "PASS"
        else:
            status = "PENDING" if review["accept"] else "FAIL"
        reviews.append({"id": rid, "message_sha256": row["message_sha256"], "status": status,
                        "origin": "independent_model", "reviewer": REVIEWER,
                        "evidence": {"scope": evidence, "reason": review["reason"]},
                        "checks": {check: "PASS" for check in CHECKS} if status == "PASS" else {},
                        **{k: review[k] for k in FAMILIES}})
    path = work / "review-for-freeze.jsonl"
    if path.exists():
        if list(records(path)) != reviews:
            raise ValueError("immutable admission reviews changed")
    else:
        write_rows(path, reviews)
    settings = read_json(CONFIG)
    tokenizer, _, _ = base_assets(settings)
    freeze(raw, path, frozen, tokenizer, settings["max_seq_len"], inspected=audit)
    write_json(work / "freeze-audit.json", audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--reviewer", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18145)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--stage", choices=("calibrate", "collect", "review", "semantic", "semantic-review", "freeze"), required=True)
    args = parser.parse_args()
    if args.stage == "calibrate":
        calibrate_reviewer(args.previous, args.work, args.port, args.concurrency, args.reviewer)
        return
    raw = collect(args.work / "generation", args.previous, args.work)
    if args.stage == "collect":
        return
    rows = sorted(records(raw), key=lambda row: row["id"])
    if args.stage == "review":
        calibration = read_json(args.work / "review-calibration/result.json")
        if (calibration["status"] != "READY" or calibration["weights"] != file_info(args.reviewer)
                or calibration["system_sha256"] != identity(REVIEW_SYSTEM)):
            raise ValueError("independent review requires calibrated model and prompt")
        independent_review(rows, args.work / "review", args.port, args.concurrency, args.reviewer)
        return
    if args.stage in ("semantic", "semantic-review"):
        history, _ = historical_tasks()
        knowledge = read_json(ROOT / "configs/v5-sft-knowledge-probes.json")
        history += [{"id": r["id"]+"-cloze", "prompt": r["cloze"]} for r in knowledge]
        if args.stage == "semantic":
            semantic_screen(rows, history, args.embedding, args.work / "semantic-screen.json")
        else:
            resolve_similarity(args.work, rows, history, read_json(args.work / "semantic-screen.json"),
                               args.port, args.concurrency)
        return
    reviews = {}
    for path in sorted((args.work / "review").glob("[0-9][0-9][0-9][0-9][0-9].json")):
        reviews.update({row["candidate"]: row for row in read_json(path)["decisions"]})
    settings = read_json(CONFIG)
    semantic = read_json(args.work / "semantic-screen.json")
    semantic["resolved_exclusions"] = read_json(args.work / "semantic-review.json")["unresolved_or_duplicate"]
    publish(args.work, raw, rows, reviews, semantic, ROOT / settings["dataset"])


if __name__ == "__main__":
    main()
