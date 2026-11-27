#!/usr/bin/env python3
"""Resumable, batched Chinese QA synthesis using a loopback llama.cpp process."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from generate_v5_sft_data import plan
from minillm.assessment_io import read_json
from minillm.corpus_utils import file_info, identity
from minillm.training_manifest import write_json
from minillm.v5_sft_data import FAMILIES, TRAIN, normalized, validate_messages

WEIGHTS_SHA256 = "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"
MODEL = "Qwen/Qwen3-4B-GGUF:Q4_K_M@bc640142c66e1fdd12af0bd68f40445458f3869b"
UNIT_SIZE = 5
UNITS_PER_BATCH = 100 // UNIT_SIZE
EXTRA_TOPICS = (
    ("train", 0, "走亲访友时的临时变化"), ("train", 0, "社区自习和共享活动"),
    ("train", 1, "山地水土及耕种"), ("train", 1, "传统日用工具和手艺"),
    ("train", 2, "整理家居时的实用原理"), ("train", 3, "按明确条件简化线下活动通知"),
    ("train", 4, "没有看到的个人物品状态"), ("train", 4, "缺少来源和时间的传闻"),
)
PLAN_SYSTEM = """为真实口语问答拟定具体而不同的写作素材。每条是一件具体小事、一个稳定事实或一个边界明确的任务。
只写素材和归类，不写最终问答。scope 写出独立可用的细节：常识/知识给可解释的稳定事实；指令给实际文字及条件；
不确定性给未知信息和合理核实方向；日常写出双方的处境和意图。相邻素材应有不同内容和理由，不用替换名词造出变体。
fact_family 写简短可核对的事实或情境标签，question_family 写独立的问法意图，dialogue_family 写细分场景。"""
CHAT_SYSTEM = """写供小模型学习的自然简体中文聊天。每条素材写独立的一段对话，不复述素材说明。
用户说的要像现实中的口语提问、请求或追问；助手直接回应，尽量1至3个短句，通常20至65个字，简短事实或指令可更短。
不要固定开场白、序号、角色标签、Markdown或多余总结。不要编造地点、人物的私有状态或当前情况；不清楚就说缺什么。
两轮样本的第二问要用到上文，并且有新信息或新问法。回答要和问题及上下文一致。
知识和常识只写稳定且容易核对的内容，避免脱离素材给出有争议的精确数字。指令题直接交付指定内容。"""
TEXT = {"type": "string", "minLength": 1}


def table(fields, size):
    row = {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}
    return {"type": "object", "properties": {"rows": {"type": "array", "items": row,
                                                        "minItems": size, "maxItems": size}},
            "required": ["rows"], "additionalProperties": False}


def inference(port, system, prompt, schema, *, seed, limit, temperature):
    parameters = {"seed": seed, "max_tokens": limit, "temperature": temperature, "top_p": 0.93,
                  "chat_template_kwargs": {"enable_thinking": False},
                  "response_format": {"type": "json_schema", "json_schema": {"name": "data", "schema": schema}}}
    request = Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                      data=json.dumps({**parameters, "messages": [{"role": "system", "content": system},
                                                                   {"role": "user", "content": prompt}]},
                                      ensure_ascii=False).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
    with build_opener(ProxyHandler({})).open(request, timeout=600) as response:
        result = json.load(response)
    choice = result["choices"][0]
    return {"parameters": parameters, "content": choice["message"]["content"], "model": result["model"],
            "finish_reason": choice.get("finish_reason"), "usage": result.get("usage")}


def until_valid(job, path, op, parse, attempts=4):
    for i in range(attempts):
        try:
            saved = read_json(path.with_suffix(f".attempt-{i}.json"))
        except FileNotFoundError:
            try:
                saved = op(i)
            except (URLError, OSError) as exc:
                if i == attempts - 1:
                    raise RuntimeError(f"local inference unavailable for {job}") from exc
                time.sleep(2 ** i)
                continue
            write_json(path.with_suffix(f".attempt-{i}.json"), saved)
        try:
            return parse(saved)
        except (TypeError, ValueError, KeyError):
            continue
    return None


def campaign_plan():
    batches = [{**row, "count": 100} for row in plan()]
    for j, (split, category, topic) in enumerate(EXTRA_TOPICS, 1):
        batches.append({"id": f"{split}-{category}-extra-{j}", "split": split, "topic": topic,
                        "topic_category": list(TRAIN)[category], "count": 100})
    return batches


def generate_inventory(base, batch, serial, port):
    path = base / batch["id"] / "inventory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return read_json(path)["rows"]
    size = 10
    integers = {"type": "integer", "minimum": 1, "maximum": size}
    fields = {"id": integers, "scope": {**TEXT, "maxLength": 45},
              **{key: {**TEXT, "maxLength": 15} for key in FAMILIES}}
    seed = serial * 100 + 8001
    inventory_prompt = (f"集合：{batch['split']}。类别：{batch['topic_category']}。主题：{batch['topic']}。"
                        f"拟{size}个各不相同的细分主题素材，id从1到{size}，同一主题的不同场景或事实角度。"
                        "各标签不超过12字，scope只写细分范围，不写问句，彼此有实质区别。尽量紧凑输出。")
    def parse_inventory(receipt):
        rows = json.loads(receipt["content"])["rows"]
        if (len(rows) != size or {r["id"] for r in rows} != set(range(1, size + 1))
                or any(not r[key].strip() for r in rows for key in ("scope", *FAMILIES))
                or len({normalized(r["scope"]) for r in rows}) != size):
            raise ValueError("invalid scope inventory")
        return rows

    inventory = until_valid(path, path,
                            lambda n: inference(port, PLAN_SYSTEM, inventory_prompt, table(fields, size),
                                                seed=seed+n, limit=800, temperature=.86), parse_inventory)
    write_json(path, {"batch": batch["id"], "rows": inventory or []})
    return inventory or []


def generate_unit(base, unit, batch, serial, port, inventory):
    path = base / batch["id"] / f"{unit:02}.json"
    if path.exists():
        return len(read_json(path)["rows"])
    if not inventory:
        write_json(path, {"batch": batch["id"], "rows": [], "status": "REJECTED_INVENTORY"})
        return 0
    first = (unit * UNIT_SIZE) % len(inventory)
    multi = (unit * 3) % UNIT_SIZE if unit % 4 != 3 else -1
    angle = ("普通日常互动", "条件和例外", "最近遇到的具体困惑", "从一个新手的角度",
             "同一题材的另一个用途", "表达不同意见", "对象或情境发生了变化", "用具体动作解释",
             "确认条件并采取行动", "与一个相邻概念做区别")[unit // 2]
    seed = serial * 100 + 8001
    candidates = []
    for i in range(UNIT_SIZE):
        source = inventory[(first+i) % len(inventory)]
        fields = ("q", "a", "q2", "a2") if i == multi else ("q", "a")
        schema = {"type": "object", "properties": {key: TEXT for key in fields},
                  "required": list(fields), "additionalProperties": False}
        prompt = (f"类别：{batch['topic_category']}。细分题材：{source['scope']}。写作角度：{angle}。"
                  "根据题材写一条独立的、具体自然的口语聊天，不照抄题材。问题说清实际情况；回答尽量20–65字。"
                  "题材涉及尚不知的情况时，说明缺失条件。"
                  "若是指令类，只提出有明确条件的简单请求并直接交付。"
                  "日常请求示例：问「门边那叠书太多，能帮我拿两本吗？」答「可以。我先把手上的篮子放下，再过来帮你拿。」"
                  + ("写两轮，q2要承接a并提出新疑问，a2回应新疑问。"
                     if i == multi else "只写一轮。"))
        key_path = path.with_suffix(f".row-{i:02}.json")
        if key_path.exists():
            saved = read_json(key_path)
            if saved:
                candidates.append(saved)
            continue
        def parse(receipt):
            result = json.loads(receipt["content"])
            text = [result[key].strip() for key in fields]
            messages = [{"role": ("user", "assistant")[j % 2], "content": value} for j, value in enumerate(text)]
            validate_messages(messages)
            if (normalized(text[0]) == normalized(source["scope"])
                    or len({normalized(v) for v in text}) != len(text)):
                raise ValueError("copied writing cue or repeated turn")
            return messages
        messages = until_valid(key_path, key_path, lambda n: inference(
            port, CHAT_SYSTEM, prompt, schema, seed=seed+i*10+n, limit=320 if i == multi else 180,
            temperature=.82), parse, attempts=3)
        candidate = None
        if messages:
            candidate = {"id": f"{batch['id']}-{unit:02}-{i+1:02}", "messages": messages,
                         "message_sha256": identity(messages), "topic": batch["topic"],
                         **{k: batch[k] for k in ("split", "topic_category")},
                         **{k: source[k] for k in FAMILIES},
                         "generation": {"model": MODEL, "batch": batch["id"], "unit": unit,
                                        "position": i, "evidence_scope": identity(source)}}
            candidates.append(candidate)
        write_json(key_path, candidate)
    write_json(path, {"batch": batch["id"], "rows": candidates,
                      "status": "COMPLETE", "rejected_structural": UNIT_SIZE-len(candidates)})
    return len(candidates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--port", type=int, default=18144)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit-units", type=int, default=0, help="process only this many planned units")
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 24:
        parser.error("concurrency must be 1–24")
    weights = file_info(args.weights)
    if weights["sha256"] != WEIGHTS_SHA256:
        raise ValueError("local generator weights differ")
    args.work.mkdir(parents=True, exist_ok=True)
    profile_path = args.work / "generation-profile.json"
    batches = campaign_plan()
    profile = {"model": MODEL, "weights": weights, "license": "Apache-2.0", "scope": "internal SFT experiment",
               "model_source": "https://huggingface.co/Qwen/Qwen3-4B-GGUF/tree/bc640142c66e1fdd12af0bd68f40445458f3869b",
               "backend": "llama.cpp b12810 CUDA, localhost", "batch_plan": batches,
               "historical_answers_visible": False, "external_api_calls": 0,
               "implementation": file_info(Path(__file__)), "inventory_system": PLAN_SYSTEM, "chat_system": CHAT_SYSTEM}
    units = [(i, batch) for i in range(UNITS_PER_BATCH) for batch in batches]
    if args.limit_units > 0:
        units = units[:args.limit_units]
    with (args.work / ".generation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if profile_path.exists() and read_json(profile_path) != profile:
            raise ValueError("generation profile changed in an existing campaign")
        write_json(profile_path, profile)
        start = time.monotonic()
        count = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            needed = {batch["id"] for _, batch in units}
            planned = {pool.submit(generate_inventory, args.work, batch, i, args.port): batch["id"]
                       for i, batch in enumerate(batches) if batch["id"] in needed}
            inventories = {}
            for task in as_completed(planned):
                inventories[planned[task]] = task.result()
                write_json(args.work / "inventory-progress.json", {
                    "batches": len(inventories), "planned_batches": len(planned),
                    "with_inventory": sum(bool(rows) for rows in inventories.values())})
            write_json(args.work / "inventory-progress.json",
                       {"batches": len(inventories), "with_inventory": sum(bool(rows) for rows in inventories.values())})
            future = {pool.submit(generate_unit, args.work, unit, batch, serial, args.port, inventories[batch["id"]]):
                      (batch["id"], unit)
                      for serial, (unit, batch) in enumerate(units)}
            try:
                for i, job in enumerate(as_completed(future), 1):
                    count += job.result()
                    if i % 8 == 0 or i == len(future):
                        progress = {"completed_units": i, "planned_units": len(future), "candidates": count,
                                    "elapsed_seconds_this_invocation": round(time.monotonic() - start, 2)}
                        write_json(args.work / "generation-progress.json", progress)
                        print(progress, flush=True)
            except Exception:
                for job in future:
                    job.cancel()
                raise


if __name__ == "__main__":
    main()
