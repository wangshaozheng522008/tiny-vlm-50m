#!/usr/bin/env python3
"""Generate frozen-plan batches with an authorized Chat Completions endpoint."""
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
from threading import Event
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from minillm.assessment_io import read_json
from minillm.corpus_utils import file_info, identity, verify_file
from minillm.training_manifest import write_json
from minillm.v5_sft_data import CONFIG, ROOT, TRAIN, freeze, validate_messages, write_rows

# Each entry identifies one batch and its split's subject scope.
TOPICS = {
    "train": (
        "晨起收拾|午饭安排|菜市场买菜|厨房清洁|剩菜储存|衣物整理|晾衣天气|寄件取件|等公交|站内换乘|骑车通勤|下班约饭|周末散步|公园约见|图书馆借书|同事协作|家庭分工|访客准备|邻里寒暄|朋友改约|出门忘带物品|网购退货|浇花安排|节日问候|临时停水",
        "河流汇合|海岸潮汐|湖泊湿地|盆地和丘陵|森林季节|草原生态|常见鸟类习性|城市昆虫|淡水鱼类|家畜特征|种子传播|植物萌发|根茎叶|花粉传播|月相观察|皮影与剪纸|古代邮驿|乡土建筑|机械传动|光线反射|声音传播",
        "低温保鲜|水汽凝结|摩擦力|声音隔离|灯光与影子|包装密封|防晒遮阳|保温隔热|走路与体感|清洗与污渍|纸张吸水|材料软硬|分类整理|天气预报表述",
        "按给定顺序排列|改单句语气|提取指定地点|用一句话转述|把叙述改为问句|拟写简短便条|选择符合指定条件的物品",
        "未知私人日程|未知室内物品位置|信息不足的代词指代",
    ),
    "val": ("理发预约|接待外地来客|雨天晾鞋", "候鸟迁徙|盐碱地植被|古代计时",
            "玻璃反光|排队叫号", "按字数改写招呼", "缺少地点的营业询问"),
    "test": ("邻里借用物品|餐馆落座|运动场地预约", "潮间带生物|高山冻土|传统织造",
             "音量与距离|橡胶有弹性", "归纳两项具体安排", "缺少型号的电器耗电询问"),
}
FAMILY_PROMPT = """为中文日常问答建立主题清单。按给定主题，列出30条不同且可独立核查的事实、
问题及对话族，支持后续生成100条问答。输出JSON对象，字段rows为列表，每条仅包含
fact_family、question_family、dialogue_family、scope四个非空字符串。scope描述本条覆盖的
细分事实或情境，避免同一事实的复述、时效性事实和用人名替换凑数。不写问答、评测题或答案。"""
PROMPT = """为约8700万参数的模型生成自然简体中文问答。仅使用提供的细分事实与问题族，
然后输出一个 JSON 对象，唯一字段 rows 为100条样本。每条包含 messages、fact_family、question_family、
dialogue_family。messages 为 user、assistant 交替的对象数组，消息字段 role 和 content。约10条为
2–3轮，需要使用已给出的上下文。回答以20–80字、1–3个短句为主，简单事实允许较短。问题用真实日常说法，
同一细分范围内使用不同的具体事实或情境；不要做替换名词的模板变体。优先稳定容易核查的事实。信息不足时说明缺少什么。
避免代码、复杂计算、长推理、Markdown、连续复读、固定开场白以及考试题改写。JSON之外无内容。"""


def plan():
    return [{"id": f"{split}-{category_index}-{i:02}", "split": split, "topic_category": category,
             "topic": topic, "count": 100, "prompt": PROMPT}
            for split, topics in TOPICS.items()
            for category_index, (category, group) in enumerate(zip(TRAIN, topics))
            for i, topic in enumerate(group.split("|"), 1)]


class RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def generate(work, split, phase, authorization_path, concurrency, attempts):
    auth = read_json(authorization_path)
    required = ("model", "provider", "source", "evidence", "scope", "currency",
                "input_per_million", "output_per_million")
    if (any(key not in auth or not auth[key] for key in required if not key.endswith("_million"))
            or any(key not in auth for key in required)
            or auth.keys() - {*required, "actual_models"}
            or auth["scope"] != "training_allowed"):
        raise ValueError("an explicit provider/model training-use authorization receipt is required")
    models = auth.get("actual_models", [auth["model"]])
    if not isinstance(models, list) or not models or any(not isinstance(model, str) or not model for model in models):
        raise ValueError("actual_models must list the covered provider model identifiers")
    for key in ("input_per_million", "output_per_million"):
        if type(auth[key]) not in (int, float) or not 0 <= auth[key] < float("inf"):
            raise ValueError("invalid token pricing")
    api_key = os.environ.get("V5_SFT_API_KEY")
    endpoint = os.environ.get("V5_SFT_API_URL", "")
    parsed = urlsplit(endpoint)
    if not api_key or parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query:
        raise ValueError("set V5_SFT_API_KEY and a credential-free HTTPS V5_SFT_API_URL")
    if not 1 <= concurrency <= 16 or not 1 <= attempts <= 10:
        raise ValueError("concurrency must be 1–16 and retry attempts 1–10")
    params = {"model": auth["model"], "temperature": 0.8, "max_tokens": 28000,
              "response_format": {"type": "json_object"}}
    binding = {"authorization": auth, "endpoint": endpoint, "request": params,
               "family_prompt": FAMILY_PROMPT, "generator": file_info(Path(__file__)), "plan": identity(plan())}
    work.mkdir(parents=True, exist_ok=True)
    with (work / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = work / "generation-profile.json"
        if receipt.exists():
            if read_json(receipt) != binding:
                raise ValueError("batch workspace is bound to another model, plan or generator")
        else:
            write_json(receipt, binding)

        fatal_error = Event()
        directory = work / phase
        directory.mkdir(exist_ok=True)

        def batch(spec):
            if fatal_error.is_set():
                return
            target = directory / f"{spec['id']}.json"
            if target.exists():
                saved = read_json(target)
                if saved["batch"] != spec or saved["profile"] != identity(binding):
                    raise ValueError(f"{target}: batch identity changed")
                return
            query = f"类别：{spec['topic_category']}。独立主题范围：{spec['topic']}。批次：{spec['id']}。"
            if phase == "candidates":
                family = read_json(work / "families" / target.name)
                if family["profile"] != identity(binding) or family["batch"] != spec:
                    raise ValueError("family plan does not belong to the candidate batch")
                allowed = {tuple(row[key] for key in ("fact_family", "question_family", "dialogue_family"))
                           for row in family["rows"]}
                query += "\n细分清单：" + json.dumps(family["rows"], ensure_ascii=False)
            system = FAMILY_PROMPT if phase == "families" else spec["prompt"]
            body = json.dumps({**params, "messages": [{"role": "system", "content": system},
                                                      {"role": "user", "content": query}]}).encode()
            request = Request(endpoint, data=body, method="POST",
                              headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
            for attempt in range(attempts):
                usage = None
                error = None
                returned_model = None
                try:
                    with build_opener(RejectRedirects()).open(request, timeout=600) as response:
                        payload = json.load(response)
                    usage = payload.get("usage")
                    returned_model = payload["model"]
                    if returned_model not in models:
                        raise ValueError("returned model is outside the recorded authorization")
                    rows = json.loads(payload["choices"][0]["message"]["content"])["rows"]
                    count = 30 if phase == "families" else spec["count"]
                    if not isinstance(rows, list) or len(rows) != count:
                        raise ValueError("batch size differs")
                    accepted = []
                    for row in rows:
                        try:
                            family = {k: row[k] for k in ("fact_family", "question_family", "dialogue_family")}
                            if any(not isinstance(v, str) or not v.strip() for v in family.values()):
                                continue
                            if phase == "families":
                                scope = row["scope"]
                                if isinstance(scope, str) and scope.strip():
                                    accepted.append({**family, "scope": scope})
                            elif tuple(family.values()) in allowed:
                                validate_messages(row["messages"])
                                accepted.append({"messages": row["messages"], **family})
                        except (ValueError, KeyError, TypeError):
                            continue
                    if not accepted:
                        raise ValueError("batch has zero structurally valid candidates")
                    if phase == "families" and len({row["fact_family"] for row in accepted}) != count:
                        raise ValueError("family plan must contain 30 distinct factual or dialogue scopes")
                except HTTPError as exc:
                    error = f"HTTP {exc.code}"
                    fatal = exc.code < 500 and exc.code not in (408, 429)
                    exc.close()
                except (URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError) as exc:
                    error, fatal = type(exc).__name__, False
                with (directory / f"{spec['id']}.attempts.jsonl").open("a", encoding="utf-8") as audit:
                    audit.write(json.dumps({"attempt": attempt + 1, "usage": usage, "model": returned_model,
                                            "error": error}) + "\n")
                    audit.flush()
                    os.fsync(audit.fileno())
                if error is None:
                    write_json(target, {"batch": spec, "profile": identity(binding), "model": returned_model,
                                        "api_response_id": payload.get("id"), "rows": accepted, "usage": usage})
                    return
                if fatal:
                    fatal_error.set()
                    break
                if attempt + 1 < attempts:
                    time.sleep(min(60, 2 ** attempt))
            raise RuntimeError(f"{spec['id']}: generation failed ({error})")

        batches = [spec for spec in plan() if spec["split"] == split]
        if phase == "candidates" and any(not (work / "families" / f"{spec['id']}.json").exists()
                                         for spec in batches):
            raise ValueError("complete the split's family inventory before generating candidates")
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            list(executor.map(batch, batches))


def collect(work, output):
    from minillm.assessment_io import records, report_directory
    with (work / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        binding = read_json(work / "generation-profile.json")
        if binding["plan"] != identity(plan()):
            raise ValueError("changed batch plan")
        rows, batches, attempts = [], {}, []
        for spec in plan():
            target = work / "candidates" / f"{spec['id']}.json"
            saved = read_json(target)
            if saved["batch"] != spec or saved["profile"] != identity(binding):
                raise ValueError(f"{target}: incompatible batch receipt")
            phases = {}
            for phase in ("families", "candidates"):
                log = work / phase / f"{spec['id']}.attempts.jsonl"
                attempts.extend({"batch": spec["id"], "phase": phase, **attempt} for attempt in records(log))
                phases[phase] = {"file": file_info(work / phase / target.name), "attempt_log": file_info(log)}
            batches[spec["id"]] = {"phases": phases,
                                   "model": saved["model"], "api_response_id": saved["api_response_id"]}
            for i, row in enumerate(saved["rows"], 1):
                key = f"{spec['id']}-{i:03}"
                rows.append({**row, "id": key, "message_sha256": identity(row["messages"]), "split": spec["split"],
                             "topic_category": spec["topic_category"], "topic": spec["topic"],
                             "generation": {"batch": spec["id"], "model": saved["model"], "profile": identity(binding)}})
    if not 8000 <= len(rows) <= 10000:
        raise ValueError(f"only {len(rows)} usable candidates after structural checks")
    known = [r["usage"] for r in attempts if isinstance(r.get("usage"), dict)
             and all(type(r["usage"].get(k)) is int and r["usage"][k] >= 0 for k in ("prompt_tokens", "completion_tokens"))]
    inputs = sum(r["prompt_tokens"] for r in known)
    outputs = sum(r["completion_tokens"] for r in known)
    auth = binding["authorization"]
    with report_directory(output) as directory:
        write_rows(directory / "candidates.jsonl", rows)
        write_rows(directory / "attempts.jsonl", attempts)
        write_json(directory / "generation.json", {"profile": binding, "plan": plan(), "batches": batches,
                   "candidates": file_info(directory / "candidates.jsonl"),
                   "calls_without_usage": len(attempts) - len(known), "candidate_count": len(rows),
                   "input_tokens": inputs, "output_tokens": outputs,
                   "estimated_reported_usage_cost": (inputs * auth["input_per_million"]
                                                     + outputs * auth["output_per_million"]) / 1e6})


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT / "artifacts/v5-language-sft-5k-v1/batches")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan").add_argument("output", type=Path)
    g = sub.add_parser("generate")
    g.add_argument("--split", choices=tuple(TOPICS), required=True)
    g.add_argument("--phase", choices=("families", "candidates"), required=True)
    g.add_argument("--authorization", type=Path, required=True)
    g.add_argument("--concurrency", type=int, default=2)
    g.add_argument("--attempts", type=int, default=4)
    sub.add_parser("collect").add_argument("output", type=Path)
    f = sub.add_parser("freeze")
    f.add_argument("candidate_directory", type=Path)
    f.add_argument("reviews", type=Path)
    f.add_argument("output", type=Path)
    f.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "plan":
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(plan(), handle, ensure_ascii=False, indent=2)
    elif args.command == "generate":
        generate(args.work, args.split, args.phase, args.authorization, args.concurrency, args.attempts)
    elif args.command == "collect":
        collect(args.work, args.output)
    else:
        from minillm.v5_sft_experiment import base_assets
        settings = read_json(CONFIG)
        tokenizer, _, _ = base_assets(settings)
        source = args.candidate_directory / "candidates.jsonl"
        generation = read_json(args.candidate_directory / "generation.json")
        verify_file(source, generation["candidates"])
        audit = {"generation": generation}
        with args.audit.open("x", encoding="utf-8") as handle:
            try:
                freeze(source, args.reviews, args.output, tokenizer, settings["max_seq_len"], inspected=audit)
            finally:
                json.dump(audit, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
