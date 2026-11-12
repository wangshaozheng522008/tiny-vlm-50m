"""Interactive human review of frozen candidates, with durable per-item decisions."""
from __future__ import annotations

import argparse
import fcntl
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .checkpoint import atomic_write
from .chinese_foundation_eval import load_foundation, review_decisions
from .diagnostic_common import FOUNDATION, ROOT
from .training_manifest import file_sha256

DEFAULT_OUTPUT = ROOT / 'reports/reviews/chinese-foundation-v1.jsonl'
DOMAINS = {
    'chinese': '中文词语、成语和语法',
    'world': '中文基础世界知识',
    'commonsense': '中文日常常识',
    'math_logic': '基础数学与逻辑',
    'english_code': '简单英文与代码',
}
FINAL_ACTIONS = {'ACCEPT', 'REJECT', 'REVISE'}


@contextmanager
def _review_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('这个结果文件正被另一个审核进程使用，请退出另一个进程或选择新的 --out') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _save(path, decisions, tasks):
    # Preserve imported Raw-prompt decisions too, in frozen asset order.
    data = ''.join(json.dumps(decisions[t['id']], ensure_ascii=False) + '\n'
                   for t in tasks if t['id'] in decisions).encode('utf-8')
    atomic_write(path, lambda handle: handle.write(data))


def _summary(tasks, decisions, output):
    counts = {action: sum(decisions[t['id']]['action'] == action for t in tasks)
              for action in (*sorted(FINAL_ACTIONS), 'REVIEW_PENDING')}
    print(f"\n审核进度：{len(tasks) - counts['REVIEW_PENDING']}/{len(tasks)}；"
          f"通过 {counts['ACCEPT']}，拒绝 {counts['REJECT']}，需修改 {counts['REVISE']}，"
          f"待审 {counts['REVIEW_PENDING']}。")
    if output.exists():
        print(f'审核结果：{output}')
        print(f'评测时添加：--reviews {output}')


def review_candidates(out=DEFAULT_OUTPUT, *, reviewer=None, directory=FOUNDATION, read=None):
    """Only an explicit human action may finalize an item; empty input never accepts."""
    read = input if read is None else read
    directory = Path(directory).resolve()
    output = Path(out).expanduser()
    if output.is_symlink():
        raise ValueError('审核结果不能是符号链接，请使用独立的 .jsonl 文件')
    output = output.resolve()
    if (output.is_relative_to(directory) or output.is_relative_to(ROOT / 'data')
            or output.suffix != '.jsonl'):
        raise ValueError('请将审核结果保存到冻结数据目录之外的 .jsonl 文件，例如 reports/reviews/')
    candidates, prompts = load_foundation(directory)
    candidate_hash = file_sha256(directory / 'candidates.jsonl')
    with _review_lock(output):
        existed = output.exists()
        decisions = review_decisions(output if existed else None, candidates, prompts, directory)
        added = False
        for task in candidates:
            if task['id'] not in decisions:
                decisions[task['id']] = {
                    'id': task['id'], 'question_version': task['question_version'],
                    'asset_sha256': candidate_hash, 'action': 'REVIEW_PENDING',
                    'reviewer': '', 'reason': '', 'human_review': False,
                }
                added = True

        print('中文基础候选题人工审核（只审核120道候选题，不自动审核30条Raw前缀）')
        print('请检查题意、参考答案是否唯一且正确，以及候选答案的长度/格式偏差。')
        print('1 + 回车：通过；2 + 回车：拒绝；s：稍后审核；q：保存退出。每题决定立即保存。')
        _summary(candidates, decisions, output)
        pending = [task for task in candidates if decisions[task['id']]['action'] not in FINAL_ACTIONS]
        if not pending:
            print('所有候选题均已审核，无需重复审核。')
            return 0
        try:
            if reviewer is not None:
                reviewer = reviewer.strip()
                if not reviewer:
                    raise ValueError('--reviewer 不能为空')
            while not reviewer:
                reviewer = read('请输入审核人姓名或标识：').strip()
                if not reviewer:
                    print('审核人不能为空。')
            if not existed or added:
                _save(output, decisions, candidates + prompts)
            for number, task in enumerate(candidates, start=1):
                if decisions[task['id']]['action'] in FINAL_ACTIONS:
                    continue
                print(f"\n{'─' * 48}\n[{number}/{len(candidates)}] {task['id']} · "
                      f"{DOMAINS.get(task['category'], task['category'])} · 版本{task['question_version']}")
                print(f"题目：{task['prompt']}")
                for key, answer in task['candidates'].items():
                    print(f'  {key}. {answer}')
                reference = task['reference']
                print(f"参考答案：{reference}. {task['candidates'][reference]}")
                print(f"来源：{task['source']}")
                while True:
                    choice = read('你的审核 [1通过 / 2拒绝 / s稍后 / q退出]：').strip().lower()
                    if choice in ('q', 'quit', '退出'):
                        _summary(candidates, decisions, output)
                        return 0
                    if choice in ('s', 'skip', '稍后', '跳过'):
                        break
                    if choice in ('1', 'a', 'accept', '通过'):
                        action, reason = 'ACCEPT', '人工逐题审查：选择通过。'
                    elif choice in ('2', 'r', 'reject', '拒绝'):
                        reason = read('拒绝原因（回车记录“未填写具体原因”）：').strip()
                        action = 'REJECT'
                        reason = reason or '人工逐题审查：拒绝，未填写具体原因。'
                    else:
                        print('请输入1、2、s或q；直接回车不会通过题目。')
                        continue
                    # Catch changes made by an external editor before committing a decision.
                    if file_sha256(directory / 'candidates.jsonl') != candidate_hash:
                        raise ValueError('题库在审核期间发生变化，停止保存；请恢复冻结题库后继续')
                    updated = {**decisions, task['id']: {
                        **decisions[task['id']], 'action': action,
                        'reviewer': reviewer, 'reason': reason, 'human_review': True,
                        'reviewed_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    }}
                    _save(output, updated, candidates + prompts)
                    decisions = updated
                    print(f"已保存：{task['id']} {'通过' if action == 'ACCEPT' else '拒绝'}")
                    break
        except (KeyboardInterrupt, EOFError):
            if output.exists():
                decisions.update(review_decisions(output, candidates, prompts, directory))
            print('\n已退出；已保存的决定保留，未保存的题目下次继续审核。')
        _summary(candidates, decisions, output)
        return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='逐题显示120道中文候选题，人工选择通过/拒绝并立即保存，可断点继续')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT, help='结果JSONL；已有合法结果自动继续')
    parser.add_argument('--reviewer', help='审核人姓名/标识；省略则启动后询问')
    args = parser.parse_args(argv)
    try:
        return review_candidates(args.out, reviewer=args.reviewer)
    except (OSError, ValueError) as exc:
        print(f'审核失败：{exc}')
        return 1
