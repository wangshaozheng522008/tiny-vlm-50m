"""Interactive choices must persist exactly and never masquerade as bulk acceptance."""
import fcntl
import json
import subprocess
import sys

import pytest

import minillm.foundation_review as review
from minillm.chinese_foundation_eval import load_foundation, review_decisions
from minillm.diagnostic_common import FOUNDATION, ROOT
from minillm.training_manifest import file_sha256


def answers(*values):
    stream = iter(values)
    def read(prompt):
        try:
            value = next(stream)
        except StopIteration:
            raise EOFError from None
        if isinstance(value, BaseException):
            raise value
        return value
    return read


def decisions(path):
    candidates, prompts = load_foundation()
    return review_decisions(path, candidates, prompts)


def test_choices_display_and_save_evaluator_ready_results(tmp_path, capsys):
    output = tmp_path / 'review.jsonl'
    assert review.review_candidates(output, reviewer='TEST-SIMULATED-ONLY',
                                    read=answers('1', '2', '题意存在歧义', 's', 'q')) == 0
    saved = decisions(output)
    assert len(saved) == 120
    assert saved['zhf-001']['action'] == 'ACCEPT'
    assert saved['zhf-002']['action'] == 'REJECT'
    assert saved['zhf-002']['reason'] == '题意存在歧义'
    assert saved['zhf-001']['human_review'] is True
    assert saved['zhf-001']['reviewed_at'].endswith('+00:00')
    assert all(saved[f'zhf-{i:03}']['action'] == 'REVIEW_PENDING' for i in range(3, 121))
    assert all(not saved[f'zhf-{i:03}']['human_review'] for i in range(3, 121))
    text = capsys.readouterr().out
    assert '[1/120] zhf-001' in text and 'A. 昏暗' in text and '参考答案：A. 昏暗' in text
    assert '通过 1，拒绝 1' in text


def test_resume_preserves_completed_items_and_reviewer(tmp_path, capsys):
    output = tmp_path / 'resume.jsonl'
    review.review_candidates(output, reviewer='TEST-FIRST', read=answers('1', 'q'))
    original = decisions(output)['zhf-001']
    capsys.readouterr()
    review.review_candidates(output, reviewer='TEST-SECOND', read=answers('2', '', 'q'))
    saved = decisions(output)
    assert saved['zhf-001'] == original
    assert saved['zhf-002']['reviewer'] == 'TEST-SECOND'
    assert saved['zhf-002']['action'] == 'REJECT'
    assert '未填写具体原因' in saved['zhf-002']['reason']
    assert '[1/120] zhf-001' not in capsys.readouterr().out


@pytest.mark.parametrize('interruption', [EOFError(), KeyboardInterrupt()])
def test_interrupt_preserves_last_decision_and_no_partial_rejection(tmp_path, interruption):
    output = tmp_path / 'interrupted.jsonl'
    review.review_candidates(output, reviewer='TEST-ONLY', read=answers('1', '2', interruption))
    saved = decisions(output)
    assert saved['zhf-001']['action'] == 'ACCEPT'
    assert saved['zhf-002']['action'] == 'REVIEW_PENDING'


def test_empty_invalid_and_skip_are_not_acceptance(tmp_path):
    output = tmp_path / 'pending.jsonl'
    review.review_candidates(output, read=answers('', 'TEST-ONLY', '', '全部通过', 's', 'q'))
    assert all(r['action'] == 'REVIEW_PENDING' and not r['human_review'] for r in decisions(output).values())


def test_only_explicit_choices_can_complete_all_120(tmp_path, capsys):
    output = tmp_path / 'all-test-only.jsonl'
    review.review_candidates(output, reviewer='TEST-SIMULATED-NOT-HUMAN', read=answers(*(['1'] * 120)))
    assert sum(r['action'] == 'ACCEPT' for r in decisions(output).values()) == 120
    previous = output.read_bytes()
    review.review_candidates(output, read=lambda _: pytest.fail('completed review must not request input'))
    assert output.read_bytes() == previous
    assert '所有候选题均已审核' in capsys.readouterr().out


def test_frozen_assets_cannot_be_used_as_output():
    output = FOUNDATION / 'review-template.jsonl'
    before = file_sha256(output)
    with pytest.raises(ValueError, match='冻结数据目录之外'):
        review.review_candidates(output, reviewer='TEST-ONLY')
    assert file_sha256(output) == before


@pytest.mark.parametrize('corruption', ['invalid-json', 'wrong-hash'])
def test_invalid_existing_review_is_never_overwritten(tmp_path, corruption):
    output = tmp_path / 'bad.jsonl'
    if corruption == 'invalid-json':
        output.write_text('not JSON\n')
    else:
        row = json.loads((FOUNDATION / 'review-template.jsonl').read_text().splitlines()[0])
        row['asset_sha256'] = '0' * 64
        output.write_text(json.dumps(row))
    previous = output.read_bytes()
    with pytest.raises(ValueError):
        review.review_candidates(output, reviewer='TEST-ONLY', read=answers('1'))
    assert output.read_bytes() == previous


def test_concurrent_session_is_rejected(tmp_path):
    output = tmp_path / 'locked.jsonl'
    with output.with_suffix('.jsonl.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match='另一个审核进程'):
            review.review_candidates(output, reviewer='TEST-ONLY', read=answers('1'))
    assert not output.exists()


def test_failed_atomic_replace_retains_previous_review(tmp_path, monkeypatch):
    import minillm.checkpoint as checkpoint
    output = tmp_path / 'disk-failure.jsonl'
    review.review_candidates(output, reviewer='TEST-ONLY', read=answers('1', 'q'))
    previous = output.read_bytes()
    def fail(*args):
        raise OSError('simulated disk full')
    monkeypatch.setattr(checkpoint.os, 'replace', fail)
    with pytest.raises(OSError, match='disk full'):
        review.review_candidates(output, reviewer='TEST-ONLY', read=answers('2', '模拟拒绝理由'))
    assert output.read_bytes() == previous
    assert decisions(output)['zhf-002']['action'] == 'REVIEW_PENDING'
    assert not list(tmp_path.glob('.*.tmp'))


def test_imported_raw_review_is_preserved(tmp_path):
    output = tmp_path / 'template-copy.jsonl'
    output.write_bytes((FOUNDATION / 'review-template.jsonl').read_bytes())
    review.review_candidates(output, reviewer='TEST-ONLY', read=answers('1', 'q'))
    saved = decisions(output)
    assert len(saved) == 150 and saved['zhraw-001']['action'] == 'REVIEW_PENDING'


def test_cli_smoke_and_frozen_asset_integrity(tmp_path):
    before = {p.name: file_sha256(p) for p in FOUNDATION.iterdir() if p.is_file()}
    output = tmp_path / 'cli-test-only.jsonl'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/review_chinese_foundation.py'),
                             '--reviewer', 'TEST-SIMULATED-NOT-HUMAN', '--out', str(output)],
                            input='1\nq\n', text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '已保存：zhf-001 通过' in result.stdout
    assert decisions(output)['zhf-001']['action'] == 'ACCEPT'
    assert before == {p.name: file_sha256(p) for p in FOUNDATION.iterdir() if p.is_file()}
