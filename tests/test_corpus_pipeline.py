"""Synthetic-only corpus engineering, integrity, failure and reproducibility tests."""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from minillm.corpus import (Pipeline, guard_tokenizer_training, inventory, make_content,
                           read_records, tokenize_split)
from minillm.corpus_audit import audit_dataset
from minillm.corpus_quality import (DEFAULT_QUALITY, classify, content_hash, normalise,
                                    quality_issue, shingles, similar, simhash)
from minillm.corpus_utils import CorpusError, file_info, identity
from minillm.data import PackedPretrainDataset, encode_document_tokens, stream_corpus_records, stream_documents

CHARS = '人工智能模型语言训练数据数学推理科学技术方法世界生活学习探索结果网络算法计算程序函数优化研究分析测试中文英文知识问题答案信息系统任务目标发展历史未来文化'


def synthetic(count, seed=5, length=160):
    rng = random.Random(seed)
    for i in range(count):
        yield {"text": ''.join(rng.choices(CHARS, k=length)) + f'。编号{i}。', "source": "synthetic", "license": "synthetic"}


def jsonl(path, records):
    with path.open('w') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    return path


@pytest.fixture
def corpus_inputs(tmp_path):
    return [jsonl(tmp_path / 'a.jsonl', synthetic(180)), jsonl(tmp_path / 'b.jsonl', synthetic(120, seed=6))]


def content(**kwargs):
    return make_content({"quality": {"min_chars": 20, "max_chars": 256}, **kwargs})


def pipeline(tmp_path, paths, name='dataset', config=None, runtime=None, log=None):
    return Pipeline(tmp_path / name, config or content(), inventory([str(p) for p in paths]), runtime=runtime,
                    log=log or (lambda *_: None))


def test_streaming_jsonl_txt_and_invalid_records(tmp_path):
    p = tmp_path / 'records.jsonl'
    p.write_bytes(b'{"text":"hello"}\ninvalid\n{"x":1}\n' + b'x' * 4096 + b'\n{"text":"tail"}\n')
    records = list(stream_corpus_records(p, max_record_bytes=1024))
    assert [r.get('text') for r in records] == ['hello', None, None, None, 'tail']
    assert records[3]['_issue'] == 'record_too_large'
    t = tmp_path / 'records.txt'
    t.write_text('first paragraph\nsecond line\n\n' + 'x' * 4096 + '\n\ntail paragraph')
    assert list(stream_documents([t])) == ['first paragraph\nsecond line\n', 'x' * 4096 + '\n', 'tail paragraph']
    bounded = list(stream_corpus_records(t, max_record_bytes=1024))
    assert [r.get('text') for r in bounded] == ['first paragraph\nsecond line\n', None, 'tail paragraph']


@pytest.mark.parametrize('text,expected', [
    ('人工智能可以帮助我们理解语言与数学。', 'zh'),
    ('This article explains language models and their evaluation.', 'en'),
    ('def add(a, b):\n    return a + b\n', 'code'),
    ('这是一段混合文本 with a sufficiently long English explanation of language models.', 'mixed'),
    ('123 + 456 = 579，678 / 2 = 339，数学公式。', 'zh'), ('123 + 456 = 579; 678 / 2 = 339', 'unknown'),
])
def test_explainable_categories(text, expected):
    assert classify(text) == expected


def test_normalization_quality_and_code_preservation():
    text = '<html><nav>导航链接</nav><p> 人工智能\u3000研究  数据 </p><script>ad()</script></html>'
    cleaned = normalise(text, 'zh')
    assert '导航' not in cleaned and 'ad()' not in cleaned and '<' not in cleaned
    assert content_hash('人工智能  研究\n数据', 'zh') == content_hash('人工智能 研究 数据', 'zh')
    code = 'def add(a, b):\n    return a + b\n'
    assert '    return' in normalise(code, 'code')
    assert quality_issue(code, 'code', DEFAULT_QUALITY) is None
    assert quality_issue('123 + 456 = 579; 678 / 2 = 339', 'unknown', DEFAULT_QUALITY) is None
    assert quality_issue('这是一段包含异常字符的测试文本。' * 8 + '\ufffd', 'zh', DEFAULT_QUALITY) == 'replacement_character'
    assert quality_issue(('模型需要数据和计算资源才能完成语言训练任务。\n') * 8, 'zh', DEFAULT_QUALITY) == 'repeated_sentence_or_paragraph'
    assert quality_issue('!!@@$$%^&*' * 20, 'unknown', DEFAULT_QUALITY) == 'low_information'
    assert quality_issue('加微信联系购买电话：123456789。' * 10, 'zh', DEFAULT_QUALITY) == 'advertising_contacts'


def test_full_document_near_duplicate_comparisons():
    body = next(synthetic(1, length=1200))['text']
    variant = body[:620] + '科' + body[621:]
    a, b = shingles('原标题\n' + body, 'zh'), shingles('新标题\n' + variant, 'zh')
    assert (simhash(a) ^ simhash(b)).bit_count() <= 4
    assert similar(a, b, .85)
    prefix = body[:256]
    different = prefix + next(synthetic(1, seed=901, length=1200))['text']
    assert not similar(shingles(body, 'zh'), shingles(different, 'zh'), .85)


def test_dedup_cross_source_whitespace_near_and_prefix(tmp_path):
    body = next(synthetic(1, length=1200))['text']
    records = [
        {'text': '标题一\n' + body, 'source': 'original'},
        {'text': '标题一  ' + body, 'source': 'repost'},
        {'text': '标题二\n' + body[:620] + '科' + body[621:], 'source': 'edited'},
        {'text': body[:256] + next(synthetic(1, seed=100, length=1200))['text'], 'source': 'different'},
    ]
    p = jsonl(tmp_path / 'dedup.jsonl', records)
    cfg = make_content({'quality': {'min_chars': 20, 'max_chars': 2000}})
    run = pipeline(tmp_path, [p], config=cfg)
    run.split()
    stats = json.loads((run.out / '.pipeline/deduplicate.stage.json').read_text())['stats']
    assert stats['duplicates_exact'] == 1
    assert stats['duplicates_near'] == 1
    kept = [r for n in ('train','val','test') for r in read_records(run.out / f'{n}.text.jsonl')]
    assert len(kept) == 2  # same prefix + different body survives


def test_reproducible_bins_input_order_and_runtime(tmp_path, corpus_inputs, tiny_tokenizer):
    tok = tmp_path / 'tok.json'; tiny_tokenizer.save(tok)
    first = pipeline(tmp_path, corpus_inputs, 'first')
    second = pipeline(tmp_path, list(reversed(corpus_inputs)), 'second', runtime={'sqlite_cache_mib': 8, 'buffer_bytes': 38, 'commit_every': 13})
    a, b = first.tokenize(tok), second.tokenize(tok)
    assert a['content_id'] == b['content_id']
    assert a['files'] == b['files']
    assert a['splits'] == b['splits']
    assert a['deduplication'] == b['deduplication']
    for split in ('train','val','test'):
        assert (first.out / f'{split}.bin').read_bytes() == (second.out / f'{split}.bin').read_bytes()
    audit = audit_dataset(first.out, retokenize=True, verify_inputs=True)
    assert audit['ok'], audit['errors']
    assert audit['exact_overlap'] == {'train/val':0, 'train/test':0, 'val/test':0}
    assert audit['known_near_duplicate_cross_split_groups'] == 0
    assert len(PackedPretrainDataset(first.out / 'train.bin', 16)) > 0


def test_parent_fragments_and_explicit_families_stay_together(tmp_path, tiny_tokenizer):
    records = list(synthetic(30, length=800))
    for record in records[:8]: record['parent_id'] = 'same-original-book'
    p = jsonl(tmp_path / 'long.jsonl', records)
    run = pipeline(tmp_path, [p], config=content(quality={'min_chars':20,'max_chars':128}))
    tok=tmp_path/'tok.json'; tiny_tokenizer.save(tok)
    run.tokenize(tok)
    parents, groups = {}, {}
    for split in ('train','val','test'):
        for row in read_records(run.out/f'{split}.text.jsonl'):
            assert len(row['text']) <= 128
            parents.setdefault(row['parent_id'], set()).add(split)
            groups.setdefault(row['group_id'], set()).add(split)
    assert all(len(s) == 1 for s in (*parents.values(), *groups.values()))
    assert audit_dataset(run.out)['ok']


def test_estimated_token_mix_exhaustion_no_replacement(tmp_path, tiny_tokenizer):
    a=jsonl(tmp_path/'short.jsonl', synthetic(80, length=80))
    b=jsonl(tmp_path/'long.jsonl', synthetic(5, seed=7, length=800))
    sources=[{'name':'short','paths':[str(a)]},{'name':'long','paths':[str(b)]}]
    cfg=content(mixing={'by':'source','weights':{'short':.5,'long':.5},'token_budget':10000})
    run=Pipeline(tmp_path/'mixed',cfg,inventory([],sources),log=lambda *_:None)
    tok=tmp_path/'tok.json';tiny_tokenizer.save(tok)
    manifest=run.tokenize(tok)
    selected=manifest['mixing']['selection']
    assert selected['long']['exhausted'] and selected['long']['estimated_shortfall']>0
    assert selected['long']['selected_documents'] == 20  # 5 long parents split into bounded fragments
    assert selected['long']['selected_estimated_tokens']<=5000
    assert sum(v['fraction'] for v in manifest['mixing']['actual_token_fractions']['by_source'].values()) == pytest.approx(1)
    assert audit_dataset(run.out)['remaining_exact_duplicates'] == 0


def test_token_stream_generator_range_and_atomic_failure(tmp_path, tiny_tokenizer):
    path=tmp_path/'tokens.bin'
    records=list(synthetic(3))
    stats=tokenize_split(iter(records),tiny_tokenizer,path,buffer_bytes=10)
    expected=b''.join(a.tobytes() for a in encode_document_tokens((r['text'] for r in records),tiny_tokenizer))
    assert path.read_bytes()==expected and stats['tokens']*2==len(expected)
    before=path.read_bytes()
    def broken():
        yield records[0]
        raise OSError('disk failure simulation')
    with pytest.raises(OSError,match='disk failure'):
        tokenize_split(broken(),tiny_tokenizer,path,buffer_bytes=10)
    assert path.read_bytes()==before and not list(tmp_path.glob('.*.tmp'))
    with patch.object(tiny_tokenizer,'encode_document',return_value=[65536]):
        with pytest.raises(ValueError,match='range'):
            tokenize_split(iter(records),tiny_tokenizer,path)
    assert path.read_bytes()==before


def test_train_only_tokenizer_and_immutable_publication(tmp_path, corpus_inputs):
    run=pipeline(tmp_path,corpus_inputs)
    run.split()
    with pytest.raises(CorpusError,match='held-out'):
        guard_tokenizer_training([run.out/'val.text.jsonl'],[])
    with patch('minillm.corpus.MiniLLMTokenizer.train', wraps=__import__('minillm.tokenizer',fromlist=['MiniLLMTokenizer']).MiniLLMTokenizer.train) as train:
        manifest=run.tokenize(vocab_size=512,max_tokenizer_chars=20000)
    assert train.call_count==1
    assert manifest['tokenizer']['mode']=='trained_on_train_only'
    assert manifest['tokenizer']['characters']<=20000
    before={p.name:p.read_bytes() for p in run.out.iterdir() if p.is_file()}
    run.tokenize()
    assert before=={p.name:p.read_bytes() for p in run.out.iterdir() if p.is_file()}
    with pytest.raises(CorpusError,match='changed'):
        pipeline(tmp_path,corpus_inputs,config=content(seed=43))
    assert before=={p.name:p.read_bytes() for p in run.out.iterdir() if p.is_file()}


def test_failed_stage_verified_resume_and_configuration_rejection(tmp_path,corpus_inputs,tiny_tokenizer):
    messages=[];run=pipeline(tmp_path,corpus_inputs,log=messages.append)
    tok=tmp_path/'tok.json';tiny_tokenizer.save(tok)
    original=tokenize_split
    def fail_val(records, tokenizer, path, **kwargs):
        if path.name=='val.bin': raise OSError('injected tokenization failure')
        return original(records,tokenizer,path,**kwargs)
    with patch('minillm.corpus.tokenize_split',side_effect=fail_val),pytest.raises(CorpusError,match='tokenize-val FAILED'):
        run.tokenize(tok)
    assert not (run.out/'dataset-manifest.json').exists()
    assert (run.out/'.pipeline/tokenize-train.stage.json').exists()
    with patch('minillm.corpus.normalize_sources',side_effect=AssertionError('must reuse normalize')):
        run.tokenize(tok)
    assert any('[reuse] normalize' in m for m in messages)
    assert audit_dataset(run.out)['ok']
    with pytest.raises(CorpusError,match='changed'):
        pipeline(tmp_path,corpus_inputs,config=content(deduplication={'threshold':.9}))


def test_audit_detects_corrupt_output_and_tokenizer(tmp_path,corpus_inputs,tiny_tokenizer):
    run=pipeline(tmp_path,corpus_inputs);tok=tmp_path/'tok.json';tiny_tokenizer.save(tok)
    manifest=run.tokenize(tok)
    path=run.out/'train.bin';raw=path.read_bytes();path.write_bytes(raw[:-1])
    report=audit_dataset(run.out)
    assert not report['ok'] and 'SHA256' in report['errors'][0]
    path.write_bytes(raw)
    fake=copy.deepcopy(manifest);fake['tokenizer']['fingerprint']='incorrect'
    assert not audit_dataset(run.out,manifest=fake)['ok']
    assert audit_dataset(run.out)['ok']


def test_unknown_license_exclusion_and_missing_metadata(tmp_path):
    a=jsonl(tmp_path/'license.jsonl',[*synthetic(30),{'text':next(synthetic(1,seed=95))['text']}])
    run=pipeline(tmp_path,[a],config=content(license_policy={'exclude_unknown':True}))
    run.split()
    stats=json.loads((run.out/'.pipeline/normalize.stage.json').read_text())['stats']
    assert any(r['reason']=='license_excluded' and r['n']==1 for r in stats['by_source_reason'])


def test_cli_legacy_two_stage_flow_and_heldout_guard(tmp_path,corpus_inputs):
    import subprocess,sys
    root=Path(__file__).resolve().parents[1]
    out=tmp_path/'cli'
    def run(script,*args,ok=True):
        result=subprocess.run([sys.executable,str(root/'scripts'/script),*map(str,args)],cwd=root,capture_output=True,text=True)
        assert (result.returncode==0)==ok, result.stdout+result.stderr
        return result
    run('prepare_text.py','--input',*corpus_inputs,'--out',out,'--sample',200,'--min-chars',20,'--max-chars',256)
    tok=out/'tokenizer.json'
    run('train_tokenizer.py','--zh',out/'val.text.jsonl','--out',tmp_path/'bad-tok.json','--vocab-size',512,ok=False)
    run('train_tokenizer.py','--zh',out/'train.text.jsonl','--forbid',out/'val.text.jsonl',out/'test.text.jsonl','--out',tok,'--vocab-size',512,'--max-chars',10000)
    run('prepare_text.py','--input',*reversed(corpus_inputs),'--out',out,'--tokenize-only','--tokenizer',tok)
    report=tmp_path/'audit.json'
    run('audit_dataset.py','--dataset',out,'--report',report,'--retokenize')
    assert json.loads(report.read_text())['ok']
    runtime=tmp_path/'runtime-only.json';runtime.write_text(json.dumps({'runtime':{'buffer_bytes':18}}))
    run('prepare_text.py','--out',out,'--tokenize-only','--tokenizer',tok,'--config',runtime)
    run('prepare_text.py','--out',out,'--tokenize-only','--train-tokenizer','--vocab-size',512,ok=False)
    before=(out/'train.bin').read_bytes()
    run('prepare_text.py','--out',out,'--tokenize-only','--tokenizer',tok,'--seed',4,ok=False)
    assert (out/'train.bin').read_bytes()==before
    with (out/'train.bin').open('ab') as f:f.write(b'x')
    run('audit_dataset.py','--dataset',out,ok=False)


def test_input_changed_and_corrupt_stage_are_not_reused(tmp_path,corpus_inputs):
    run=pipeline(tmp_path,corpus_inputs);run.split()
    original=corpus_inputs[0].read_bytes()
    corpus_inputs[0].write_bytes(original+b'{"text":"changed input"}\n')
    with pytest.raises(CorpusError,match='changed'):
        pipeline(tmp_path,corpus_inputs)
    corpus_inputs[0].write_bytes(original)
    (run.work/'normalize.sqlite').write_bytes(b'corrupt SQLite')
    with pytest.raises(CorpusError,match='SHA256'):
        run.split()


def test_hot_near_bucket_fails_closed_before_publication(tmp_path):
    p=jsonl(tmp_path/'dense.jsonl',synthetic(200))
    run=pipeline(tmp_path,[p],config=content(deduplication={'max_candidates':2}))
    with patch('minillm.corpus_store.simhash',return_value=0),pytest.raises(CorpusError,match='candidate budget exceeded'):
        run.split()
    assert not (run.out/'dataset-manifest.json').exists()
    assert not (run.out/'.pipeline/deduplicate.stage.json').exists()
    assert not list(run.work.glob('.*.tmp'))


def test_bad_stream_checksum_never_publishes_dataset(tmp_path,corpus_inputs,tiny_tokenizer):
    run=pipeline(tmp_path,corpus_inputs);tok=tmp_path/'tok.json';tiny_tokenizer.save(tok)
    original=tokenize_split
    def corrupt_after_write(records,tokenizer,path,**kwargs):
        stats=original(records,tokenizer,path,**kwargs)
        with path.open('r+b') as f:f.write(b'\x00\x00')
        return stats
    with patch('minillm.corpus.tokenize_split',side_effect=corrupt_after_write),pytest.raises(CorpusError,match='checksum'):
        run.tokenize(tok)
    assert not (run.out/'dataset-manifest.json').exists()
    assert not (run.out/'.pipeline/tokenize-train.stage.json').exists()


def test_known_near_dense_transitive_groups_and_distinct_prefixes(tmp_path):
    body=next(synthetic(1,length=1600))['text']
    records=[]
    for i in range(60):
        edited=body[:500]+str(i)+body[502:]
        records.append({'text':f'标题{i}\n'+edited,'source':'reposts'})
    p=jsonl(tmp_path/'dense.jsonl',records)
    run=pipeline(tmp_path,[p],config=make_content({'quality':{'min_chars':20,'max_chars':2000}}))
    run.split()
    stats=json.loads((run.out/'.pipeline/deduplicate.stage.json').read_text())['stats']
    assert stats['duplicates_near'] >= 55
    assert stats['parent_search']['max_candidates_observed']<=59


def test_audit_independently_finds_exact_cross_split_leakage(tmp_path,corpus_inputs,tiny_tokenizer):
    run=pipeline(tmp_path,corpus_inputs);tok=tmp_path/'tok.json';tiny_tokenizer.save(tok)
    manifest=run.tokenize(tok)
    row=next(read_records(run.out/'train.text.jsonl'))
    with (run.out/'val.text.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    info=file_info(run.out/'val.text.jsonl')
    forged=copy.deepcopy(manifest)
    forged['files']['val.text.jsonl']=info
    forged['identity']['files']=forged['files']
    forged['content_id']=identity(forged['identity'])
    report=audit_dataset(run.out,manifest=forged)
    assert not report['ok'] and report['exact_overlap']['train/val']>=1
    assert report['parent_isolation_violations']>=1


def test_published_dataset_identity_preserves_reliable_training_resume(tmp_path,corpus_inputs,tiny_tokenizer,nano_cfg):
    import torch
    from minillm.model import MiniLLM
    from minillm.train import TrainingOptions,fit,load_training_state,CheckpointError
    from minillm.utils import set_seed
    run=pipeline(tmp_path,corpus_inputs);tok=tmp_path/'tok.json';tiny_tokenizer.save(tok)
    published=run.tokenize(tok)
    data=PackedPretrainDataset(run.out/'train.bin',16)
    out=tmp_path/'training'
    opts=TrainingOptions(out=str(out),max_steps=2,batch_size=2,grad_accum=4,device='cpu',precision='fp32',threads=1,save_every=1,val_every=0,log_every=0)
    set_seed(opts.seed);model=MiniLLM(nano_cfg)
    original=fit(opts,model,tiny_tokenizer,data,None)
    state=load_training_state(out/'state.pt')
    assert state['manifest']['dataset']['train']['dataset_content_id']==published['content_id']
    opts.resume=str(out/'state.pt')
    resumed=fit(opts,model,tiny_tokenizer,data,None)
    assert original['tokens_seen']==resumed['tokens_seen'] and resumed['steps']==2
    before=(out/'state.pt').read_bytes()
    with data.path.open('r+b') as f:f.write(b'\x00\x00')
    with pytest.raises(CheckpointError,match='dataset.*hash mismatch'):
        fit(opts,model,tiny_tokenizer,data,None)
    assert (out/'state.pt').read_bytes()==before


def test_source_priority_and_txt_family_grouping(tmp_path):
    text=next(synthetic(1,length=600))['text']
    a=jsonl(tmp_path/'a.jsonl',[{'text':text}])
    b=jsonl(tmp_path/'b.jsonl',[{'text':text}])
    txt=tmp_path/'source.txt';txt.write_text('\n\n'.join(r['text'] for r in synthetic(20,seed=2)))
    sources=[{'name':'low','paths':[str(a)],'priority':0},
             {'name':'high','paths':[str(b)],'priority':10,'license':'declared-test'},
             {'name':'book','paths':[str(txt)]}]
    run=Pipeline(tmp_path/'sources',content(),inventory([],sources),log=lambda *_:None);run.split()
    rows=[r for name in ('train','val','test') for r in read_records(run.out/f'{name}.text.jsonl')]
    assert any(r['source']=='high' and r['license']=='declared-test' for r in rows)
    assert all(r['source']!='low' for r in rows)
    assert len({r['group_id'] for r in rows if r['source']=='book'})==1


def test_ten_thousand_documents_without_whole_token_concatenation(tmp_path,tiny_tokenizer):
    p=jsonl(tmp_path/'ten-thousand.jsonl',synthetic(10000))
    run=pipeline(tmp_path,[p]);tok=tmp_path/'tok.json';tiny_tokenizer.save(tok)
    with patch('numpy.concatenate',side_effect=AssertionError('whole dataset token concatenation forbidden')):
        manifest=run.tokenize(tok)
    assert manifest['quality']['records_seen']==10000
    assert sum(s['documents'] for s in manifest['splits'].values())==10000
    assert manifest['deduplication']['duplicates_near']==0
    assert audit_dataset(run.out)['ok']


def test_working_directory_can_change_when_verified_stages_are_reused(tmp_path,corpus_inputs,tiny_tokenizer):
    cfg=content();inputs=inventory([str(p) for p in corpus_inputs])
    run=Pipeline(tmp_path/'dataset',cfg,inputs,temp_dir=tmp_path/'first-work',log=lambda *_:None)
    with patch('minillm.corpus.deduplicate',side_effect=OSError('stop after normalization')),pytest.raises(CorpusError):
        run.split()
    resumed=Pipeline(run.out,cfg,inputs,temp_dir=tmp_path/'second-work',log=lambda *_:None)
    tok=tmp_path/'tok.json';tiny_tokenizer.save(tok)
    with patch('minillm.corpus.normalize_sources',side_effect=AssertionError('must reuse')):
        resumed.tokenize(tok)
    assert audit_dataset(resumed.out)['ok']


def test_oversize_parents_unicode_and_reserved_input_metadata(tmp_path):
    records=[*synthetic(30),{'text':'人'*220000},
             {'text':next(synthetic(1,seed=98))['text']+'\ud800'},
             {'text':next(synthetic(1,seed=99))['text'],'_issue':'injected_reason'}]
    invalid_unicode=records.pop(31)
    p=jsonl(tmp_path/'invalid.jsonl',records)
    with p.open('a') as handle:
        handle.write(json.dumps(invalid_unicode,ensure_ascii=True)+'\n')
    run=pipeline(tmp_path,[p]);run.split()
    stats=json.loads((run.out/'.pipeline/normalize.stage.json').read_text())['stats']
    reasons={row['reason']:row['n'] for row in stats['by_source_reason']}
    assert reasons['parent_too_long']==1 and reasons['invalid_unicode']==1
    assert 'injected_reason' not in reasons


def test_zero_hamming_radius_uses_full_unsigned_fingerprint(tmp_path,corpus_inputs):
    run=pipeline(tmp_path,corpus_inputs,config=content(deduplication={'hamming_distance':0}))
    run.split()
    assert (run.out/'.pipeline/split.stage.json').exists()


def test_new_tokenizer_training_is_reproducible(tmp_path,corpus_inputs):
    first=pipeline(tmp_path,corpus_inputs,'trained-first').tokenize(vocab_size=512,max_tokenizer_chars=20000)
    second=pipeline(tmp_path,list(reversed(corpus_inputs)),'trained-second').tokenize(vocab_size=512,max_tokenizer_chars=20000)
    assert first['tokenizer']['fingerprint']==second['tokenizer']['fingerprint']
    assert first['content_id']==second['content_id'] and first['files']==second['files']


def test_audit_checks_token_bounds_even_with_matching_file_hashes(tmp_path,corpus_inputs,tiny_tokenizer):
    run=pipeline(tmp_path,corpus_inputs);tok=tmp_path/'tok.json';tiny_tokenizer.save(tok)
    manifest=run.tokenize(tok)
    with (run.out/'train.bin').open('r+b') as f:f.write(b'\xff\xff')
    # Recompute file checksums as an independent manifest producer might; range
    # checking must still reject an invalid Token rather than trust those hashes.
    forged=copy.deepcopy(manifest)
    forged['files']['train.bin']=file_info(run.out/'train.bin')
    forged['identity']['files']=forged['files'];forged['content_id']=identity(forged['identity'])
    assert any('outside tokenizer' in e for e in audit_dataset(run.out,manifest=forged)['errors'])


def test_math_comparisons_and_angle_variables_are_not_stripped_as_html():
    formula='Given a < b and b > c, compare x < y > z; use <variable> notation.'
    assert normalise(formula,'unknown')==formula
    assert normalise('用<x>表示未知数，条件为 1 < x > 0。','zh')=='用<x>表示未知数，条件为 1 < x > 0。'
    assert normalise('<div><strong>说明</strong><p>模型与数学。</p></div>','zh')=='说明\n模型与数学。'


def test_missing_source_license_and_declared_category_are_explicit_unknown(tmp_path):
    rows=[{'text':r['text']} for r in synthetic(30)]
    rows.append({'text':next(synthetic(1,seed=90))['text'],'source':'\ud800','license':'\ud800'})
    path=tmp_path/'unknown.jsonl'
    path.write_text('\n'.join(json.dumps(r,ensure_ascii=True) for r in rows)+'\n')
    run=pipeline(tmp_path,[path]);run.split()
    selected=[r for name in ('train','val','test') for r in read_records(run.out/f'{name}.text.jsonl')]
    assert len(selected)==31
    assert all(r['source']=='unknown' and r['license']=='unknown' and r['declared_category']=='unknown' for r in selected)
    assert all(r['category']=='zh' for r in selected)  # inference does not pretend the input declared a category


def test_invalid_license_policy_and_config_schema_are_rejected():
    with pytest.raises(CorpusError,match='license_policy'):
        content(license_policy={'allow':'MIT'})
    with pytest.raises(CorpusError,match='schema_version'):
        content(schema_version=2)
