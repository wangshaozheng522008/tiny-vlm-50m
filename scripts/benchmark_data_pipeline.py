#!/usr/bin/env python3
"""Synthetic CPU corpus benchmark; memory measurements run in fresh child processes.

Default: 10,000 documents. Extended: --documents 10000 100000.
RSS numbers are measurements, not CI pass/fail thresholds or a throughput promise.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import resource
import string
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def duplicate_quality_cases(out):
    from minillm.corpus import Pipeline, inventory, make_content, read_records
    rng=random.Random(5)
    chars='人工智能模型语言训练数据数学推理科学技术方法世界生活学习探索结果网络算法计算程序函数优化研究分析测试中文英文知识问题答案信息系统任务目标发展历史未来文化'
    body=''.join(rng.choices(chars,k=1200))+'。编号0。'
    other=''.join(random.Random(500).choices(chars,k=1200))+'。另一文章。'
    cases=[('identical',body,body,True),('whitespace','标题\n'+body,'标题  '+body,True),
           ('light_edit',body,body[:600]+'科'+body[601:],True),
           ('new_title','原标题\n'+body,'新标题\n'+body,True),
           ('same_prefix_different_body',body,body[:256]+other,False),
           ('cross_source_repost',body,body,True),('different_content',body,other,False)]
    code='def classify_value(value):\n    if value:\n        return 1\n    return 0\n'
    cases.extend([('cross_category_code',code,code,True),
                  ('code_indent_variants',code,code.replace('    return 0','        return 0'),False)])
    results=[]
    for name,a,b,expected in cases:
        raw=out/(name+'.jsonl')
        records=[{'text':text,'source':source,'license':'synthetic-only'} for text,source in ((a,'original'),(b,'repost'))]
        if name in ('cross_category_code','code_indent_variants'):
            records[0]['category']='code'
            records[1]['category']='zh' if name=='cross_category_code' else 'code'
        raw.write_text('\n'.join(json.dumps(record,ensure_ascii=False) for record in records)+'\n')
        pipeline=Pipeline(out/name,make_content({'split':[1,0,0],'quality':{'min_chars':20,'max_chars':4000}}),inventory([str(raw)]),log=lambda *_:None)
        pipeline.split()
        kept=sum(1 for _ in read_records(pipeline.out/'train.text.jsonl'))
        stats=json.loads((pipeline.out/'.pipeline/deduplicate.stage.json').read_text())['stats']
        results.append({'case':name,'expected_duplicate':expected,'detected_duplicate':kept==1,
                        'exact_removed':stats['duplicates_exact'],'near_removed':stats['duplicates_near']})
    tp=sum(r['expected_duplicate'] and r['detected_duplicate'] for r in results)
    fp=sum(not r['expected_duplicate'] and r['detected_duplicate'] for r in results)
    fn=sum(r['expected_duplicate'] and not r['detected_duplicate'] for r in results)
    return {'cases':results,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),
            'scope':'Nine synthetic labeled pairs only; not an estimate of real-world precision/recall.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--documents',type=int,nargs='+',default=[10000])
    p.add_argument('--out',type=Path,help='dedicated empty benchmark directory')
    p.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.worker:
        from minillm.corpus import Pipeline, inventory, make_content
        from minillm.corpus_audit import audit_dataset
        from minillm.corpus_utils import write_json
        import torch
        torch.set_num_threads(1)
        baseline_peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        baseline_rss=None
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):baseline_rss=int(line.split()[1])/1024
        started=time.monotonic()
        work=args.worker
        logs=[]
        pipeline=Pipeline(work/'dataset',make_content({'quality':{'min_chars':20,'max_chars':1024}}),
                          inventory([str(work/'raw.jsonl')]),log=logs.append)
        manifest=pipeline.tokenize(work.parent/'tokenizer.json')
        pipeline_seconds=time.monotonic()-started
        audit=audit_dataset(work/'dataset',retokenize=True)
        if not audit['ok']:raise RuntimeError(audit['errors'])
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        stages={path.name:json.loads(path.read_text())['seconds'] for path in (work/'dataset/.pipeline').glob('*.stage.json')}
        report={'input_documents':manifest['quality']['records_seen'],'published_documents':sum(s['documents'] for s in manifest['splits'].values()),
                'tokens':sum(s['tokens'] for s in manifest['splits'].values()),'pipeline_seconds':round(pipeline_seconds,3),
                'documents_per_second':round(manifest['quality']['records_seen']/pipeline_seconds,2),
                'tokens_per_second':round(sum(s['tokens'] for s in manifest['splits'].values())/pipeline_seconds,2),
                'baseline_peak_rss_mib':round(baseline_peak,2),'baseline_idle_rss_mib':round(baseline_rss,2),
                'peak_rss_mib':round(peak,2),'incremental_peak_rss_mib':round(max(0,peak-baseline_peak),2),
                'peak_minus_idle_rss_mib':round(max(0,peak-baseline_rss),2),'stages_seconds':stages,
                'exact_overlap':audit['exact_overlap'],'category_independent_overlap':audit['category_independent_overlap'],
                'raw_overlap':audit['raw_overlap'],'known_near_duplicate_cross_split_groups':audit['known_near_duplicate_cross_split_groups'],
                'content_id':manifest['content_id'],'deduplication':manifest['deduplication'],
                'environment':{'python':sys.version.split()[0],'platform':sys.platform,'torch':torch.__version__,
                               'sqlite_cache_mib':16,'threads':1,'measurement':'Linux ru_maxrss; fresh child, imports before baseline; full audit included in peak'}}
        write_json(work/'benchmark.json',report)
        (work/'pipeline.log').write_text('\n'.join(logs))
        print(json.dumps(report,indent=2))
        return 0
    from minillm.tokenizer import MiniLLMTokenizer
    from minillm.corpus_utils import write_json
    if any(n<=0 for n in args.documents):p.error('document counts must be positive')
    out=args.out or Path(tempfile.mkdtemp(prefix='minillm-data-benchmark-'))
    if out.exists() and any(out.iterdir()):p.error('--out must be empty')
    out.mkdir(parents=True,exist_ok=True)
    MiniLLMTokenizer.train(['independent synthetic bootstrap training text']*10,vocab_size=264).save(out/'tokenizer.json')
    quality=duplicate_quality_cases(out)
    reports=[]
    chinese='人工智能模型语言训练数据数学推理科学技术方法世界生活学习探索结果网络算法计算程序函数优化研究分析测试中文英文知识问题答案信息系统任务目标发展历史未来文化'
    for n in args.documents:
        case=out/str(n);case.mkdir()
        rng=random.Random(42)
        with (case/'raw.jsonl').open('w') as handle:
            for i in range(n):
                pick=rng.random()
                if pick<.8:
                    category='zh';text=''.join(rng.choices(chinese,k=160))+f'。文档{i}。'
                elif pick<.95:
                    category='en';text=' '.join(''.join(rng.choices(string.ascii_lowercase,k=8)) for _ in range(20))+f' record {i}.'
                else:
                    category='code';text=f'def function_{i}(value):\n    # '+''.join(rng.choices(string.ascii_letters,k=160))+'\n    return value + 1\n'
                handle.write(json.dumps({'text':text,'category':category,'source':category,'license':'synthetic-only'},ensure_ascii=False)+'\n')
        subprocess.run([sys.executable,__file__,'--worker',str(case)],check=True,env={**os.environ,'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        reports.append(json.loads((case/'benchmark.json').read_text()))
    write_json(out/'benchmark-report.json',{'cases':reports,'dedup_quality':quality,'notes':'Synthetic only; throughput depends on hardware, text length and candidate density. No absolute RSS CI assertion.'})
    print(f'PASS corpus benchmark: {out}/benchmark-report.json')
    return 0


if __name__=='__main__':raise SystemExit(main())
