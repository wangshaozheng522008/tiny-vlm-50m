"""Read-only, bounded normalization diagnostic. Never rewrites a tokenizer/BIN."""
import argparse
import heapq
import hashlib
import json
from pathlib import Path

from minillm.assessment_io import report_directory, publish_manifest
from minillm.corpus_quality import classify
from minillm.corpus_utils import file_info
from minillm.data import stream_corpus_records
from minillm.text_policy import WIDTH_FOLD, apply_text_policy
from minillm.tokenizer import MiniLLMTokenizer
from minillm.training_manifest import write_json


def audit(paths, tokenizer_path, out, *, sample=512, seed=42):
    if not 1 <= sample <= 4096:
        raise ValueError('sample must be 1..4096')
    paths = sorted(set(map(lambda p: str(Path(p).resolve()), paths)))
    before = {p:file_info(Path(p)) for p in paths+[str(Path(tokenizer_path).resolve())]}
    tokenizer = MiniLLMTokenizer.load(tokenizer_path)
    heap = []
    seen = changed = malformed = too_long = 0
    for path in paths:
        for row in stream_corpus_records(path):
            if '_issue' in row:
                malformed += 1; continue
            seen += 1
            text = row['text']; category = classify(text, row.get('category'))
            folded = apply_text_policy(text, WIDTH_FOLD, category=category)
            changed += text != folded
            if len(text) > 8192:
                too_long += 1; continue
            rank = int(hashlib.sha256(f'{seed}:{text}'.encode()).hexdigest(),16)
            item = (-rank, text, category)
            if len(heap) < sample:
                heapq.heappush(heap,item)
            elif item > heap[0]:
                heapq.heapreplace(heap,item)
    raw_tokens = folded_tokens = changed_sample = 0
    for _,text,category in sorted(heap):
        folded = apply_text_policy(text, WIDTH_FOLD, category=category)
        raw_tokens += len(tokenizer.encode_document(text))
        folded_tokens += len(tokenizer.encode_document(folded))
        changed_sample += text != folded
    if not raw_tokens:
        raise ValueError('no eligible text for normalization sample')
    if any(file_info(Path(p))!=expected for p,expected in before.items()):
        raise ValueError('audit input changed')
    report = {'schema_version':1,'status':'DIAGNOSTIC_ONLY','policy':WIDTH_FOLD,
        'tokenizer_fingerprint':tokenizer.fingerprint,'tokenizer_has_internal_normalizer':tokenizer._tok.normalizer is not None,
        'inputs':before,'documents_scanned':seen,'documents_changed':changed,'malformed_records':malformed,
        'sample':{'seed':seed,'documents':len(heap),'changed_documents':changed_sample,'raw_tokens_including_eos':raw_tokens,
            'policy_tokens_including_eos':folded_tokens,'relative_token_reduction':1-folded_tokens/raw_tokens,
            'rule':'lowest seeded full-text SHA256 ranks; duplicate occurrences retained; documents <=8192 chars',
            'long_documents_excluded_from_token_sample':too_long},
        'interpretation':'Token efficiency only; does not demonstrate quality improvement. Existing checkpoints keep identity-v1. Applying the new policy requires a new model/data experiment.'}
    with report_directory(out) as directory:
        write_json(directory/'normalization-audit.json',report)
        publish_manifest(directory,'normalization-manifest.json',{'schema_version':1,'inputs':before,'policy':WIDTH_FOLD})
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',nargs='+',required=True);parser.add_argument('--tokenizer',required=True)
    parser.add_argument('--out',required=True);parser.add_argument('--sample',type=int,default=512)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    try:
        result=audit(args.input,args.tokenizer,args.out,sample=args.sample,seed=args.seed)
    except (ValueError,OSError) as exc:
        parser.exit(1,f'FAIL normalization audit: {exc}\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
