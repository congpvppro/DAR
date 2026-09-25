"""Independent comparison with actual released evaluator functions and artifact checks."""
import ast
import hashlib
import json
from pathlib import Path
import typing
import numpy as np
from analyze import read_gt, read_predictions
from metrics import aggregate, video_stats

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
source=ROOT/'DAR/test.py'
tree=ast.parse(source.read_text())
names={'calculate_iou','evaluate_single_video','compute_overall_metrics'}
tree.body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
namespace={'Dict':typing.Dict,'List':typing.List}
exec(compile(tree,str(source),'exec'),namespace)
gt=read_gt('test')
analysis=json.loads((OUT/'analysis.json').read_text())
checked={}
for model,result in analysis['full_test'].items():
    paths=[ROOT/r['path'] for r in result['sources']]
    pred,seen,provenance=read_predictions(paths)
    assert provenance==result['sources'],'Existing prediction artifact changed'
    reference=namespace['compute_overall_metrics']({v:namespace['evaluate_single_video'](r['segments'],gt[v]['segments']) for v,r in pred.items()})
    for ours,theirs in [('index_miou','avg_iou'),('segment_count_accuracy_valid','segment_count_match_rate'),('unconditioned_emotion_accuracy','emotion_accuracy')]:
        assert abs(result[ours]-reference[theirs])<1e-12,(model,ours,result[ours],reference[theirs])
    assert result['compared_pairs']==reference['total_compared_segments']
    checked[model]={'official_metrics_match':True,'valid_ids':len(pred)}
manifest=json.loads((OUT/'manifest.json').read_text())
ids={r['video_id'] for r in manifest['videos']}
assert len(ids)==48
assert hashlib.sha256((ROOT/'data/DAR-R1-annotations/test.jsonl').read_bytes()).hexdigest()==manifest['test_sha256']
pilot={}
all_inputs={}
for model in ['dar','qwen']:
    records=[r for p in sorted(OUT.glob(f'{model}-shard[01].jsonl')) for r in map(json.loads,p.read_text().splitlines())]
    indexed={(r['video_id'],r['condition']):r for r in records}
    assert len(records)==len(indexed)==192
    assert set(indexed)=={(v,c) for v in ids for c in manifest['conditions']}
    for vid in ids:
        conditions=[indexed[vid,c] for c in manifest['conditions']]
        assert len({tuple(r['input_shape']) for r in conditions})==1
        assert all(r['input_shape'][0]==16 for r in conditions)
        assert len({r['prompt_tokens'] for r in conditions})==1
        # A gray intervention must have a different input from the actual video.
        assert indexed[vid,'gray']['input_sha256']!=indexed[vid,'original']['input_sha256']
        expected_gray=np.full(indexed[vid,'gray']['input_shape'],127.5,dtype=np.float32)
        assert hashlib.sha256(expected_gray.tobytes()).hexdigest()==indexed[vid,'gray']['input_sha256']
    all_inputs[model]={k:r['input_sha256'] for k,r in indexed.items()}
    pilot[model]={'unique_condition_video_pairs':len(indexed),'parse_success':sum(bool(r['segments']) for r in records)}
assert all_inputs['dar']==all_inputs['qwen'],'Paired models received different video tensors'
exits=json.loads((OUT/'worker-exits.json').read_text())
assert len(exits)==4 and all(x['exit_code']==0 for x in exits)
affect=[json.loads(x) for x in (OUT/'affectgpt-fresh.jsonl').read_text().splitlines()]
affect_ids={Path(x['video']).stem for x in map(json.loads,(OUT/'affectgpt-fresh-input.jsonl').read_text().splitlines())}
assert len(affect)==len(affect_ids)==4
assert {r['video_id'] for r in affect}==affect_ids
assert all(r['segments'] and r['frames_shape'][1]==8 for r in affect)
assert all(r['reversed_feature_max_abs_delta']==0 and r['reversed_feature_relative_l2']==0 for r in affect)
result={'full_test':checked,'pilot':pilot,'identical_visual_inputs_across_models':True,
        'gray_inputs_verified_uniform_float32':True,
        'affectgpt_fresh':{'video_count':4,'valid_count':4,'all_reversed_features_exactly_equal':True},
        'official_code_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'test_gt_sha256':manifest['test_sha256'],'worker_exits_all_zero':True}
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
