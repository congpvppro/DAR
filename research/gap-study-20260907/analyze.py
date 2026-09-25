"""Re-score immutable baseline files and controlled pilot; all comparisons exploratory."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
from metrics import aggregate, video_stats, iou, label

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def read_gt(split):
    return {Path(r['video']).stem: dict(segments=json.loads(r['conversations'][1]['value'])['segments'],duration=r['video_duration'])
            for r in read_jsonl(ROOT/f'data/DAR-R1-annotations/{split}.jsonl')}


def read_predictions(paths):
    pred, seen, provenance = {}, set(), []
    for path in paths:
        rows = read_jsonl(path)
        provenance.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),physical_rows=len(rows)))
        for row in rows:
            vid = row['video_id']
            seen.add(vid)
            if row.get('segments'):
                pred[vid] = row
    return pred, seen, provenance


def bootstrap(rows, metric, other=None, repeats=2000):
    rng = np.random.default_rng(20260907)
    specs = {'conditional_emotion_accuracy':('joint','qualified'),
             'fixed_gt_index_joint_recall':('joint','gt'),
             'matched_emotion_recall':('tp','gt'),
             'index_miou':('iou_sum','pairs'),
             'segment_count_accuracy_all':('count_correct',None)}
    num, den = specs[metric]
    def arrays(rs):
        return np.array([r[num] for r in rs]), np.array([r[den] if den else 1 for r in rs])
    a,b = arrays(rows)
    if other is not None:
        c,d = arrays(other)
    values=[]
    for _ in range(repeats):
        idx=rng.integers(0,len(rows),len(rows))
        val=a[idx].sum()/max(b[idx].sum(),1)
        if other is not None:
            val-=c[idx].sum()/max(d[idx].sum(),1)
        values.append(float(val))
    return [float(v) for v in np.quantile(values,[.025,.975])]


def summarize(pred, gt):
    ids=sorted(gt)
    rows=[video_stats(pred.get(vid,{}).get('segments',[]),gt[vid]['segments']) for vid in ids]
    result=aggregate(rows)
    result['failed_ids']=[v for v in ids if v not in pred or not pred[v].get('segments')]
    result['strata_by_gt_count']={str(n):aggregate([r for r in rows if r['gt_count']==n]) for n in sorted({r['gt_count'] for r in rows})}
    result['ci95']={k:bootstrap(rows,k) for k in ['conditional_emotion_accuracy','fixed_gt_index_joint_recall','matched_emotion_recall']}
    result['predicted_emotion_counts']=dict(Counter(label(s) for r in pred.values() for s in r.get('segments',[])))
    result['gt_emotion_counts']=dict(Counter(label(s) for r in gt.values() for s in r['segments']))
    result['invalid_time_segment_count']=sum(not(0 <= float(s['start_time']) < float(s['end_time']) <= gt[vid]['duration']+.5)
                                             for vid,r in pred.items() if vid in gt for s in r.get('segments',[]))
    return result,rows


def reward(reasons):
    values=[]
    previous=set()
    for reason in reasons:
        words=reason.split()
        current=set(words)
        jac=len(current&previous)/len(current|previous) if current|previous else 0
        values.append(math.exp(-abs(len(words)-120)/40)*(1-int(jac>.75)))
        previous=current
    return sum(values)/len(values) if values else 0


def main():
    gt=read_gt('test')
    train=read_gt('train')
    darbase=ROOT/'outputs/paper-repro-gpu01'
    qbase=ROOT/'outputs/baselines-gpu0123/qwen25-vl-3b'
    abase=ROOT/'outputs/baselines-gpu0123/affectgpt-paper-aligned'
    files={
        'dar':sorted((darbase/'strict-4096').glob('predictions_shard_*.jsonl'))+sorted((darbase/'retry-8192').glob('predictions_retry_shard_*.jsonl')),
        'qwen':sorted((qbase/'strict-4096').glob('predictions_shard_*.jsonl'))+sorted((qbase/'retry-8192').glob('predictions*.jsonl'),key=lambda p:(3 if 'retry3' in p.name else 2 if 'retry2' in p.name else 1,p.name)),
        'affectgpt':sorted((abase/'strict-4096').glob('predictions_shard_*.jsonl'))}
    output={'date':'2026-09-07','baseline_origin':'pre-existing inference, freshly rescored','full_test':{},'dataset':{},'controls':{}}
    predictions,all_rows={},{}
    for name,paths in files.items():
        pred,seen,provenance=read_predictions(paths)
        assert not (seen-set(gt)),(name,seen-set(gt))
        predictions[name]=pred
        result,rows=summarize(pred,gt)
        result['sources']=provenance
        output['full_test'][name]=result
        all_rows[name]=rows
    output['paired_dar_minus_qwen_ci95']={k:bootstrap(all_rows['dar'],k,all_rows['qwen']) for k in ['conditional_emotion_accuracy','fixed_gt_index_joint_recall','matched_emotion_recall','index_miou']}
    output['dataset']={split:dict(videos=len(data),segments=sum(len(r['segments']) for r in data.values()),counts=dict(Counter(len(r['segments']) for r in data.values())),
                                emotion_counts=dict(Counter(label(s) for r in data.values() for s in r['segments']))) for split,data in [('train',train),('test',gt)]}
    train_counts=Counter(len(r['segments']) for r in train.values())
    mode_count=train_counts.most_common(1)[0][0]
    majority=Counter(label(s) for r in train.values() for s in r['segments']).most_common(1)[0][0]
    position_labels=[]
    for index in range(mode_count):
        counts=Counter(label(r['segments'][index]) for r in train.values() if len(r['segments'])==mode_count)
        # Enforce the released adjacent-different-label constraint using training priors only.
        chosen=next(l for l,n in counts.most_common() if not position_labels or l!=position_labels[-1])
        position_labels.append(chosen)
    for name,labels in [('train_majority_one_segment',[majority]),('train_mode_equal_bins',position_labels)]:
        pred={vid:dict(segments=[dict(start_time=i*r['duration']/len(labels),end_time=(i+1)*r['duration']/len(labels),emotion=e) for i,e in enumerate(labels)]) for vid,r in gt.items()}
        stats,control_rows=summarize(pred,gt)
        output['controls'][name]=dict(labels=labels,metrics=stats,uses='training label/count priors and test duration only; no video or test labels',
            paired_dar_minus_control_ci95={k:bootstrap(all_rows['dar'],k,control_rows) for k in ['index_miou','segment_count_accuracy_all','fixed_gt_index_joint_recall','matched_emotion_recall']})
    for name,pred in predictions.items():
        first={v:dict(segments=r['segments'][:1]) for v,r in pred.items()}
        stats,_=summarize(first,gt)
        output['controls'][name+'_drop_trailing']=dict(metrics=stats,uses='diagnostic only; deliberately violates full-video coverage; no ground truth read by the truncation transform')
    # Eq. 8 has no access to the image or meaning of the words: bijective relabeling preserves its inputs.
    differences=[]
    examples=[]
    for vid,row in predictions['dar'].items():
        reasons=[s.get('reason','') for s in row['segments']]
        vocabulary={w:f'zz{i}' for i,w in enumerate(sorted(set(' '.join(reasons).split())))}
        replaced=[' '.join(vocabulary[w] for w in reason.split()) for reason in reasons]
        before,after=reward(reasons),reward(replaced)
        differences.append(abs(before-after))
        if len(examples)<2:
            examples.append(dict(video_id=vid,before=before,after=after,original=reasons[0],substituted=replaced[0]))
    output['reason_reward_invariance']=dict(videos=len(differences),max_absolute_change=max(differences),examples=examples,
        caveat='Analytical word-level Eq.8 diagnostic, not a trained-policy ablation or proof of reward hacking; tokenization convention uses whitespace.')
    manifest=json.loads((OUT/'manifest.json').read_text())
    pilot_gt={r['video_id']:dict(segments=r['gt_segments'],duration=r['video_duration']) for r in manifest['videos']}
    output['pilot']={}
    for model in ['dar','qwen']:
        paths=sorted(OUT.glob(f'{model}-shard[01].jsonl'))
        raw=[r for p in paths for r in read_jsonl(p)]
        if not raw:
            continue
        conditions={c:{r['video_id']:r for r in raw if r['condition']==c} for c in manifest['conditions']}
        counts={c:len(rs) for c,rs in conditions.items()}
        result=dict(physical_rows=len(raw),expected_rows=192,counts=counts,complete=all(n==48 for n in counts.values()),conditions={},paired={})
        condition_rows={}
        for condition,pred in conditions.items():
            stats,rows=summarize(pred,pilot_gt)
            stats['finish_reasons']=dict(Counter(r['finish_reason'] for r in pred.values()))
            stats['mean_generated_tokens']=float(np.mean([r['generated_tokens'] for r in pred.values()])) if pred else 0
            stats['multi_phase_predictions']=sum(len(r['segments'])>1 for r in pred.values())
            stats['mean_reason_words']=float(np.mean([len(s.get('reason','').split()) for r in pred.values() for s in r['segments']])) if any(r['segments'] for r in pred.values()) else 0
            result['conditions'][condition]=stats
            condition_rows[condition]=rows
        original=conditions['original']
        for condition in manifest['conditions'][1:]:
            modified=conditions[condition]
            common=[v for v in pilot_gt if v in original and v in modified and original[v]['segments'] and modified[v]['segments']]
            unchanged_seq=sum([label(s) for s in original[v]['segments']]==[label(s) for s in modified[v]['segments']] for v in common)
            unchanged_count=sum(len(original[v]['segments'])==len(modified[v]['segments']) for v in common)
            result['paired'][condition]=dict(valid_pairs=len(common),unchanged_label_sequence_count=unchanged_seq,unchanged_count_count=unchanged_count,
                original_minus_modified_ci95={k:bootstrap(condition_rows['original'],k,condition_rows[condition]) for k in ['fixed_gt_index_joint_recall','index_miou','segment_count_accuracy_all']},
                interpretation='Original-label retention/sensitivity only; transformed videos have no counterfactual annotations.')
        output['pilot'][model]=result
    affect_path=OUT/'affectgpt-fresh.jsonl'
    if affect_path.exists():
        affect_rows=read_jsonl(affect_path)
        output['affectgpt_fresh']=dict(videos=len(affect_rows),valid_count=sum(bool(r['segments']) for r in affect_rows),
            representation_comparisons=[{k:r[k] for k in ['video_id','reversed_feature_max_abs_delta','reversed_feature_relative_l2','reversed_feature_cosine','feature_shape','frames_shape']} for r in affect_rows],
            caveat='Native 8-frame interface, four videos, original generation only; reverse tests compare visual embeddings, not generated texts.')
    (OUT/'analysis.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(full_test={k:{m:v[m] for m in ['valid_count','conditional_emotion_accuracy','index_miou','matched_emotion_f1','fixed_gt_index_joint_recall']} for k,v in output['full_test'].items()},
                         pilot={k:dict(complete=v['complete'],counts=v['counts']) for k,v in output['pilot'].items()}),indent=2))


if __name__=='__main__':
    main()
