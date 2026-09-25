"""Original-only task metrics and paired, explicitly scoped hypothesis tests."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from study import SEED
from statistics_study import event_summary,paired_binary,holm

OUT=Path(__file__).resolve().parent
DATA=OUT/'timing-corrected'
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'research/gap-study-20260907'))
from metrics import video_stats,aggregate


def read_outputs(folder):
    rows={}
    for p in sorted(folder.glob('*-shard*.jsonl')):
        for line in p.read_text().splitlines():
            r=json.loads(line)
            key=(r['model'],r['case_id'])
            assert key not in rows,key
            rows[key]=r
    return rows


def bootstrap_metrics(a,b):
    assert len(a)==len(b)>0
    keys=['index_miou','segment_count_accuracy_all','fixed_gt_index_joint_recall','matched_emotion_f1']
    rng=np.random.default_rng(SEED)
    effects={k:[] for k in keys}
    for _ in range(5000):
        indices=rng.integers(0,len(a),len(a))
        aa=aggregate([a[i] for i in indices])
        bb=aggregate([b[i] for i in indices])
        for k in keys:
            effects[k].append(aa[k]-bb[k])
    aa,bb=aggregate(a),aggregate(b)
    return {k:dict(difference=aa[k]-bb[k],ci95=np.quantile(effects[k],[.025,.975]).tolist(),
                   ci90=np.quantile(effects[k],[.05,.95]).tolist()) for k in keys}


def prior_from_train():
    path=ROOT/'data/DAR-R1-annotations/train.jsonl'
    segments=[json.loads(json.loads(x)['conversations'][1]['value'])['segments'] for x in path.read_text().splitlines() if x.strip()]
    mode=Counter(map(len,segments)).most_common(1)[0][0]
    labels=[Counter(ss[i]['emotion'] for ss in segments if len(ss)==mode).most_common(1)[0][0] for i in range(mode)]
    assert mode==2 and labels==['Interest','Amusement']
    return dict(mode=mode,labels=labels,train_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    manifest=json.loads((DATA/'manifest.json').read_text())
    natural=[v for v in manifest['videos'] if v['kind']=='natural']
    current=read_outputs(DATA)
    assert len(current)==736, len(current)
    legacy=read_outputs(OUT)
    prior=prior_from_train()
    control=[video_stats([dict(start_time=i*v['video_duration']/2,end_time=(i+1)*v['video_duration']/2,emotion=prior['labels'][i]) for i in range(2)],v['gt_segments']) for v in natural]
    result=dict(seed=SEED,sample_size=len(natural),gt_segments=sum(len(v['gt_segments']) for v in natural),
                sample_gt_count_distribution=dict(Counter(len(v['gt_segments']) for v in natural)),
                current_records=len(current),legacy_records=len(legacy),prior=dict(**prior,metrics=aggregate(control)),
                original_task_metrics={},legacy_original_task_metrics={},timing_effects={},original_dar_minus_prior={})
    all_stats={}
    for model in ('dar','qwen'):
        for prompt in ('released','conservative'):
            name=f'{model}:{prompt}'
            stats=[video_stats(current[(model,f"{v['video_id']}:original:{prompt}")]['segments'],v['gt_segments']) for v in natural]
            all_stats[name]=stats
            result['original_task_metrics'][name]=aggregate(stats)
            if len(legacy)==736:
                old=[video_stats(legacy[(model,f"{v['video_id']}:original:{prompt}")]['segments'],v['gt_segments']) for v in natural]
                result['legacy_original_task_metrics'][name]=aggregate(old)
                result['timing_effects'][name]=bootstrap_metrics(stats,old)
    result['original_dar_minus_prior']=bootstrap_metrics(all_stats['dar:released'],control)
    ci90=result['original_dar_minus_prior']['index_miou']['ci90']
    result['miou_practical_equivalence_within_5pp']=ci90[0]>-.05 and ci90[1]<.05
    result['original_dar_minus_qwen']=bootstrap_metrics(all_stats['dar:released'],all_stats['qwen:released'])
    result['prompt_task_effects']={m:bootstrap_metrics(all_stats[f'{m}:conservative'],all_stats[f'{m}:released']) for m in ('dar','qwen')}
    result['output_status']={}
    result['output_status_by_condition']={}
    for model in ('dar','qwen'):
        for condition in ('original','static_middle','gray','gray_to_white'):
            for prompt in ('released','conservative','factual'):
                selected=[r for (m,_),r in current.items() if m==model and r['condition']==condition and r['prompt_kind']==prompt]
                if not selected:continue
                issues=[]
                for r in selected:
                    if prompt=='factual':continue
                    flags=[]
                    if not r['segments']:flags.append('empty_segments')
                    if any(s['end_time']<=s['start_time'] for s in r['segments']):flags.append('nonpositive_interval')
                    if any(s['start_time']<0 or s['end_time']>r['video_duration']+1e-6 or s['start_time']>r['video_duration']+1e-6 for s in r['segments']):flags.append('out_of_bounds')
                    if flags:issues.append(dict(case_id=r['case_id'],flags=flags,segments=r['segments']))
                result['output_status_by_condition'][f'{model}:{condition}:{prompt}']=dict(n=len(selected),
                    nonempty_normalized_segments=sum(bool(r['segments']) for r in selected) if prompt!='factual' else None,
                    temporal_interval_issues=issues,stop_counts=dict(Counter(r['finish_reason'] for r in selected)))
    result['factual']={}
    factual_coding={}
    fcp=DATA/'factual-coding.jsonl'
    if fcp.exists():
        fk=json.loads((DATA/'factual-key.json').read_text())
        for line in fcp.read_text().splitlines():
            fc=json.loads(line)
            ident=fk[fc['blind_id']]
            key=(ident['model'],ident['case_id'])
            assert key not in factual_coding
            assert fc['judgment'] in ('correct','incorrect','ambiguous','unassessable')
            factual_coding[key]=fc
        assert len(factual_coding)==176
    result['factual_coding_status']='complete' if len(factual_coding)==176 else 'pending'
    for model in ('dar','qwen'):
        for kind in ('natural','gray','step'):
            for prompt in ('released','conservative','factual'):
                selected=[r for (m,_),r in current.items() if m==model and r['kind']==kind and r['prompt_kind']==prompt]
                if not selected:
                    continue
                name=f'{model}:{kind}:{prompt}'
                result['output_status'][name]=dict(n=len(selected),valid_segments=sum(bool(r['segments']) for r in selected),
                    stop_counts=dict(Counter(r['finish_reason'] for r in selected)),
                    total_generated_tokens=sum(r['generated_tokens'] for r in selected))
                if prompt=='factual':
                    expected=kind=='step'
                    valid=[r for r in selected if isinstance(r['parsed'],dict) and type(r['parsed'].get('visible_change')) is bool]
                    correct=[r for r in valid if r['parsed']['visible_change']==expected]
                    result['factual'][f'{model}:{kind}']=dict(n=len(selected),valid=len(valid),boolean_correct=len(correct),expected_visible_change=expected,
                        boolean_and_evidence_correct=sum(factual_coding.get((model,r['case_id']),{}).get('judgment')=='correct' for r in selected) if factual_coding else None,
                        rows=[dict(video_id=r['video_id'],parsed=r['parsed'],raw_output=r['raw_output']) for r in selected])
    coding_path=DATA/'coding-final.jsonl'
    provisional_coding=False
    if not coding_path.exists() and (DATA/'coding-primary.jsonl').exists():
        coding_path=DATA/'coding-primary.jsonl'
        provisional_coding=True
    if not coding_path.exists():
        result['coding_status']='pending; no event-rate inference yet'
    else:
        key=json.loads((DATA/'coding-key.json').read_text())
        coding={}
        for line in coding_path.read_text().splitlines():
            r=json.loads(line)
            k=key[r['blind_id']]
            k=(k['model'],k['case_id'])
            assert k not in coding
            assert r['label'] in ('confirmed','none','ambiguous','unassessable')
            if r['label']=='confirmed':
                assert r['quote'] and r['quote'] in current[k]['raw_output']
            coding[k]=r
        assert len(coding)==304
        result['coding_status']='primary AI coding only; independent critic failed quality gate; provisional' if provisional_coding else 'complete'
        result['coding_inference_status']='descriptive/exploratory only; p-values and intervals conditional on provisional labels, not confirmatory validation' if provisional_coding else 'reviewed'
        result['event_rates']={}
        vectors={}
        for model in ('dar','qwen'):
            for prompt in ('released','conservative'):
                name=f'{model}:{prompt}'
                labels=[coding[(model,f"{v['video_id']}:static_middle:{prompt}")]['label'] for v in natural]
                vectors[name]=labels
                result['event_rates'][name]=event_summary(labels)
        pairs=[('dar_prompt','dar:released','dar:conservative'),('qwen_prompt','qwen:released','qwen:conservative'),('model_released','dar:released','qwen:released')]
        result['event_contrasts']={}
        pvalues=[]
        for name,a,b in pairs:
            v=paired_binary([int(x=='confirmed') for x in vectors[a]],[int(x=='confirmed') for x in vectors[b]])
            pvalues.append(v['mcnemar_exact_p'])
            clean=[i for i,(x,y) in enumerate(zip(vectors[a],vectors[b])) if x in ('confirmed','none') and y in ('confirmed','none')]
            v['decidable_pairs_sensitivity']=paired_binary([int(vectors[a][i]=='confirmed') for i in clean],[int(vectors[b][i]=='confirmed') for i in clean]) if clean else None
            result['event_contrasts'][name]=v
        for (name,_,_),adj in zip(pairs,holm(pvalues)):
            result['event_contrasts'][name]['holm_p']=adj
        result['gray_grid']={}
        result['factual_reasoning_cross_tabs']={}
        for model in ('dar','qwen'):
            for prompt in ('released','conservative'):
                selected=[r for (m,_),r in current.items() if m==model and r['condition']=='gray' and r['prompt_kind']==prompt]
                result['gray_grid'][f'{model}:{prompt}']=[dict(duration=r['video_duration'],segments=len(r['segments']),label=coding[(model,r['case_id'])]['label'],blind_id=coding[(model,r['case_id'])]['blind_id']) for r in sorted(selected,key=lambda r:r['video_duration'])]
                for kind,condition in [('natural','static_middle'),('gray','gray')]:
                    entries=[]
                    for v in [v for v in manifest['videos'] if v['kind']==kind]:
                        ident=(model,f"{v['video_id']}:{condition}:factual")
                        r=current[ident]
                        valid=isinstance(r['parsed'],dict) and type(r['parsed'].get('visible_change')) is bool
                        entries.append(dict(video_id=v['video_id'],boolean_correct=valid and r['parsed']['visible_change'] is False,
                            factual_correct=factual_coding.get(ident,{}).get('judgment')=='correct' if factual_coding else None,
                            event_confirmed=coding[(model,f"{v['video_id']}:{condition}:{prompt}")]['label']=='confirmed'))
                    result['factual_reasoning_cross_tabs'][f'{model}:{prompt}:{kind}']=dict(n=len(entries),
                        factual_correct=sum(x['factual_correct'] for x in entries) if factual_coding else None,
                        factual_correct_and_event_error=sum(x['factual_correct'] and x['event_confirmed'] for x in entries) if factual_coding else None,
                        rows=entries)
    (OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('sample_size','gt_segments','current_records','legacy_records','coding_status','event_rates','miou_practical_equivalence_within_5pp')},indent=2))


if __name__=='__main__':
    main()
