"""Audit current artifacts, input provenance, and released evaluator agreement."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import typing

from analyze import read_outputs, video_stats, aggregate
from study import build_cases, CONSERVATIVE, FACTUAL

OUT = Path(__file__).resolve().parent
DATA = OUT / 'timing-corrected'
ROOT = OUT.parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def main():
    frozen = json.loads((OUT/'manifest.json').read_text())
    manifest = json.loads((DATA/'manifest.json').read_text())
    assert sha(OUT/'manifest.json') == manifest['selection_manifest_sha256']
    assert sha(OUT/'protocol.md') == frozen['protocol_sha256']
    assert sha(DATA/'protocol.md') == manifest['protocol_sha256']
    assert frozen['videos'] == manifest['videos']
    assert sha(ROOT/'data/DAR-R1-annotations/test.jsonl') == manifest['source_test_sha256']
    assert manifest['prompts'] == dict(conservative_append=CONSERVATIVE, factual=FACTUAL)
    natural = [v for v in manifest['videos'] if v['kind']=='natural']
    ids = {v['video_id'] for v in natural}
    old_ids = {v['video_id'] for v in json.loads((ROOT/'research/gap-study-20260907/manifest.json').read_text())['videos']}
    assert len(ids)==64 and len(old_ids)==48 and ids.isdisjoint(old_ids)
    assert set(manifest['excluded_prior_ids']) == old_ids
    expected = {(m,c['case_id']) for m in ('dar','qwen') for c in build_cases(manifest['videos'])}
    assert len(expected)==736
    current, legacy = read_outputs(DATA), read_outputs(OUT)
    assert set(current)==set(legacy)==expected
    exits_checked = 0
    for folder in (OUT, DATA):
        exits = json.loads((folder/'worker-exits.json').read_text())
        assert len(exits)==4 and all(r['exit_code']==0 for r in exits)
        exits_checked += len(exits)
        for path in folder.glob('*-shard*-metadata.json'):
            meta = json.loads(path.read_text())
            assert meta['manifest_sha256'] == sha(folder/'manifest.json')
            for name,digest in meta['source_hashes'].items():
                assert sha(ROOT/name) == digest, name
    for key,row in current.items():
        other = current[('qwen' if key[0]=='dar' else 'dar',key[1])]
        for field in ('input_sha256','input_shape','video_metadata','prompt_sha256','prompt_tokens','mm_processor_kwargs'):
            assert row[field]==other[field], (key,field)
        for field in ('input_sha256','input_shape','video_metadata','prompt_sha256','prompt_tokens','prompt'):
            assert row[field]==legacy[key][field], (key,field)
        assert hashlib.sha256(row['prompt'].encode()).hexdigest()==row['prompt_sha256']
        assert row['mm_processor_kwargs'] == dict(do_sample_frames=False,fps=[row['sampled_fps']])
        assert legacy[key]['mm_processor_kwargs'] == dict(do_sample_frames=False)
        assert abs(row['expected_second_per_grid_ts']-2/row['sampled_fps'])<1e-12
        assert row['generated_tokens'] <= (256 if row['prompt_kind']=='factual' else 4096)
    inp = json.loads((OUT/'input-verification.json').read_text())
    assert inp['status']=='passed' and inp['videos']==88 and inp['records_verified']==736
    verified_keys = set()
    for video in inp['results']:
        for image in video['inputs']:
            related = [(key,row) for key,row in current.items() if row['video_id']==video['video_id'] and row['condition']==image['condition']]
            assert len(related)==image['matching_records']
            for key,row in related:
                assert row['input_sha256']==image['raw_sha256']
                assert row['input_shape']==image['shape']
                assert row['sampled_fps']==video['sampled_fps']
                verified_keys.add(key)
    assert verified_keys == expected

    coding_checks = {}
    for prefix,total,field in [('coding',304,'label'),('factual',176,'judgment')]:
        blind = {r['blind_id']:r for r in jsonl(DATA/f'{prefix}-blind.jsonl')}
        key = json.loads((DATA/f'{prefix}-key.json').read_text())
        name = ('coding-final.jsonl' if (DATA/'coding-final.jsonl').exists() else 'coding-primary.jsonl') if prefix=='coding' else 'factual-coding.jsonl'
        labels = jsonl(DATA/name)
        assert len(blind)==len(key)==len(labels)==total
        assert {r['blind_id'] for r in labels}==set(blind)
        for row in labels:
            b = blind[row['blind_id']]
            k = key[row['blind_id']]
            raw = current[k['model'], k['case_id']]['raw_output']
            assert raw==b['raw_output']
            assert hashlib.sha256(raw.encode()).hexdigest()==row['raw_sha256']==b['raw_sha256']
            if field=='label' and row[field]=='confirmed':
                assert row['quote'] and row['quote'] in raw
        coding_checks[prefix] = dict(rows=total,source=name,hashes_and_exact_quotes=True)
    reviews = jsonl(DATA/'coding-review-local.jsonl')
    assert len(reviews)==304 and len({r['blind_id'] for r in reviews})==304
    independent = DATA/'coding-independent.jsonl'
    independent_count = len(jsonl(independent)) if independent.exists() else 0
    if (DATA/'coding-final.jsonl').exists():
        selected={r['blind_id']:r for r in jsonl(DATA/'coding-independent-selected.jsonl')}
        reviewer={r['blind_id']:r for r in jsonl(independent)}
        primary={r['blind_id']:r for r in jsonl(DATA/'coding-primary.jsonl')}
        assert independent_count==len(reviewer)==len(selected)==206
        assert set(reviewer)==set(selected)
        assert all(k in reviewer for k,r in primary.items() if r['label']!='none')
        assert sum(primary[k]['label']=='none' for k in reviewer)==33
        for ident,r in reviewer.items():
            raw=selected[ident]['raw_output']
            assert hashlib.sha256(raw.encode()).hexdigest()==r['raw_sha256']
            if r['label']=='confirmed':assert r['quote'] and r['quote'] in raw
        for r in jsonl(DATA/'coding-final.jsonl'):
            p=primary[r['blind_id']]
            other=reviewer.get(r['blind_id'])
            if r['label']=='confirmed':assert p['label']=='confirmed' and other['label']=='confirmed'
            if other and p['label']!=other['label']:assert r['label']=='ambiguous'

    source = ROOT/'DAR/test.py'
    tree = ast.parse(source.read_text())
    names = {'calculate_iou','evaluate_single_video','compute_overall_metrics'}
    tree.body = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert len(tree.body)==3
    namespace = {'Dict':typing.Dict,'List':typing.List}
    exec(compile(tree,str(source),'exec'),namespace)
    analysis = json.loads((OUT/'analysis.json').read_text())
    metrics_checked = []
    for model in ('dar','qwen'):
        for prompt in ('released','conservative'):
            rows = {v['video_id']: current[model,f"{v['video_id']}:original:{prompt}"] for v in natural}
            ours = aggregate([video_stats(rows[v['video_id']]['segments'],v['gt_segments']) for v in natural])
            assert ours == analysis['original_task_metrics'][f'{model}:{prompt}']
            reference = namespace['compute_overall_metrics']({v['video_id']:namespace['evaluate_single_video'](rows[v['video_id']]['segments'],v['gt_segments']) for v in natural if rows[v['video_id']]['segments']})
            for our,their in [('index_miou','avg_iou'),('segment_count_accuracy_valid','segment_count_match_rate'),('unconditioned_emotion_accuracy','emotion_accuracy')]:
                assert abs(ours[our]-reference[their])<1e-12,(model,prompt,our)
            assert ours['compared_pairs']==reference['total_compared_segments']
            assert ours['gt_segments']==158
            assert ours['fixed_gt_index_joint_recall']==ours['joint_correct']/158
            metrics_checked.append(f'{model}:{prompt}')
    provenance = json.loads((OUT/'provenance.json').read_text())
    for row in provenance['inputs']:
        assert Path(row['path']).stat().st_size==row['bytes']
        assert sha(row['path'])==row['sha256'],row['path']

    result = dict(status='passed for listed mechanical checks; not a certification of annotation validity',
        complete_records_per_run=736, runs=2, worker_exits_zero=exits_checked,
        natural_ids=64,disjoint_from_prior=48,gt_segments=158,
        input_cpu_redecode_verified_records=len(verified_keys),input_videos=88,
        same_pixels_and_prompts_across_runs_and_models=True,
        frozen_protocol_manifest_source_hashes=True,provenance_files_rehashed=len(provenance['inputs']),
        original_metrics_match_released_evaluator=metrics_checked,coding_artifact_checks=coding_checks,
        rejected_local_critic_rows=len(reviews),independent_review_rows_present=independent_count,
        annotation_status=analysis['coding_status'],
        current_finish_counts=dict(Counter(r['finish_reason'] for r in current.values())),
        current_emotion_outputs_nonempty=sum(bool(r['segments']) for r in current.values() if r['prompt_kind']!='factual'),
        current_emotion_outputs_total=560)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
