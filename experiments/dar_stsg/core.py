"""Training-only STSG auxiliary supervision. No torch dependency in data tools."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARMS = ('baseline', 'caption', 'stsg', 'stsg_no_links')


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def official_functions():
    """Load pure prompt/scoring functions without importing the vLLM entry point."""
    names = {'EMOTION_CANDIDATES', 'build_eval_prompt', 'calculate_iou',
             'evaluate_single_video', 'compute_overall_metrics',
             'merge_adjacent_same_emotion_segments', 'validate_and_fix_segments'}
    tree = ast.parse((ROOT / 'test.py').read_text(encoding='utf-8'))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            selected.append(node)
    from typing import List, Dict
    scope = {'List': List, 'Dict': Dict}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(ROOT / 'test.py'), 'exec'), scope)
    return scope


def number(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError('Expected a finite numeric value')
    return float(x)


def parse_json(text):
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError('Expected a JSON object')
    return obj


def validate_graph(graph, duration):
    """Structural validation only; it does not certify visual truth."""
    if set(graph) != {'entities', 'events', 'temporal_links'}:
        raise ValueError('Invalid STSG top-level keys')
    entities, events, links = (graph[k] for k in ('entities', 'events', 'temporal_links'))
    if not all(isinstance(x, list) for x in (entities, events, links)):
        raise ValueError('Graph fields must be lists')
    if not 1 <= len(entities) <= 8 or not 1 <= len(events) <= 8 or len(links) > 16:
        raise ValueError('Graph size outside pilot limits')
    entity_ids = set()
    for entity in entities:
        if set(entity) != {'id', 'label'} or not all(isinstance(v, str) and v.strip() for v in entity.values()):
            raise ValueError('Invalid entity')
        if entity['id'] in entity_ids:
            raise ValueError('Duplicate entity ID')
        entity_ids.add(entity['id'])
    event_ids = {}
    for event in events:
        if set(event) != {'id', 'start_time', 'end_time', 'observation', 'relations'}:
            raise ValueError('Invalid event keys')
        if not isinstance(event['id'], str) or not event['id'] or event['id'] in event_ids:
            raise ValueError('Invalid/duplicate event ID')
        start, end = number(event['start_time']), number(event['end_time'])
        if not 0 <= start < end <= duration + .051:
            raise ValueError('Event outside video')
        if not isinstance(event['observation'], str) or not event['observation'].strip():
            raise ValueError('Missing visual observation')
        if not isinstance(event['relations'], list) or len(event['relations']) > 8:
            raise ValueError('Invalid relations')
        for edge in event['relations']:
            if not isinstance(edge, list) or len(edge) != 3 or not all(isinstance(v, str) for v in edge):
                raise ValueError('Expected subject-predicate-object triple')
            if edge[0] not in entity_ids or edge[2] not in entity_ids or not edge[1].strip():
                raise ValueError('Dangling relation')
        event_ids[event['id']] = event
    for edge in links:
        if not isinstance(edge, list) or len(edge) != 3:
            raise ValueError('Invalid temporal edge')
        a, kind, b = edge
        if a not in event_ids or b not in event_ids or a == b or kind != 'before':
            raise ValueError('Invalid temporal references')
        if event_ids[a]['end_time'] > event_ids[b]['start_time'] + .051:
            raise ValueError('Contradictory before edge')
    return graph


def evidence_prompt(duration, kind):
    common = (f'Observe this SILENT video of duration {duration:.1f}s. '
              'Describe only visible entities, interactions, appearance, camera/scene changes and events. '
              'Do not use audio, viewer emotion labels, inferred intentions or imagined outcomes. '
              'Omit uncertain details. An event boundary is not necessarily an emotion boundary. ')
    if kind == 'caption':
        return common + 'Return ONLY JSON {"caption":"a concise chronological visual description, at most 250 words"}.'
    return common + '''Return ONLY one compact JSON object, with no markdown or explanation.
It must have exactly three top-level keys: entities, events, temporal_links.
entities is a nonempty array of at most 8 objects, each with exactly id and label.
Use a distinct short string ID for each visible entity and a specific visual label.
events is a nonempty chronological array of at most 8 objects, each with exactly
id, start_time, end_time, observation, relations. Use distinct event IDs.
Times are numeric seconds within the video, with start_time strictly before end_time.
The observation must describe a concrete visible state or change.
relations is an array of at most 8 triples of strings: subject entity ID,
visible relation, object entity ID. Both IDs MUST occur in entities.
temporal_links is an array of at most 16 triples of strings: earlier event ID,
the word before, later event ID. Both IDs MUST occur in events and the earlier
event must end no later than the later event starts.
Use empty arrays for relations or temporal_links when no justified edge exists.
Never invent an ID merely to fill a relation or temporal link. Never copy schema
descriptions into labels or observations. Do not invent bounding boxes.'''


def dar_prompt(duration):
    return official_functions()['build_eval_prompt'](duration)


def annotation(row, video_root=None):
    video = row.get('video') or row.get('video_path')
    if not isinstance(video, str):
        raise ValueError('Missing video')
    basename = video.replace('\\', '/').split('/')[-1]
    path = str(Path(video_root) / basename) if video_root else video
    duration = number(row.get('video_duration'))  # Never infer duration from GT boundaries.
    if duration <= 0:
        raise ValueError('Invalid duration')
    answers = [x['value'] for x in row.get('conversations', []) if x.get('from') == 'gpt']
    target = parse_json(answers[-1]) if answers else {'segments': row.get('gt_segments')}
    if not isinstance(target.get('segments'), list) or not target['segments']:
        raise ValueError('Missing DAR target')
    return dict(video_id=Path(basename).stem, video_path=path, video_duration=duration, target=target)


def sft_rows(row, evidence, arm):
    def sample(prompt, target, suffix):
        return {'id': row['video_id'] + ':' + suffix,
                'messages': [{'role': 'user', 'content': '<video>\n' + prompt},
                             {'role': 'assistant', 'content': json.dumps(target, ensure_ascii=False)}],
                'videos': [row['video_path']]}
    answer = sample(dar_prompt(row['video_duration']), row['target'], 'dar')
    if arm == 'baseline':
        auxiliary = copy.deepcopy(answer)
        auxiliary['id'] = row['video_id'] + ':dar-repeat'
    elif arm == 'caption':
        auxiliary = sample(evidence_prompt(row['video_duration'], 'caption'), evidence['caption'], 'caption')
    else:
        graph = copy.deepcopy(evidence['stsg'])
        prompt = evidence_prompt(row['video_duration'], 'stsg')
        if arm == 'stsg_no_links':
            graph['temporal_links'] = []
            prompt += '\nFor this task leave temporal_links empty; retain event timestamps.'
        auxiliary = sample(prompt, graph, arm)
    return [answer, auxiliary]
