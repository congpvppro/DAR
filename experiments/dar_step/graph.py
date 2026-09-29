"""Validate FSGs, merge identities within clips, and add cross-clip references."""
from __future__ import annotations

import copy
import re

from common import bounded_text, number, require


def fsg(graph):
    require(set(graph) == {'objects', 'relations', 'observation'}, 'Invalid frame graph keys')
    require(isinstance(graph['objects'], list) and len(graph['objects']) <= 6, 'Invalid object count')
    ids = set()
    for obj in graph['objects']:
        require(set(obj) == {'id', 'label', 'kind', 'attributes'}, 'Invalid object keys')
        require(isinstance(obj['id'], str) and re.fullmatch(r'o\d{1,3}', obj['id'])
                and obj['id'] not in ids, 'Invalid/duplicate object ID')
        ids.add(obj['id'])
        bounded_text(obj['label'], 60)
        require(obj['kind'] in ('static', 'dynamic'), 'Invalid object kind')
        require(isinstance(obj['attributes'], list) and len(obj['attributes']) <= 3,
                'Invalid attributes')
        for attr in obj['attributes']:
            bounded_text(attr, 80)
    require(isinstance(graph['relations'], list) and len(graph['relations']) <= 6, 'Invalid relations')
    for edge in graph['relations']:
        require(isinstance(edge, list) and len(edge) == 3 and edge[0] in ids
                and edge[2] in ids and edge[0] != edge[2], 'Invalid relation endpoint')
        bounded_text(edge[1], 80)
    bounded_text(graph['observation'])
    return graph


def verified_graph(graph, verdict):
    require(set(verdict) == {'objects', 'attributes', 'relations', 'observation'}, 'Invalid verification keys')
    require(all(isinstance(verdict[k], list) for k in ('objects', 'attributes', 'relations')),
            'Verification fields must be lists')
    by_id = {o['id']: o for o in graph['objects']}
    require(all(isinstance(x, str) and x in by_id for x in verdict['objects']), 'Unknown verified ID')
    keep = set(verdict['objects'])
    attributes = set()
    for pair in verdict['attributes']:
        require(isinstance(pair, list) and len(pair) == 2 and pair[0] in by_id
                and type(pair[1]) is int and 0 <= pair[1] < len(by_id[pair[0]]['attributes']),
                'Unknown verified attribute')
        attributes.add(tuple(pair))
    require(all(type(i) is int and 0 <= i < len(graph['relations']) for i in verdict['relations']),
            'Unknown verified relation')
    require(type(verdict['observation']) is bool, 'Verification must be boolean')
    result = copy.deepcopy(graph)
    result['objects'] = [o for o in result['objects'] if o['id'] in keep]
    for obj in result['objects']:
        obj['attributes'] = [a for i, a in enumerate(obj['attributes']) if (obj['id'], i) in attributes]
    result['relations'] = [e for i, e in enumerate(graph['relations'])
                           if i in verdict['relations'] and e[0] in keep and e[2] in keep]
    result['observation'] = graph['observation'] if verdict['observation'] else ''
    return result


def scoped(graph, frame_id):
    result = copy.deepcopy(graph)
    mapping = {o['id']: frame_id + '_' + o['id'] for o in graph['objects']}
    for obj in result['objects']:
        obj['id'] = mapping[obj['id']]
    result['relations'] = [[mapping[a], p, mapping[b]] for a, p, b in graph['relations']]
    return result


def pair_result(result, left, right, bridge=False):
    require(set(result) == {'matches', 'motions'} and all(isinstance(result[k], list)
            for k in result), 'Invalid pairing response')
    a = {o['id']: o for o in left['objects']}
    b = {o['id']: o for o in right['objects']}
    seen_a, seen_b, pairs = set(), set(), set()
    for edge in result['matches']:
        require(isinstance(edge, list) and len(edge) == 2, 'Invalid match')
        x, y = edge
        require(x in a and y in b and x not in seen_a and y not in seen_b, 'Match must be one-to-one')
        require(a[x]['kind'] == b[y]['kind'], 'Conflicting static/dynamic identity')
        seen_a.add(x)
        seen_b.add(y)
        pairs.add((x, y))
    require(len(result['motions']) <= 6, 'Too many motion edges')
    require(not bridge or not result['motions'], 'Motion across a cut is unsupported')
    for edge in result['motions']:
        require(isinstance(edge, list) and len(edge) == 3 and (edge[0], edge[2]) in pairs,
                'Motion must use matched observations')
        require(a[edge[0]]['kind'] == b[edge[2]]['kind'] == 'dynamic', 'Static motion is invalid')
        bounded_text(edge[1], 80)
    return result


class IdentityGroups:
    """Union-find that cannot collapse two distinct objects observed together."""
    def __init__(self, members):
        self.parent = {key: key for key in members}
        self.scope = {key: {scope} for key, scope in members.items()}

    def find(self, key):
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        require(not self.scope[a] & self.scope[b], 'Identity collision: two objects in the same frame/scene')
        self.parent[b] = a
        self.scope[a] |= self.scope[b]


def assemble(sampling, parsed, local_pairs, bridges, descriptions):
    frames = sampling['frames']
    objects = {o['id']: (f, o) for f in frames for o in parsed[f['id']]['objects']}
    require(bool(objects), 'No verified objects in video')
    groups = IdentityGroups({oid: f['id'] for oid, (f, _) in objects.items()})
    motions = []
    for result in local_pairs:
        for a, b in result['matches']:
            require(objects[a][0]['scene'] == objects[b][0]['scene'], 'Local merge crosses scene')
            groups.join(a, b)
        motions.extend(result['motions'])
    entities, entity_ids, observation_entity = [], {}, {}
    for oid, (frame, obj) in objects.items():
        root = groups.find(oid)
        if root not in entity_ids:
            eid = f'e{len(entities)}'
            entity_ids[root] = eid
            entities.append(dict(id=eid, scene=frame['scene'], label=obj['label'], kind=obj['kind']))
        observation_entity[oid] = entity_ids[root]
    references = set()
    cross_groups = IdentityGroups({e['id']: e['scene'] for e in entities})
    for result in bridges:
        for a, b in result['matches']:
            x, y = observation_entity[a], observation_entity[b]
            require(objects[a][0]['scene'] != objects[b][0]['scene'], 'Reference is within one scene')
            cross_groups.join(x, y)
            references.add((x, 'same_as', y))
    result = dict(scenes=[dict(s, description=descriptions[s['id']]) for s in sampling['scenes']],
                  entities=entities, frames=[], motion_links=motions,
                  reference_links=[list(e) for e in sorted(references)],
                  event_links=[[a['id'], 'before', b['id']]
                               for a, b in zip(sampling['scenes'], sampling['scenes'][1:])])
    for frame in frames:
        graph = parsed[frame['id']]
        result['frames'].append(dict(id=frame['id'], scene=frame['scene'], time=frame['time'],
            objects=[dict(id=o['id'], entity=observation_entity[o['id']], attributes=o['attributes'])
                     for o in graph['objects']], relations=graph['relations'], observation=graph['observation']))
    return result


def validate(graph, duration):
    require(set(graph) == {'scenes', 'entities', 'frames', 'motion_links', 'reference_links', 'event_links'},
            'Invalid whole-video graph keys')
    require(all(isinstance(x, list) for x in graph.values()), 'Graph fields must be lists')
    scenes, entities, observations, frame_ids = {}, {}, {}, set()
    previous_end = 0
    for scene in graph['scenes']:
        require(set(scene) == {'id', 'start', 'end', 'description'}, 'Invalid scene')
        bounded_text(scene['id'], 30)
        require(scene['id'] not in scenes, 'Duplicate scene')
        start, end = number(scene['start']), number(scene['end'])
        require(abs(start - previous_end) <= .001 and start < end <= duration + .001, 'Scene gap/overlap/bounds')
        if scene['description']:
            bounded_text(scene['description'])
        else:
            require(scene['description'] == '', 'Invalid empty description')
        scenes[scene['id']] = scene
        previous_end = end
    require(scenes and abs(previous_end - duration) <= .25, 'Scenes do not cover video')
    for entity in graph['entities']:
        require(set(entity) == {'id', 'scene', 'label', 'kind'}, 'Invalid entity')
        bounded_text(entity['id'], 30)
        bounded_text(entity['label'], 60)
        require(entity['id'] not in entities and entity['scene'] in scenes
                and entity['kind'] in ('static', 'dynamic'), 'Invalid/duplicate entity')
        entities[entity['id']] = entity
    require(entities and 0 < len(graph['frames']) <= 64, 'Empty/oversize graph')
    previous_time, covered = -1, set()
    for frame in graph['frames']:
        require(set(frame) == {'id', 'scene', 'time', 'objects', 'relations', 'observation'}, 'Invalid frame')
        require(frame['scene'] in scenes and frame['id'] not in frame_ids, 'Invalid frame reference')
        frame_ids.add(frame['id'])
        covered.add(frame['scene'])
        time = number(frame['time'])
        scene = scenes[frame['scene']]
        require(previous_time < time and scene['start'] <= time < scene['end'], 'Invalid frame time')
        previous_time = time
        local, local_entities = set(), set()
        require(isinstance(frame['objects'], list) and len(frame['objects']) <= 6, 'Object count')
        for obj in frame['objects']:
            require(set(obj) == {'id', 'entity', 'attributes'} and obj['id'] not in observations,
                    'Invalid/duplicate observation')
            require(obj['entity'] in entities and entities[obj['entity']]['scene'] == frame['scene']
                    and obj['entity'] not in local_entities, 'Entity collision/scope')
            require(isinstance(obj['attributes'], list) and len(obj['attributes']) <= 3, 'Attributes')
            for attr in obj['attributes']:
                bounded_text(attr, 80)
            local.add(obj['id'])
            local_entities.add(obj['entity'])
            observations[obj['id']] = (frame, obj)
        require(isinstance(frame['relations'], list) and len(frame['relations']) <= 6, 'Relation count')
        for edge in frame['relations']:
            require(isinstance(edge, list) and len(edge) == 3 and edge[0] in local and edge[2] in local
                    and edge[0] != edge[2], 'Invalid spatial edge')
            bounded_text(edge[1], 80)
        if frame['observation']:
            bounded_text(frame['observation'])
        else:
            require(frame['observation'] == '', 'Invalid empty observation')
    require(covered == set(scenes), 'Scene missing keyframes')
    require({o['entity'] for _, o in observations.values()} == set(entities), 'Unused entity')
    for edge in graph['motion_links']:
        require(isinstance(edge, list) and len(edge) == 3 and edge[0] in observations
                and edge[2] in observations, 'Invalid motion references')
        a, ao = observations[edge[0]]
        b, bo = observations[edge[2]]
        require(a['time'] < b['time'] and a['scene'] == b['scene'] and ao['entity'] == bo['entity']
                and entities[ao['entity']]['kind'] == 'dynamic', 'Invalid motion identity/time')
        bounded_text(edge[1], 80)
    cross = IdentityGroups({e: entities[e]['scene'] for e in entities})
    for edge in graph['reference_links']:
        require(isinstance(edge, list) and len(edge) == 3 and edge[0] in entities and edge[2] in entities
                and edge[1] == 'same_as', 'Invalid reference edge')
        a, b = entities[edge[0]], entities[edge[2]]
        require(scenes[a['scene']]['start'] < scenes[b['scene']]['start'] and a['kind'] == b['kind'],
                'Invalid cross-clip reference')
        cross.join(a['id'], b['id'])
    expected = [[a, 'before', b] for a, b in zip(scenes, list(scenes)[1:])]
    require(graph['event_links'] == expected, 'Invalid scene chronology')
    return graph
