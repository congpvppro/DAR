"""Pure input transformations and a frozen, explicit inference matrix."""
import numpy as np

SEED = 20260908
DURATIONS = [4,6,8,10,12,15,18,20,24,30,40,60]
CONSERVATIVE = ('When no visible event occurs, do not invent one to justify a transition. '
 'A single segment covering the whole video is acceptable. '
 'If you infer habituation or another viewer response without any visible change, explicitly say so. '
 'Distinguish directly observed changes from inferred or uncertain events; '
 'do not present inferred events as observations.')
FACTUAL = ('Inspect only the supplied sampled video frames. Between these frames, is there any '
 'visible change of objects, positions, text, lighting, or camera view? '
 'Do not infer motion between sampled frames. Output ONLY a JSON object with keys '
 '"visible_change" (boolean) and "evidence" (one short sentence). Do not predict emotions.')


def intervene(frames, condition):
    if condition == 'original':
        return frames.copy()
    if condition == 'static_middle':
        return np.repeat(frames[len(frames)//2:len(frames)//2+1],len(frames),axis=0)
    if condition in ('gray','gray_to_white'):
        out=np.full(frames.shape,127.5,dtype=np.float32)
        if condition == 'gray_to_white':
            out[len(out)//2:]=255.0
        return out
    raise ValueError(condition)


def build_cases(videos):
    cases=[]
    for row in videos:
        if row['kind']=='natural':
            combinations=[(c,p) for c in ('original','static_middle') for p in ('released','conservative')]
            combinations.append(('static_middle','factual'))
        elif row['kind']=='gray':
            combinations=[('gray',p) for p in ('released','conservative','factual')]
        elif row['kind']=='step':
            combinations=[('gray_to_white','factual')]
        else:
            raise ValueError(row['kind'])
        for condition,prompt_kind in combinations:
            cases.append(dict(video_id=row['video_id'],condition=condition,prompt_kind=prompt_kind,
                              case_id=f"{row['video_id']}:{condition}:{prompt_kind}"))
    return cases
