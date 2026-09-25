"""Export existing answers and verifiable visual review material; no model inference."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('FORCE_QWENVL_VIDEO_READER', 'decord')
import hashlib
import json
from pathlib import Path
import sys
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import decord
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'DAR/qwen-vl-utils/src'))
from qwen_vl_utils.vision_process import fetch_video
torch.set_num_threads(2)
SOURCE = ROOT / 'research/hypothesis-validation-20260908/timing-corrected'
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def sheets(frames, indices, fps, dest, title, cols=4, per_page=24):
    paths=[]
    tw, th = 300, 240
    for start in range(0, len(indices), per_page):
        subset=frames[start:start+per_page]
        canvas=Image.new('RGB',(cols*tw,40+math.ceil(len(subset)/cols)*th),'#151515')
        draw=ImageDraw.Draw(canvas)
        draw.text((8,8),title, font=FONT, fill='white')
        for j,frame in enumerate(subset):
            im=Image.fromarray(frame)
            im.thumbnail((tw-8,th-30))
            x=(j%cols)*tw;y=40+(j//cols)*th
            canvas.paste(im,(x+(tw-im.width)//2,y+26))
            idx=int(indices[start+j])
            draw.text((x+5,y+3),f'frame {idx} | {idx/fps:.3f}s',font=FONT,fill='white')
        path=dest.parent/f'{dest.stem}-{start//per_page+1:02d}.jpg'
        canvas.save(path,quality=94)
        paths.append(str(path.relative_to(OUT)))
    return paths

def main():
    rows={}
    for p in sorted(SOURCE.glob('dar-shard*.jsonl')):
        for lineno,line in enumerate(p.read_text().splitlines(),1):
            r=json.loads(line)
            if r['kind']=='natural' and r['condition']=='original' and r['prompt_kind']=='released':
                assert r['video_id'] not in rows
                rows[r['video_id']]=(r,p,lineno)
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    ids=[r['video_id'] for r in manifest['videos'] if r['kind']=='natural']
    assert len(ids)==len(rows)==64
    exports=[]
    for number,vid in enumerate(ids):
        r,p,lineno=rows[vid]
        folder=OUT/'cases'/vid
        folder.mkdir(parents=True,exist_ok=True)
        vp=ROOT/'data/vce_dataset/videos'/f'{vid}.mp4'
        tensor,meta=fetch_video(dict(video=str(vp),nframes=16,min_pixels=100352,max_pixels=100352),return_video_metadata=True)
        digest=hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
        assert digest==r['input_sha256'],vid
        assert meta['frames_indices']==r['video_metadata']['frames_indices']
        arr=tensor.byte().permute(0,2,3,1).numpy()
        model_sheets=sheets(arr,meta['frames_indices'],meta['fps'],folder/'model.jpg',f'{vid} | exact 16 model frames',per_page=16)
        for j,frame in enumerate(arr): Image.fromarray(frame).save(folder/f'model-{j:02d}.png')
        vr=decord.VideoReader(str(vp),num_threads=2)
        fps=float(vr.get_avg_fps())
        dense_indices=sorted(set([min(len(vr)-1,round(t*fps)) for t in np.arange(0,len(vr)/fps,0.5)]+[len(vr)-1]))
        dense_frames=vr.get_batch(dense_indices).asnumpy()
        dense_sheets=sheets(dense_frames,dense_indices,fps,folder/'source.jpg',f'{vid} | source video at 2 fps + final frame')
        (folder/'raw.txt').write_text(r['raw_output']+'\n')
        item=dict(video_id=vid,group=number//16,video_path=str(vp),video_duration=r['video_duration'],source_fps=fps,source_frame_count=len(vr),source_sha256=sha(vp),raw_output=r['raw_output'],raw_sha256=hashlib.sha256(r['raw_output'].encode()).hexdigest(),segments=r['uncorrected_segments'],source_record=str(p),source_line=lineno,model_input_sha256=digest,model_frame_indices=meta['frames_indices'],model_sheets=model_sheets,source_sheets=dense_sheets,source_review_indices=dense_indices)
        (folder/'record.json').write_text(json.dumps(item,ensure_ascii=False,indent=2)+'\n')
        exports.append(item)
        print(f'{number+1}/64 {vid}: matched tensor; {len(dense_indices)} source frames',flush=True)
    (OUT/'records.json').write_text(json.dumps(exports,ensure_ascii=False,indent=2)+'\n')
    for group in range(4):
        group_rows=[r for r in exports if r['group']==group]
        (OUT/f'group-{group}.json').write_text(json.dumps(group_rows,ensure_ascii=False,indent=2)+'\n')
    (OUT/'preparation-verification.json').write_text(json.dumps(dict(records=64,matched_input_hashes=64,source_files={str(p):sha(p) for p in sorted(SOURCE.glob('dar-shard*.jsonl'))},protocol_sha256=sha(OUT/'protocol.md'),source_frame_views=sum(len(r['source_review_indices']) for r in exports)),indent=2)+'\n')

if __name__=='__main__': main()
