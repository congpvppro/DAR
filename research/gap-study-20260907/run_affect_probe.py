"""Fresh 4-video compatibility and frame-order representation diagnostic."""
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='2'
sys.path.insert(0,str(ROOT/'DAR'))
from tools.run_affectgpt_dar import AffectGPTBackend, load_test_data, select_checkpoint, build_frame_only_prompt, make_prediction_record
import torch

torch.set_num_threads(2)
records=load_test_data(str(OUT/'affectgpt-fresh-input.jsonl'))
config=ROOT/'AffectGPT-src/AffectGPT/train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz.yaml'
checkpoint=select_checkpoint(ROOT/'model/AffectGPT/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz_20250408110',30)
backend=AffectGPTBackend(ROOT/'AffectGPT-src/AffectGPT',config,checkpoint,str(ROOT/'data/vce_dataset'),'/vce_dataset/',4096,8192,20260907)
metadata=dict(seed=20260907,checkpoint=str(checkpoint),config=str(config),nframes=8,device='GPU0',
              max_tokens=4096,temperature=0,do_sample=False,scope='four fixed manifest videos; native AffectGPT frame-only interface; not directly comparable to 16-frame paired pilot',
              started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
(OUT/'affectgpt-fresh-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
with (OUT/'affectgpt-fresh.jsonl').open('x') as stream, torch.inference_mode():
    for row in records:
        path=ROOT/'data/vce_dataset/videos'/f"{row['video_id']}.mp4"
        sample=backend.dataset.read_frame_face_audio_text(video_path=str(path),face_npy=None,audio_path=None,image_path=None)
        reverse=dict(sample)
        for key in ['frame','raw_frame']:
            reverse[key]=sample[key].flip(1)
        _, original_features=backend.chat.postprocess_frame(sample)
        _, reverse_features=backend.chat.postprocess_frame(reverse)
        a,b=original_features.float(),reverse_features.float()
        delta=float((a-b).abs().max())
        relative=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a))
        cosine=float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0))
        prompt=build_frame_only_prompt(backend.dataset,row['video_duration'])
        started=time.monotonic()
        raw=backend.chat.answer_sample(prompt=prompt,img_list=dict(audio=None,frame=original_features,face=None,image=None,multi=None),
             num_beams=1,temperature=0.1,do_sample=False,top_p=.9,max_new_tokens=4096,max_length=8192)
        result=make_prediction_record(row['video_id'],str(path),row['video_duration'],raw)
        result.update(reversed_feature_max_abs_delta=delta,reversed_feature_relative_l2=relative,reversed_feature_cosine=cosine,
                      feature_shape=list(a.shape),frames_shape=list(sample['frame'].shape),elapsed_seconds=time.monotonic()-started)
        stream.write(json.dumps(result,ensure_ascii=False)+'\n')
        stream.flush()
        print(json.dumps(dict(video_id=row['video_id'],valid=bool(result['segments']),relative_feature_delta=relative,cosine=cosine)),flush=True)
