"""Launch four bounded, disjoint workers; wait and record every exit status."""
import json
import os
from pathlib import Path
import subprocess
import sys

OUT = Path(__file__).resolve().parent
workers = []
for gpu, (model, shard) in enumerate([('dar',0),('dar',1),('qwen',0),('qwen',1)]):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', HF_HUB_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', VLLM_WORKER_MULTIPROC_METHOD='spawn',
               VLLM_CACHE_ROOT=str(OUT/'cache'), XDG_CACHE_HOME=str(OUT/'cache'))
    log = (OUT/f'{model}-shard{shard}.log').open('a')
    command = [sys.executable,str(OUT/'run_probes.py'),'--model',model,'--shard',str(shard)]
    process = subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    workers.append((model,shard,process,log,command))
    print(f'Started {model} shard {shard} on GPU {gpu}: PID {process.pid}',flush=True)
exits = []
for model,shard,process,log,command in workers:
    code = process.wait()
    log.close()
    exits.append(dict(model=model,shard=shard,exit_code=code,command=command))
    print(f'Finished {model} shard {shard}: exit {code}',flush=True)
(OUT/'worker-exits.json').write_text(json.dumps(exits,indent=2)+'\n')
sys.exit(int(any(x['exit_code'] for x in exits)))
