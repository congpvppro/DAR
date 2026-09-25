from pathlib import Path
import requests
from kaggle.api.kaggle_api_extended import KaggleApi,ApiListKernelSessionOutputRequest
api=KaggleApi();api.authenticate();r=ApiListKernelSessionOutputRequest();r.user_name='vucongaaa';r.kernel_slug='dar-grpo';r.page_size=200
wanted={'trainer_state.json','model.safetensors.index.json'}
base=Path('kaggle/dar-grpo-v2/checkpoint-download/dar_grpo/checkpoints/v0-20260923-042158/checkpoint-4100')
with api.build_kaggle_client() as client:
 for _ in range(3):
  out=client.kernels.kernels_api_client.list_kernel_session_output(r)
  for item in out.files or []:
   if item.file_name.startswith('dar_grpo/checkpoints/v0-20260923-042158/checkpoint-4100/') and item.file_name.rsplit('/',1)[-1] in wanted:
    p=base/item.file_name.rsplit('/',1)[-1]; p.write_bytes(requests.get(item.url,timeout=30).content); print(p.name,p.stat().st_size)
  r.page_token=out.next_page_token
