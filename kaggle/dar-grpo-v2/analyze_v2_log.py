import json
from kaggle.api.kaggle_api_extended import KaggleApi,ApiListKernelSessionOutputRequest
api=KaggleApi();api.authenticate();r=ApiListKernelSessionOutputRequest();r.user_name='vuhuycong';r.kernel_slug='dar-grpo';r.page_size=20
with api.build_kaggle_client() as c: out=c.kernels.kernels_api_client.list_kernel_session_output(r)
log=json.loads(out.log)
print('events',len(log))
for e in log:
 d=e.get('data','')
 if any(x in d.lower() for x in ['error:', 'no such file', 'not found', 'could not', 'failed', 'looking in links', 'wheelhouse','pip install']):
  print(e.get('stream_name'),round(e.get('time',0),2),repr(d[:500]))
