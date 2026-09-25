from collections import defaultdict
from kaggle.api.kaggle_api_extended import KaggleApi
from packaging.utils import parse_wheel_filename
api=KaggleApi();api.authenticate()
token=None;page=0;wheels=[]
while True:
 r=api.dataset_list_files('vucongaaa/dar-r1',page_token=token,page_size=200)
 for f in r.files or []:
  if 'offline-wheelhouse-kaggle-py312-cu128/' in f.name and f.name.endswith('.whl'):
   wheels.append(f.name.rsplit('/',1)[-1])
 page+=1
 token=r.next_page_token
 if not token:break
print('pages',page,'wheels',len(wheels))
by=defaultdict(list)
for w in wheels:
 try:
  name,version,*_=parse_wheel_filename(w)
  by[str(name)].append((str(version),w))
 except Exception as e:print('invalid',w,str(e)[:100])
for name,versions in sorted(by.items()):
 if len(set(v for v,_ in versions))>1:
  print(name,versions)
