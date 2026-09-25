import json,pathlib,hashlib,collections,html.parser,urllib.parse
P=pathlib.Path(__file__).resolve().parent
load=lambda n:json.loads((P/n).read_text())
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
A=load('annotations.json'); R={r['video_id']:r for r in load('records.json')}; S=load('summary.json'); V=load('preparation-verification.json')
assert len(A)==len(R)==len({a['video_id'] for a in A})==64
assert {a['video_id'] for a in A}==set(R)
assert dict(collections.Counter(a['status'] for a in A))==S['status_counts']
assert V['matched_input_hashes']==64
assert sha(P/'protocol.md')==V['protocol_sha256']
for f,h in V['source_files'].items():assert sha(f)==h
claims=0;imgs=set()
for a in A:
 r=R[a['video_id']]
 assert sha(r['video_path'])==r['source_sha256']
 assert hashlib.sha256(r['raw_output'].encode()).hexdigest()==r['raw_sha256']
 assert (P/'cases'/a['video_id']/'raw.txt').read_text().strip()==r['raw_output'].strip()
 source=json.loads(pathlib.Path(r['source_record']).read_text().splitlines()[r['source_line']-1])
 assert source['video_id']==r['video_id'] and source['raw_output']==r['raw_output']
 assert (source['kind'],source['condition'],source['prompt_kind'])==('natural','original','released')
 assert source['input_sha256']==r['model_input_sha256']
 assert len(r['model_frame_indices'])==16
 assert set(r['model_sheets']+r['source_sheets'])<=set(a['inspected_images'])
 for c in a['claims']:
  assert c['quote'] in r['raw_output']
  assert any(c['quote'] in t['reason'] for t in r['segments'])
  assert set(c['evidence_images'])<=set(a['inspected_images']);claims+=1
 for im in a['inspected_images']:assert (P/im).is_file();imgs.add(im)
 assert (a['status']=='confirmed')==any(c['judgment']=='contradicted' for c in a['claims'])
assert len(set(S['example_ids']))==10
assert all(next(a for a in A if a['video_id']==i)['status']=='confirmed' for i in S['example_ids'])
assert S['event_error_videos']==sum(any(c['judgment']=='contradicted' and c['claim_type']=='invented_event' for c in a['claims']) for a in A)
class Links(html.parser.HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.ids=set()
 def handle_starttag(self,tag,attrs):
  d=dict(attrs)
  if 'id' in d:self.ids.add(d['id'])
  for k in ['href','src']:
   if k in d:self.links.append(d[k])
h=Links();h.feed((P/'review.html').read_text())
for link in h.links:
 u=urllib.parse.urlsplit(link)
 if u.path:assert (P/urllib.parse.unquote(u.path)).is_file(),link
 if not u.path and u.fragment:assert u.fragment in h.ids,link
out={'passed':True,'records_checked':64,'source_video_hashes_checked':64,'raw_output_hashes_checked':64,'exact_claim_quotes_checked':claims,'inspected_image_files_checked':len(imgs),'html_links_checked':len(h.links),'ten_examples_checked':True,'status_counts':S['status_counts'],'event_error_videos':S['event_error_videos'],'note':'Integrity checks only; these assertions do not independently validate visual judgments.'}
(P/'final-verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))
