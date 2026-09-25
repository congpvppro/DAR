from prepare_review import sheets,OUT
import json, numpy as np, decord
R={r['video_id']:r for r in json.loads((OUT/'records.json').read_text())}
specs=[('05367',8.3,10.8,10),('04466',5.8,8.2,10),('02872',5.8,7,20),('00872',1.6,3,20),('51665',13,15.8,10),('57947',8,9.4,20),('26027',4.3,8.8,5),('05128',7,11.8,5),('18290',1,5,5),('10509',16,21.6,4)]
out={}
for vid,a,b,f in specs:
 r=R[vid];vr=decord.VideoReader(r['video_path'],num_threads=2);fps=vr.get_avg_fps();idx=sorted(set(min(len(vr)-1,round(t*fps)) for t in np.arange(a,b+.001,1/f)))
 out[vid]=sheets(vr.get_batch(idx).asnumpy(),idx,fps,OUT/'cases'/vid/'verification.jpg',vid+' | dense source verification')
(OUT/'extra-verification.json').write_text(json.dumps(out,indent=2))
print(out)
