"""Reproducible source extraction and figures for the R1 report extension."""
from pathlib import Path
import json
import hashlib
import math
from bs4 import BeautifulSoup

HERE = Path(__file__).resolve().parent
IDS = ['2506.07218v3', '2505.19094v2', '2504.07954v1']

def extract():
    records = []
    for paper in IDS:
        path = HERE / 'sources' / f'{paper}.html'
        soup = BeautifulSoup(path.read_text(encoding='utf-8'), 'html.parser')
        article = soup.find('article') or soup
        (path.with_suffix('.txt')).write_text(article.get_text(' ', strip=True), encoding='utf-8')
        blocks = []
        for element in article.find_all(['section', 'figure']):
            if element.name == 'section' and element.find('section'):
                continue
            blocks.append(f"\nID={element.get('id', '')}\n" + element.get_text(' ', strip=True))
        (HERE / 'sources' / f'{paper}.sections.txt').write_text('\n\n'.join(blocks), encoding='utf-8')
        records.append({'id': paper, 'url': f'https://arxiv.org/html/{paper}',
                        'file': str(path.relative_to(HERE)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    root = HERE.parents[1]
    code_paths = ['ms-swift/examples/train/grpo/plugin/dar_plugin.py',
                  'ms-swift/examples/train/grpo/dar/prepare_dar_grpo_data.py',
                  'ms-swift/examples/train/grpo/dar/train_qwen2.5vl_grpo_dar.sh',
                  'data_construction/04_qwen3vl_differential_description.py',
                  'data_construction/05_qwen3vl_stream_affect_reason.py', 'test.py']
    (HERE / 'r1-provenance.json').write_text(json.dumps({'access_date':'2026-09-17',
        'scope':'Targeted lookup of two homonymous Perception-R1 papers and SATORI-R1, not systematic review',
        'api':'https://export.arxiv.org/api/query?id_list=2506.07218,2504.07954,2505.19094&max_results=3',
        'full_text':records,
        'DAR_code': [{'path':p,'sha256':hashlib.sha256((root/p).read_bytes()).hexdigest()} for p in code_paths]}, indent=2), encoding='utf-8')

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10, 'text.color':'black',
        'axes.labelcolor':'black', 'xtick.color':'black', 'ytick.color':'black',
        'axes.spines.top':False, 'axes.spines.right':False, 'svg.fonttype':'none'})
    full = [74.2,54.3,28.6,72.0,60.8,42.4,64.5,27.5]
    minus = [73.6,53.0,27.6,70.4,57.2,40.1,63.5,27.9]
    labels = ['MathVista','MathVerse','MathVision','WeMath','MMMU','MMMU-Pro','MMStar','EMMA']
    contrasts = [round(a-b,1) for a,b in zip(full,minus)]
    satori = {'free_form':[64.6,50.4], 'grounding_no_think':[76.5,55.9], 'full':[76.9,56.1]}
    def save(fig,name):
        for ext in ['png','svg']:
            fig.savefig(HERE/'figures'/f'{name}.{ext}', dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5.6),gridspec_kw={'width_ratios':[1,1.35]})
    ax=axes[0]
    ax.barh(labels,contrasts,color=['#42809c' if x>=0 else '#c98268' for x in contrasts])
    ax.invert_yaxis(); ax.axvline(0,color='black',lw=.8); ax.set_xlim(-1.3,4.3)
    for i,v in enumerate(contrasts):
        ax.text(v+(.08 if v>=0 else -.08),i,f'{v:+.1f}',va='center',ha='left' if v>=0 else 'right')
    ax.set_title('A. PR-X: contribution of visual reward\nFull minus no-visual-reward (7B)',fontsize=11)
    ax.set_xlabel('Accuracy difference (percentage points)')
    ax=axes[1]
    names=['Grounding package\nvs free-form RL','Additional thinking\nvs grounding without thinking']
    a=[round(satori['grounding_no_think'][0]-satori['free_form'][0],1),round(satori['full'][0]-satori['grounding_no_think'][0],1)]
    b=[round(satori['grounding_no_think'][1]-satori['free_form'][1],1),round(satori['full'][1]-satori['grounding_no_think'][1],1)]
    y=np.arange(2)
    for offset,vals,color,label in [(-.18,a,'#42809c','MMBench'),(.18,b,'#b7c9d2','MMStar')]:
        ax.barh(y+offset,vals,height=.32,color=color,label=label)
        for yi,v in zip(y+offset,vals): ax.text(v+.12,yi,f'+{v:.1f}',va='center')
    ax.set_yticks(y,names); ax.invert_yaxis(); ax.set_xlim(0,13.6)
    ax.set_title('B. SATORI: separate grounding from thinking\nWithin Table 2 only (3B)',fontsize=11)
    ax.set_xlabel('Accuracy difference (percentage points)'); ax.legend(loc='lower right')
    fig.suptitle('Ablation evidence: benefits depend on the component and benchmark',fontsize=14,y=1.02)
    fig.tight_layout(w_pad=2.5)
    fig.text(.5,-.035,'Sources: PR-X v3 Table 2; SATORI v2 Table 2. No reported seed CI. These are not DAR results.',ha='center',fontsize=9)
    save(fig,'r1-ablation-contrasts')

    fig,ax=plt.subplots(figsize=(12,6.5)); ax.set_xlim(0,12); ax.set_ylim(0,6.5); ax.axis('off')
    def box(x,y,w,h,title,body,fill):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.10',facecolor=fill,edgecolor='#5a6870',lw=1))
        ax.text(x+w/2,y+h-.22,title,ha='center',va='top',weight='bold',fontsize=11)
        ax.text(x+w/2,y+.23,body,ha='center',va='bottom',fontsize=9.5,linespacing=1.45)
    def arrow(a,b,label=None):
        ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'black','lw':1.3})
        if label: ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,ha='center',fontsize=9)
    box(.15,4.05,2.25,1.35,'Observed video','Fixed frames + times\nGlobal context retained','#e9eff2')
    box(3.0,4.05,2.5,1.35,'Student evidence','Event facts + time IDs\nBoxes optional','#e3eef1')
    box(6.1,4.05,2.35,1.35,'Viewer appraisal','Expectation → update\nNot actor emotion','#e3eef1')
    box(9.05,4.05,2.65,1.35,'DAR prediction','Segments / emotion\nReason + raw output','#e3eef1')
    for a,b in [((2.5,4.7),(2.88,4.7)),((5.60,4.7),(5.98,4.7)),((8.55,4.7),(8.93,4.7))]: arrow(a,b)
    box(.25,1.55,3.1,1.45,'Training references only','Label-blind verified facts\nFrame provenance + source split\nNever inserted in student prompt','#f3eee4')
    box(4.25,1.55,3.2,1.45,'Reward comparisons','Old / removed / lexical / factual\nCoverage + support + time\nOne-to-one matching separately','#f3eee4')
    box(8.35,1.55,3.25,1.45,'Independent evaluation','Blind video-based review\nJoint temporal-emotion F1\nCoverage, cost and invalid rate','#e9f0e7')
    arrow((3.46,2.3),(4.13,2.3))
    arrow((4.25,3.93),(5.55,3.12)); arrow((10.4,3.93),(10.0,3.12))
    ax.text(6,.70,'Matched controls: same facts, checkpoint, frames and training exposure; measure any remaining cost differences.',ha='center',fontsize=9)
    ax.text(6,.30,'Proposed DAR study. No model training or new video inference was performed.',ha='center',fontsize=9)
    ax.set_title('Transfer mechanisms and the controls needed to test them',fontsize=15,pad=12)
    save(fig,'r1-dar-transfer')

    rewards=np.array([0.,1.,2.,3.]); scaled=.1*rewards
    adv=lambda r:(r-r.mean())/r.std()
    assert np.allclose(adv(rewards),adv(scaled))
    fig,axes=plt.subplots(1,2,figsize=(10.5,4))
    for ax,series,title in [(axes[0],[rewards,scaled],'A. Raw rewards differ'),(axes[1],[adv(rewards),adv(scaled)],'B. Standardized advantages are identical')]:
        for vals,style,label in zip(series,['o-','s--'],['R','0.1 × R']): ax.plot(range(1,5),vals,style,label=label)
        ax.set_xticks(range(1,5)); ax.set_xlabel('Rollout in a synthetic group'); ax.set_title(title,fontsize=11); ax.legend()
    axes[0].set_ylabel('Reward'); axes[1].set_ylabel('Advantage (epsilon omitted)')
    fig.tight_layout(); fig.text(.5,-.055,'Analytical illustration only: Var(R)=1.25; Var(0.1R)=0.0125. Not measured SATORI or DAR gradients.',ha='center',fontsize=9)
    save(fig,'r1-grpo-scale')
    def mcnemar(b,c):
        n=b+c
        return min(1.,2*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n)
    results={'scope':'Manual transcription of primary tables plus deterministic mathematical examples; no model inference.',
        'PR-X':{'source':'2506.07218v3 Table 2','benchmarks':labels,'full':full,'without_visual':minus,'difference_pp':contrasts},
        'SATORI':{'source':'2505.19094v2 Table 2','benchmarks':['MMBench','MMStar'],**satori},
        'mcnemar_exact':{f'{b},{c}':mcnemar(b,c) for b,c in [(1,5),(2,4),(2,10)]},
        'scale_probe':{'reward':rewards.tolist(),'scaled':scaled.tolist(),'variance':[float(rewards.var()),float(scaled.var())],
                       'advantage':[adv(rewards).tolist(),adv(scaled).tolist()]},
        'union_probe':{'A':[[0,10]],'B':[[0,5],[5,10]],'intersection_length':10,'union_length':10,'union_IoU':10/10,
                       'limitation':'Union coverage does not identify segmentation structure, count or event order.'},
        'variance_counterexample':{'var_R1':1,'var_R2':100,'covariance':0,'weights':[.5,.5], 'combined_variance':.25*1+.25*100}}
    (HERE/'r1-figure-data.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')

def update_report():
    path=HERE/'report.vi.md'
    original=path.read_text(encoding='utf-8').split('\n## 12. Định danh paper')[0].rstrip()
    extension=(HERE/'r1-extension.vi.md').read_text(encoding='utf-8')
    path.write_text(original+'\n\n'+extension,encoding='utf-8')
    from build_report import render_html
    render_html()

if __name__ == '__main__':
    extract()
    figures()
    update_report()
    print('Updated report.vi.md/html, three new PNG/SVG figures and R1 provenance/data.')
