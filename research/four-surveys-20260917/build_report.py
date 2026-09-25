"""Reproduce report probes and figures. CPU only; no inference or training."""
import ast
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from markdown_it import MarkdownIt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = HERE / 'figures'
FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'svg.fonttype': 'none', 'figure.facecolor': 'white'})


def load_functions(relative, classes=False):
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if (getattr(node, 'module', '') or '').startswith(('swift', 'dar_pipeline_common')):
                continue
            nodes.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.Assign)):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Subscript) for t in node.targets):
                continue
            nodes.append(node)
        elif classes and isinstance(node, ast.ClassDef):
            nodes.append(node)
    env = {'ORM': object}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    return env


def probes():
    env = load_functions('ms-swift/examples/train/grpo/plugin/dar_plugin.py', True)
    committee = load_functions('data_construction/08_dual_consistency_committee.py')
    reasons = {
        'supported_12_words': 'The dog stays outside the closed gate while the viewer awaits entry.',
        'contradicted_12_words': 'The dog walks inside the open gate while the viewer watches entry.',
        'banana_120_words': ' '.join(['banana'] * 120),
    }
    names = ['DARStructuralReward', 'DARSegmentCountReward', 'DARTemporalSegmentationReward',
             'DAREmotionAccuracyReward', 'DARReasoningQualityReward']
    weights = [.10, .25, .25, .25, .15]
    def completion(reason):
        return json.dumps({'segments': [{'start_time': 0., 'end_time': 5., 'emotion': 'Interest', 'reason': reason}]})
    solution = completion(reasons['supported_12_words'])
    scores = {}
    for key, reason in reasons.items():
        vector = [env[n]()([completion(reason)], solution=[solution])[0] for n in names]
        scores[key] = {'word_count': env['_word_count'](reason), 'vector': vector,
                       'weighted_total': sum(w*r for w, r in zip(weights, vector))}
    assert scores['supported_12_words'] == scores['contradicted_12_words']
    assert scores['banana_120_words']['weighted_total'] == 1.
    # Execute the real committee main() with injected in-memory I/O to test empty record.
    captured = {}
    from types import SimpleNamespace
    args = SimpleNamespace(annotations_jsonl='annotations', qwen_judge_jsonl='qwen',
                           internvl_judge_jsonl='intern', output_jsonl='output',
                           rewrite_manifest_jsonl='rewrite', pass_threshold=3.5)
    committee['parse_args'] = lambda: args
    committee['load_by_video'] = lambda p: {'synthetic': {'segments': []}} if p == 'annotations' else {}
    committee['write_jsonl'] = lambda p, rows: captured.update({p: rows})
    committee['main']()
    empty_pass = captured['output'][0]['committee_passed']
    assert empty_pass is True
    result = {'scope': 'Synthetic inputs, unchanged function bodies; no real-video behavior measured.',
              'reward_order': names, 'weights': weights, 'scores': scores,
              'one_judge': committee['combine_segment_feedback'](0, {0: {'average_score': 5}}, {}, 3.5),
              'disagreement': committee['combine_segment_feedback'](0, {0: {'average_score': 2}}, {0: {'average_score': 5}}, 3.5),
              'empty_record_passed': empty_pass}
    (HERE / 'probe-results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def save(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_reward(result):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={'width_ratios': [1.15, 1]})
    xs = list(range(1, 281))
    axes[0].plot(xs, [math.exp(-abs(x-120)/40) for x in xs], color='#176b87', lw=2.5)
    axes[0].axvline(120, color='#555555', ls=':', lw=1)
    axes[0].set(xlabel='Words in rationale', ylabel='Reason reward', ylim=(0, 1.08),
                title='A. Exact length objective (no adjacent duplicate)')
    values = [s['vector'][-1] for s in result['scores'].values()]
    bars = axes[1].bar(['Supported\n12 words', 'Contradicted\n12 words', 'Unrelated\n120 words'], values,
                       color=['#176b87', '#bb5566', '#d59b32'], width=.6)
    axes[1].bar_label(bars, fmt='%.3f', padding=4)
    axes[1].set(ylim=(0, 1.15), ylabel='Reason reward', title='B. Executed synthetic probes')
    fig.suptitle('DAR reward diagnostic — objective behavior, not model behavior', fontsize=14, weight='bold')
    fig.text(.5, -.02, 'Source: dar_plugin.py:354–377. Scene truth is stipulated; no video inference. KL is not included.', ha='center', fontsize=9)
    fig.tight_layout()
    save(fig, 'reward-diagnostic')


def plot_primary():
    rows = [
        ('Pixel Reasoner', 'MVBench: full − base', 67.8, 63.8, '2505.15966v3', 'Table 1'),
        ('Pixel Reasoner', 'MVBench: full − no curiosity', 67.8, 66.4, '2505.15966v3', 'Table 1'),
        ('Pixel Reasoner', 'MVBench: full − no correction data', 67.8, 63.6, '2505.15966v3', 'Table 1'),
        ('VLM-R³', 'ScienceQA: full − no interleaved CoT', 87.9, 75.4, '2505.16192v2', 'Table 2'),
        ('VLM-R³', 'MathVista: full − no interleaved CoT', 70.4, 67.1, '2505.16192v2', 'Table 2'),
        ('DeepEyes', 'HR-8K: iMCoT − text CoT', 72.6, 60.8, '2505.14362v3', 'Table 9'),
        ('DeepEyes', 'HR-4K: iMCoT − text CoT', 75.1, 75.4, '2505.14362v3', 'Table 9'),
    ]
    data = [{'study': study, 'contrast': label, 'treatment': a, 'control': b,
             'difference_pp': round(a-b, 2), 'source': f'https://arxiv.org/html/{id}', 'table': table}
            for study, label, a, b, id, table in rows]
    (HERE / 'figure-data.json').write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    palette = {'Pixel Reasoner': '#176b87', 'VLM-R³': '#8856a7', 'DeepEyes': '#b66e1d'}
    for i, d in enumerate(data):
        y = len(data)-1-i
        delta = d['difference_pp']
        ax.plot([0, delta], [y, y], color=palette[d['study']], lw=3)
        ax.scatter(delta, y, color=palette[d['study']], s=65, zorder=3)
        ax.text(delta+.3, y+.12, f'{delta:+.1f}', color=palette[d['study']], fontsize=10, weight='bold')
    ax.set_yticks(range(len(data)), [f"{d['study']} | {d['contrast']}" for d in reversed(data)])
    ax.axvline(0, c='#666', lw=1)
    ax.set(xlim=(-1.5, 14), ylim=(-.6, 6.6), xlabel='Within-study accuracy difference (percentage points)')
    ax.set_title('Primary-study contrasts — different tasks, no pooled effect, no DAR results', pad=17, fontsize=13, weight='bold')
    fig.text(.5, -.01, 'Reported table values. No confidence intervals available for these contrasts; statistical significance is unknown.', ha='center', fontsize=9)
    fig.tight_layout()
    save(fig, 'primary-ablation')


def plot_map():
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set(xlim=(0, 12), ylim=(0, 6.4))
    ax.axis('off')
    def box(x, y, w, h, title, subtitle, color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.10',facecolor=color,edgecolor='#526574',lw=1))
        ax.text(x+w/2,y+h*.68,title,ha='center',va='center',fontsize=11,weight='bold')
        ax.text(x+w/2,y+h*.28,subtitle,ha='center',va='center',fontsize=9)
    def arrow(a,b, label=None):
        ax.annotate('', xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#526574','lw':1.5})
        if label: ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,ha='center',fontsize=8)
    box(.2,4.1,2.1,1.25,'Video + sampling','Real frames / timestamps','#e6f1f5')
    box(3.05,4.1,2.5,1.25,'Evidence record','Observed facts ≠ appraisal','#e6f1f5')
    box(6.25,4.1,2.35,1.25,'Viewer appraisal','Context + observed change','#e6f1f5')
    box(9.35,4.1,2.35,1.25,'DAR output','Time / emotion / reason','#e6f1f5')
    arrow((2.4,4.75),(2.95,4.75)); arrow((5.65,4.75),(6.15,4.75)); arrow((8.7,4.75),(9.25,4.75))
    box(.4,1.8,3.1,1.2,'Optional reinspection','Only if evidence tests justify it','#fff0d9')
    box(4.1,1.8,3.25,1.2,'Verifier + coverage','Calibrate before semantic RL','#fff0d9')
    box(8.0,1.8,3.3,1.2,'Independent evaluation','Human support + joint task F1','#e6f3e9')
    arrow((3.1,4.05),(2.4,3.1),'unresolved premise')
    arrow((.75,3.1),(.75,4.0),'new observation')
    arrow((4.6,4.0),(5.3,3.1)); arrow((7.2,4.0),(6.3,3.1))
    arrow((10.5,4.0),(10.0,3.1))
    ax.text(6,.75,'Controls: verified prose vs structured record • uniform vs adaptive frames • flat list vs graph',ha='center',fontsize=10)
    ax.text(6,.30,'Proposed experiment architecture. No new DAR performance or causal effect has been measured.',ha='center',fontsize=9,color='#555555')
    ax.set_title('Where the proposed changes enter DAR',fontsize=16,weight='bold',pad=12)
    save(fig,'intervention-map')


def provenance():
    records = json.loads((HERE/'sources/arxiv-records.json').read_text(encoding='utf-8'))
    versions = {d['arxiv_id']: d['arxiv_id_versioned'] for d in records['entries']}
    files = []
    for path in sorted((HERE/'sources').glob('*.html')):
        files.append({'file': str(path.relative_to(HERE)), 'retrieved_url': 'https://arxiv.org/html/'+path.stem,
                      'version': versions.get(path.stem,path.stem), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    code = []
    for relative in ['ms-swift/examples/train/grpo/plugin/dar_plugin.py', 'data_construction/05_qwen3vl_stream_affect_reason.py',
                     'data_construction/08_dual_consistency_committee.py', 'test.py']:
        code.append({'path':relative,'sha256':hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()})
    payload = {'access_date':'2026-09-17','retrieval_scope':'Targeted four surveys plus three primary studies; not exhaustive.',
               'api':{'endpoint':'https://export.arxiv.org/api/query','id_list':'2505.04921,2503.12605,2506.23918,2504.21277',
                      'max_results':4,'expected':4,'returned':records['returned']},'full_text':files,'code':code}
    (HERE/'provenance.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')


def render_html():
    md = MarkdownIt('commonmark', {'html':False}).enable('table')
    content = md.render((HERE/'report.vi.md').read_text(encoding='utf-8'))
    css = '''body{font:17px/1.7 system-ui,sans-serif;color:#20313c;background:#fafafa;margin:0}
    main{max-width:1080px;margin:auto;background:white;padding:48px 60px}h1{font-size:34px;line-height:1.25}
    h2{margin-top:48px;color:#176b87}h3{margin-top:32px}a{color:#116b90}img{max-width:100%;height:auto}
    table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.5;margin:24px 0}
    th,td{border:1px solid #dce3e7;padding:10px;vertical-align:top}th{background:#eaf1f5}
    tr:nth-child(even){background:#f8fafb}pre{white-space:pre-wrap;background:#f0f4f6;padding:18px;font-size:13px}
    code{overflow-wrap:anywhere}p,li{overflow-wrap:break-word}@media(max-width:700px){main{padding:20px}table{font-size:12px}}
    @media print{body{background:white}main{padding:0;max-width:none}h2,h3{break-after:avoid}img,tr{break-inside:avoid}a{color:inherit}}'''
    (HERE/'report.vi.html').write_text('<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Multimodal reasoning, Perception-R1, SATORI-R1 và DAR</title><style>'+css+'</style><main>'+content+'</main></html>',encoding='utf-8')


if __name__ == '__main__':
    result = probes()
    plot_reward(result)
    plot_primary()
    plot_map()
    provenance()
    render_html()
    print('Built probes, 3 PNG/SVG figures, provenance, data and HTML report.')
