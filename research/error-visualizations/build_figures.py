"""Figures from frozen local DAR error analyses. No model inference."""
import csv
import hashlib
import html
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/dar-error-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCES = {
    'full': ROOT / 'research/gap-study-20260907/full-error-breakdown.json',
    'original': ROOT / 'research/original-grounding-audit-20260909/summary.json',
    'annotations': ROOT / 'research/original-grounding-audit-20260909/annotations.json',
    'static': ROOT / 'research/hypothesis-validation-20260908/analysis.json',
}
D = {k: json.loads(p.read_text()) for k, p in SOURCES.items()}
full, original, static = D['full'], D['original'], D['static']
panel = full['panels']['0.5']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.facecolor': 'white', 'pdf.fonttype': 42})
RED, BLUE, GOLD, GREY, GREEN = '#b63a47', '#276b9c', '#d6a238', '#c9d0d8', '#388478'
figures = []
pdf = PdfPages(OUT / 'DAR-error-analysis.pdf')


def save(fig, name, title, note):
    fig.savefig(OUT / f'{name}.png', dpi=170)
    fig.savefig(OUT / f'{name}.svg')
    pdf.savefig(fig)
    figures.append(dict(name=name, title=title, note=note))
    plt.close(fig)


def footer(fig, text):
    fig.text(.055, .025, text, fontsize=9, color='#455565', va='bottom')


# 1. Most frequent directional mistakes, with their class denominators.
mistakes = [r for r in panel['confusion'] if r['gt'] != r['pred']][:12]
fig, ax = plt.subplots(figsize=(13, 8))
labels = [f'{r["gt"]} → {r["pred"]}' for r in mistakes]
ax.barh(labels, [r['count'] for r in mistakes], color=RED, height=.65)
for i, r in enumerate(mistakes):
    ax.text(r['count'] + .25, i, f'{r["count"]}/{r["qualified_gt"]} = {100*r["count"]/r["qualified_gt"]:.1f}%', va='center')
ax.invert_yaxis(); ax.set_xlim(0, 25)
ax.set_xlabel('Số cặp đoạn nhầm nhãn (không phải số video)')
ax.set_title('DAR-R1: các cặp cảm xúc hay nhầm trên 1.441 video', loc='left', fontsize=17, pad=20)
ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
fig.subplots_adjust(left=.36, right=.98, top=.88, bottom=.15)
footer(fig, 'Chiều: nhãn benchmark → dự đoán. Ghép cùng chỉ số, chỉ giữ IoU ≥ 0,5 (1.550 cặp).\nMẫu số bên mỗi thanh = tổng cặp đạt IoU của nhãn GT đó. Sai nhãn không tự chứng minh bịa sự kiện.')
save(fig, '01-emotion-confusions', 'Các cặp cảm xúc hay nhầm', 'Tần suất và tỷ lệ nhầm có điều kiện theo nhãn GT; chỉ xét các đoạn khớp thời gian.')

# 2. Full row-normalized matrix, all 27 classes and qualified supports.
classes = [r['emotion'] for r in panel['per_class']]
support = {r['emotion']: r['qualified'] for r in panel['per_class']}
lookup = {(r['gt'], r['pred']): r['count'] for r in panel['confusion']}
matrix = np.array([[100*lookup.get((g,p), 0)/support[g] for p in classes] for g in classes])
fig, ax = plt.subplots(figsize=(16, 14))
im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=100)
ax.set_xticks(range(len(classes)), classes, rotation=65, ha='right', fontsize=8)
ax.set_yticks(range(len(classes)), [f'{c} (n={support[c]})' for c in classes], fontsize=9)
for i in range(len(classes)):
    for j in range(len(classes)):
        if matrix[i,j] >= 10 or (i == j and matrix[i,j] > 0):
            ax.text(j, i, f'{matrix[i,j]:.0f}', ha='center', va='center', fontsize=7, color='white' if matrix[i,j]>50 else '#172c42')
ax.set_xlabel('Nhãn model dự đoán'); ax.set_ylabel('Nhãn benchmark')
ax.set_title('Ma trận nhầm lẫn cảm xúc — chuẩn hóa theo hàng (%)', loc='left', fontsize=18, pad=18)
fig.colorbar(im, ax=ax, fraction=.035, pad=.02, label='% số cặp đủ overlap của nhãn GT')
fig.subplots_adjust(left=.23, right=.92, top=.91, bottom=.23)
footer(fig, 'Full benchmark 1.441 video; 1.550 cặp cùng chỉ số có IoU ≥ 0,5. Đường chéo = đúng nhãn.\nKhông bao gồm đoạn không ghép được hoặc IoU < 0,5. Nhãn ít mẫu như Craving (n=5) có độ bất định lớn.')
save(fig, '02-confusion-matrix', 'Ma trận đầy đủ 27 cảm xúc', 'Mỗi hàng dùng mẫu số riêng; số n cạnh nhãn là số cặp đạt IoU, không phải toàn bộ đoạn GT.')

# 3. Per-class correctness.
ordered = sorted(panel['per_class'], key=lambda r: r['correct']/r['qualified'])
fig, ax = plt.subplots(figsize=(12, 11))
rates = [100*r['correct']/r['qualified'] for r in ordered]
ax.barh([r['emotion'] for r in ordered], rates, color=[RED if r<25 else BLUE for r in rates], height=.72)
for i, (r, rate) in enumerate(zip(ordered, rates)):
    ax.text(rate+.7, i, f'{r["correct"]}/{r["qualified"]} · {rate:.1f}%', va='center', fontsize=9)
ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel('Tỷ lệ đúng nhãn trong cặp IoU ≥ 0,5 (%)')
ax.set_title('Full benchmark: tỷ lệ đúng nhãn theo từng cảm xúc', loc='left', fontsize=16, pad=20)
ax.axvline(100*483/1550, color='#5b6570', linestyle='--', label='Trung bình 31,16%', zorder=0)
ax.legend(loc='lower right'); ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
fig.subplots_adjust(left=.25, right=.97, top=.91, bottom=.1)
footer(fig, 'Tỷ lệ có điều kiện, không phải recall trên toàn bộ GT. Đỏ: <25%. Không xếp Craving là kết quả vững vì chỉ có 5 cặp.\nMức độ đúng được xác định theo nhãn benchmark; chưa chấm lại cảm xúc bằng người xem độc lập.')
save(fig, '03-class-weaknesses', 'Độ đúng theo từng cảm xúc', 'Hiển thị cả tử số và mẫu số, tránh kết luận mạnh từ nhãn hiếm.')

# 4. Segment count matrix and under/over by number of GT segments.
countmat = np.zeros((6,6), dtype=int)
for r in full['segment_count_matrix']:
    countmat[r['gt_count']-1,r['predicted_count']-1] = r['videos']
fig, (ax,bx) = plt.subplots(1, 2, figsize=(14,7), gridspec_kw={'width_ratios':[1,1.1]})
ax.imshow(countmat, cmap='Blues')
for i in range(6):
    for j in range(6):
        ax.text(j,i,str(countmat[i,j]),ha='center',va='center',color='white' if countmat[i,j]>180 else '#172c42',fontsize=12)
ax.set_xticks(range(6),range(1,7));ax.set_yticks(range(6),range(1,7))
ax.set_xlabel('Số đoạn dự đoán');ax.set_ylabel('Số đoạn GT');ax.set_title('Ma trận số đoạn: số video')
totals = countmat.sum(axis=1)
under = np.array([countmat[i,:i].sum() for i in range(6)])
equal = np.diag(countmat); over = totals-under-equal
left=np.zeros(6)
for vals,col,lab in [(under,RED,'Thiếu đoạn'),(equal,GREEN,'Đúng số đoạn'),(over,GOLD,'Thừa đoạn')]:
    widths=100*vals/totals
    bx.barh(range(6),widths,left=left,color=col,label=lab)
    for i,(v,w,l) in enumerate(zip(vals,widths,left)):
        if w>=12:bx.text(l+w/2,i,f'{v}\n{w:.0f}%',ha='center',va='center',fontsize=9,color='white' if col!=GOLD else '#172c42')
    left+=widths
bx.set_yticks(range(6),[f'{i+1} pha GT (n={totals[i]})' for i in range(6)])
bx.invert_yaxis();bx.set_xlim(0,100);bx.set_xlabel('% video trong nhóm');bx.set_title('Video nhiều pha thường bị dự đoán thiếu')
bx.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3,fontsize=9)
fig.suptitle('Full benchmark: 583 thiếu · 570 đúng · 288 thừa / 1.441 video',fontsize=17,y=.97)
fig.subplots_adjust(left=.07,right=.98,top=.84,bottom=.24,wspace=.4)
footer(fig,'90/109 video một pha bị chia thừa; 402/664 video ba pha và 97/114 video bốn pha bị chia thiếu.\nNhóm 5–6 pha chỉ có 15 video. Đây là lỗi phân đoạn, không tự là bằng chứng bịa sự kiện.')
save(fig,'04-segmentation','Chia thừa và thiếu pha','Ma trận đếm và tỷ lệ theo độ phức tạp của nhãn GT.')

# 5. Hallucination evidence: mutually exclusive original categories, separate static protocols.
fig,(ax,bx)=plt.subplots(1,2,figsize=(15,8),gridspec_kw={'width_ratios':[1,1.15]})
oc=original['status_counts'];ev=original['event_error_videos']
vals=[ev,oc['confirmed']-ev,oc['ambiguous'],oc['no_clear_error']]
labs=['Có sự kiện sai','Chỉ lỗi hình ảnh khác\nđã ghi nhận','Chưa kết luận','Chưa thấy lỗi rõ']
ax.barh(labs,np.array(vals)/64*100,color=[RED,BLUE,GOLD,GREY])
for i,v in enumerate(vals):ax.text(v/64*100+.7,i,f'{v}/64 = {v/64*100:.2f}%',va='center')
ax.invert_yaxis();ax.set_xlim(0,65);ax.set_xlabel('% trên 64 output');ax.set_title('VIDEO GỐC · prompt phát hành\nMột AI đối chiếu chuỗi frame',loc='left',fontsize=14)
keys=['dar:released','dar:conservative','qwen:released']
left=np.zeros(3)
for key,color,lab in [('confirmed',RED,'Có sự kiện sai'),('ambiguous',GOLD,'Chưa kết luận'),('none',GREY,'Chưa thấy lỗi rõ')]:
    counts=np.array([static['event_rates'][k][key] for k in keys]);widths=counts/64*100
    bx.barh(range(3),widths,left=left,color=color,label=lab,height=.55)
    for i,(n,w,l) in enumerate(zip(counts,widths,left)):
        if n:bx.text(l+w/2,i,str(n),ha='center',va='center',color='white' if color==RED else '#172c42',fontsize=11)
    left+=widths
bx.set_yticks(range(3),['DAR · phát hành','DAR · thận trọng','Qwen · phát hành'])
bx.invert_yaxis();bx.set_xlim(0,100);bx.set_xlabel('% trên 64 output; số trong thanh = output')
bx.set_title('STATIC · lặp một frame 16 lần\nCó đối chiếu AI theo protocol cũ',loc='left',fontsize=14)
bx.legend(loc='upper left',bbox_to_anchor=(0,-.14),fontsize=9)
fig.suptitle('Bằng chứng về sự kiện được viện dẫn sai',fontsize=19,y=.97)
fig.subplots_adjust(left=.16,right=.98,top=.79,bottom=.28,wspace=.65)
footer(fig,'Video gốc: 27/64 có sự kiện sai; 38/64 có lỗi hình ảnh (27 + 11), không cộng 27 và 38.\nStatic: DAR 47/64 (phát hành), 48/64 (thận trọng). Cùng ID nhưng khác input và quy trình chấm: không suy ra hiệu ứng nhân quả.\nQwen 0 lỗi xác nhận ở static vẫn có 21 trường hợp mơ hồ. Chưa có người chấm độc lập; không suy rộng tỷ lệ ra full benchmark.')
save(fig,'05-grounding-evidence','Bằng chứng bịa sự kiện: gốc và static','Hai panel tách điều kiện và cách chấm. Không diễn giải tỷ lệ static là tỷ lệ trên video gốc.')

# 6. Uncertainty and sensitivity check, with the practical reference explicitly labeled.
rates=[original['event_error_percent'],original['confirmed_percent'],100*static['event_rates']['dar:released']['rate_all'],100*static['event_rates']['dar:conservative']['rate_all']]
cis=[original['event_error_wilson95_percent'],original['confirmed_wilson95_percent']]+[[100*x for x in static['event_rates'][k]['wilson95']] for k in keys[:2]]
fig,ax=plt.subplots(figsize=(13,7))
labels=['Gốc · phát hành · sự kiện sai (27/64)','Gốc · phát hành · mọi lỗi hình ảnh (38/64)','Static · phát hành · sự kiện sai (47/64)','Static · thận trọng · sự kiện sai (48/64)']
for i,(r,ci) in enumerate(zip(rates,cis)):
    ax.errorbar(r,i,xerr=[[r-ci[0]],[ci[1]-r]],fmt='o',color=BLUE if i==1 else RED,capsize=6,markersize=8)
    ax.text(99,i+.18,f'{r:.2f}% [{ci[0]:.2f}; {ci[1]:.2f}]',ha='right',va='top',fontsize=10)
ax.set_yticks(range(4),labels);ax.set_ylim(3.55,-.4);ax.set_xlim(0,100);ax.set_xlabel('Tỷ lệ output có lỗi và khoảng Wilson 95%')
ax.set_title('Tỷ lệ quan sát và độ bất định — mỗi điều kiện n=64',loc='left',fontsize=16,pad=20)
ax.grid(axis='x',alpha=.2);fig.subplots_adjust(left=.39,right=.99,top=.83,bottom=.28)
footer(fig,'Khoảng Wilson chỉ mô tả biến thiên theo giả định nhị thức; không bao phủ sai số AI chấm hoặc thiên lệch chọn mẫu.\nH1 static/phát hành so với mốc thực hành 5%: p = 4,18 × 10⁻⁴⁷. Mốc 5% do study đặt trước, không phải chuẩn lĩnh vực.\nSố liệu ủng hộ sự tồn tại của viện dẫn sai; chưa chứng minh cơ chế “chọn cảm xúc trước rồi bịa lý do”.')
save(fig,'06-grounding-uncertainty','Khoảng bất định và kiểm định static','Không dùng p-value hoặc CI để tuyên bố biết cơ chế nội tại hay tỷ lệ lỗi toàn benchmark.')
pdf.close()

# Portable offline gallery, evidence links, and machine-readable provenance.
cards=[]
for item in figures:
    n=item['name']
    cards.append(f'<section><h2>{html.escape(item["title"])}</h2><p>{html.escape(item["note"])}</p><a href="{n}.png">PNG</a> · <a href="{n}.svg">SVG</a><a href="{n}.png"><img loading="lazy" src="{n}.png" alt="{html.escape(item["title"])}"></a></section>')
page='''<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>DAR — lỗi benchmark và bằng chứng grounding</title><style>body{font:17px system-ui;max-width:1250px;margin:30px auto;padding:0 20px;line-height:1.6;color:#203246}img{width:100%;display:block;margin:18px 0}section{border-top:1px solid #d5dce3;margin-top:40px}a{color:#216694}aside{background:#eef3f8;padding:18px}</style><h1>DAR-R1: lỗi benchmark và bằng chứng grounding</h1><p>Full benchmark: 1.441 video. Audit grounding: mẫu 64 video; static dùng cùng ID nhưng input đã thay bằng một frame lặp lại. Không chạy inference mới.</p><aside>Confusion cảm xúc và lỗi phân đoạn đo sai khác với nhãn benchmark, không tự chứng minh bịa. Bằng chứng bịa nằm ở khẳng định sự kiện trái với input và ảnh đối chiếu. Các tỷ lệ grounding được chấm bởi AI; chưa có người chấm độc lập.</aside><p><a href="DAR-error-analysis.pdf">Tải PDF 6 trang</a> · <a href="../original-grounding-audit-20260909/review.html">Phát video, xem reasoning và ảnh bằng chứng</a> · <a href="../gap-study-20260907/full-error-breakdown.vi.md">Bảng lỗi đầy đủ</a></p>'''
page+=''.join(cards)
page+='<section><h2>Ví dụ có thể kiểm tra trực tiếp</h2><table><tr><th>ID</th><th>Model kể</th><th>Quan sát trong audit</th></tr>'
for vid,q,obs in [('01203','tower suddenly collapses at 1.8 seconds','Tháp chai vẫn đứng; người thực hiện ăn mừng.'),('05128','he unexpectedly pulls out a pen and begins signing the paper','Người mở quà và giơ ghế gấp.'),('60095','before landing smoothly on the tiled platform','Người trượt ván ngã xuống bậc thang.')]:
    page+=f'<tr><td><a href="../original-grounding-audit-20260909/review.html#v{vid}">{vid}</a></td><td>{html.escape(q)}</td><td>{obs}</td></tr>'
page+='</table></section><p><a href="provenance.json">Nguồn dữ liệu, hash và kiểm tra số đếm</a></p></html>'
(OUT/'index.html').write_text(page)
assert sum(r['count'] for r in panel['confusion']) == 1550
assert sum(r['count'] for r in panel['confusion'] if r['gt']==r['pred']) == 483
assert countmat.sum()==1441 and under.sum()==583 and equal.sum()==570 and over.sum()==288
annotations=D['annotations']
assert sum(a['status']=='confirmed' for a in annotations)==38
assert sum(any(c['judgment']=='contradicted' and c['claim_type']=='invented_event' for c in a['claims']) for a in annotations)==27
assert vals==[27,11,17,9]
assert all(static['event_rates'][k]['confirmed']+static['event_rates'][k]['none']+static['event_rates'][k]['ambiguous']==64 for k in keys)
provenance=dict(sources={k:dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in SOURCES.items()},figures=figures,verification='passed: matrix totals, segmentation counts, original claim annotations, static denominators',new_inference_runs=0)
(OUT/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'figures':len(figures),'png':6,'svg':6,'pdf_pages':6,'checks':'passed'},ensure_ascii=False))
