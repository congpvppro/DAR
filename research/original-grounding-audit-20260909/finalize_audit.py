import json, pathlib, collections, math, csv, html, os, hashlib
P=pathlib.Path(__file__).resolve().parent
R={x['video_id']:x for x in json.loads((P/'records.json').read_text())}
extra=json.loads((P/'extra-verification.json').read_text())
A=[]
for i in range(4):
    a=json.loads((P/f'annotations-group-{i}.json').read_text())
    for x in a:
        for im in extra.get(x['video_id'],[]):
            if im not in x['inspected_images']: x['inspected_images'].append(im)
            for c in x['claims']:
                if im not in c['evidence_images']: c['evidence_images'].append(im)
    (P/f'annotations-group-{i}.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
    A+=a
D={x['video_id']:x for x in A}
chosen=['01203','22109','05367','04466','05128','10509','02872','60095','58018','52516']
def dump(name,data): (P/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def wilson(k,n=64):
    z=1.959963984540054;p=k/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [round(100*(c-h),2),round(100*(c+h),2)]
counts=dict(collections.Counter(x['status'] for x in A))
types={t:[x['video_id'] for x in A if any(c['judgment']=='contradicted' and c['claim_type']==t for c in x['claims'])] for t in ['invented_event','wrong_object','wrong_attribute']}
events=len(types['invented_event'])
s={'n':len(A),'status_counts':counts,'confirmed_percent':100*counts['confirmed']/64,'confirmed_wilson95_percent':wilson(counts['confirmed']),'event_error_videos':events,'event_error_percent':100*events/64,'event_error_wilson95_percent':wilson(events),'type_video_ids_nonexclusive':types,'example_ids':chosen,'new_inference_runs':0,'independent_human_reviewers':0,'method':'Single AI visual audit of exact 16 model frames and 2 fps source frames plus final; targeted denser review. Reused original-input released-prompt outputs. Rates are response-level within selected 64 videos, not full-benchmark prevalence.'}
dump('annotations.json',A);dump('summary.json',s)
with (P/'annotations.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['video_id','status','summary_vi','contradicted_types'])
    for x in A:w.writerow([x['video_id'],x['status'],x['summary_vi'],','.join(sorted({c['claim_type'] for c in x['claims'] if c['judgment']=='contradicted'}))])
(P/'protocol-amendment.md').write_text('''# Thay đổi cách thực hiện — 2026-09-09

Protocol gốc được giữ nguyên để lưu dấu vết. Ba tác vụ reviewer phụ dừng do giới hạn sử dụng trước khi lưu bảng chấm. Vì vậy, kế hoạch bốn nhóm chạy song song và kiểm tra chéo không được thực hiện như dự kiến: tác nhân chính trực tiếp đọc và chấm toàn bộ 64 video, chia thành bốn file để lưu tiến độ. Các nhãn parent-group-0..3 là cùng một tác nhân, không phải bốn người chấm độc lập.

Đã xem reasoning đầy đủ, 16 frame model và tất cả bảng frame nguồn 2 fps + frame cuối cho từng ID. Kiểm tra dày hơn có mục tiêu ở các đoạn nghi ngờ, lưu trong inspected_images; không tuyên bố toàn bộ 38 ca dương tính đều đã được chấm độc lập lần hai. Mười ví dụ được chọn sau khi chấm để dễ kiểm tra, không dùng riêng mười ca này làm mẫu tính tỷ lệ.

Đây là audit do AI bằng frame giải mã, không phải người xem phát video liên tục và nghe âm thanh. Tác nhân đã có bối cảnh nghiên cứu trước đó; không phải chấm mù. Chưa có đánh giá đồng thuận giữa người chấm. Khoảng Wilson không đo sai số gán nhãn. Các nhận xét ambiguous không được chuyển thành lỗi chỉ vì có nghi ngờ.
''')
labels={'confirmed':'Có lỗi rõ','ambiguous':'Chưa kết luận','no_clear_error':'Chưa thấy lỗi rõ'}
lines=['# DAR-R1: đối chiếu reasoning trên 64 video gốc','',f'Đã kiểm tra **64/64** output với video gốc: **38/64 (59,38%)** có ít nhất một khẳng định hình ảnh mâu thuẫn rõ; riêng khẳng định sự kiện sai có **{events}/64 ({100*events/64:.2f}%)**. Đây là kết quả audit bằng AI, cần người kiểm tra lại; không phải tỷ lệ đã xác nhận cho toàn benchmark.','', '## Dữ liệu và cách kiểm tra','', 'Dùng lại 64 ID natural từ thí nghiệm timing-corrected trước đó, điều kiện original, prompt_kind=released. Đây là video thuộc benchmark DAR/VCE, không phải video ngoài benchmark. Mẫu này gồm video ngắn ≤30 giây, được chọn cho nghiên cứu trước và tách khỏi 48 video pilot; không coi là mẫu đại diện toàn benchmark. Không chạy thêm inference trong audit này. Nguồn output: [shard 0](../hypothesis-validation-20260908/timing-corrected/dar-shard0.jsonl), [shard 1](../hypothesis-validation-20260908/timing-corrected/dar-shard1.jsonl).','', 'Đã đọc toàn bộ reasoning từng output; xem 16 frame đầu vào model và chuỗi frame nguồn 2 fps + frame cuối (2.002 lượt frame nguồn cơ sở), bổ sung frame dày hơn tại các đoạn quyết định. Việc tái dựng đầu vào đã khớp SHA256 tensor ở **64/64** ca theo [kiểm tra chuẩn bị](preparation-verification.json). Video “gốc” nghĩa là không có can thiệp đảo/static/gray; model vẫn chỉ nhận 16 frame theo cấu hình lần chạy này. Không khẳng định đã xem phát liên tục hoặc nghe toàn bộ MP4.','', 'Đơn vị đếm là **video/output có ≥1 khẳng định sai**, không phải phần trăm câu reasoning sai. Chỉ sai thời điểm, khác cảm xúc, suy đoán ý định hoặc âm thanh không được tính thành bịa sự kiện. “Chưa thấy lỗi rõ” không có nghĩa toàn bộ reasoning đúng. Một tác nhân AI chấm cả 64, không có người chấm độc lập; xem [thay đổi protocol](protocol-amendment.md).','', '## Kết quả','', '| Nhóm | Video | Tỷ lệ trên 64 |','|---|---:|---:|', '| Có ít nhất một lỗi hình ảnh rõ | 38 | 59,38% |',f'| Trong đó: có khẳng định sự kiện sai | {events} | {100*events/64:.2f}% |','| Chưa đủ bằng chứng kết luận | 17 | 26,56% |','| Chưa thấy lỗi rõ | 9 | 14,06% |','',f'Khoảng Wilson 95%: lỗi hình ảnh {s["confirmed_wilson95_percent"]}% và lỗi sự kiện {s["event_error_wilson95_percent"]}%. Chỉ là khoảng tham khảo theo giả định nhị thức; lựa chọn mẫu và sai số người/AI chấm không được bao phủ. Không dùng khoảng này để suy ra tỷ lệ toàn benchmark.','',f'Các loại lỗi có thể chồng lặp: sự kiện {events} video; vật thể {len(types["wrong_object"])} video; thuộc tính {len(types["wrong_attribute"])} video. Không cộng ba số này. Danh sách claim là bằng chứng đủ để phân loại từng output, không phải bộ tách toàn bộ claim để tính precision ở cấp câu.','', '## Mười ví dụ để kiểm tra','', 'Mở [trang đối chiếu 64 video](review.html) bằng trình duyệt để phát MP4 và đọc output đầy đủ. Mỗi ví dụ dưới đây có trích nguyên văn, khoảng thời gian model gán và ảnh bằng chứng có timestamp.']
for id in chosen:
    x=D[id];r=R[id];v=os.path.relpath(r['video_path'],P)
    lines += ['',f'### Video {id}','',x['summary_vi'],'']
    for c in x['claims']:
        if c['judgment']!='contradicted':continue
        lines += [f'> {c["quote"]}','',f'Model gán: {c["claimed_interval"]} giây. Khoảng nguồn đối chiếu: {c["source_interval"]} giây.','']
    ims=[q for q in x['inspected_images'] if 'dense' in q or 'verification' in q]
    lines += [f'[Video gốc]({v}) · [Reasoning đầy đủ](cases/{id}/raw.txt) · [16 frame model]({r["model_sheets"][0]}) · '+' · '.join(f'[Ảnh đối chiếu {i+1}]({im})' for i,im in enumerate(ims))]
lines += ['', '## Rút ra được gì','', 'Có bằng chứng trực tiếp cho phát biểu hẹp: **DAR-R1 có thể tạo tiền đề hình ảnh sai ngay trên input gốc với prompt phát hành**. Hiện tượng không chỉ xuất hiện khi thay video bằng gray/static. Các ca tháp vẫn đứng nhưng kể sụp (01203), ngã nhưng kể tiếp đất êm (60095), lấy ghế nhưng kể ký giấy (05128) cho thấy sai cả diễn biến/kết quả được dùng trong lời giải thích.','', 'Research gap được hỗ trợ là đánh giá độ bám bằng chứng của lời giải thích theo sự kiện, tách khỏi tính hợp lý của nhãn cảm xúc và lỗi timestamp. Kết quả này chưa chứng minh cơ chế nội tại “model quyết định nhãn trước rồi bịa lý do”, chưa chứng minh mọi reasoning sai, và chưa đo quan hệ nhân quả giữa lỗi grounding với điểm benchmark. Muốn xác nhận cơ chế cần thí nghiệm riêng; muốn công bố tỷ lệ cần chấm độc lập bởi người và mở rộng mẫu.','', '## Toàn bộ 64 kết quả','', '| ID | Kết quả | Quan sát |','|---|---|---|']
for x in A: lines += [f'| [{x["video_id"]}](review.html#v{x["video_id"]}) | {labels[x["status"]]} | {x["summary_vi"].replace("|","/")} |']
lines += ['', 'Dữ liệu có thể kiểm toán: [annotations.json](annotations.json), [CSV](annotations.csv), [summary.json](summary.json), [manifest nguồn](records.json), [protocol](protocol.md), [kiểm tra cuối](final-verification.json).']
(P/'report.vi.md').write_text('\n'.join(lines)+'\n')
e=html.escape
parts=['<!doctype html><html lang="vi"><meta charset="utf-8"><title>DAR-R1 original grounding audit</title><style>body{font:17px system-ui;max-width:1100px;margin:2em auto;padding:1em;line-height:1.5}video{width:100%;max-height:480px}pre{white-space:pre-wrap;overflow-wrap:anywhere}img{max-width:100%}article{border-top:2px solid #999;margin-top:3em}a{margin-right:.6em}blockquote{border-left:4px solid #999;padding-left:1em}</style><h1>DAR-R1 — 64 video gốc</h1><p>38/64 có lỗi hình ảnh; '+str(events)+'/64 có sự kiện sai. Một AI kiểm tra frame; chưa có xác nhận độc lập bởi người. Nhấn ID, phát video và đối chiếu câu trích. Các mốc là giây trong video nguồn.</p><p><a href="report.vi.md">Báo cáo</a><a href="annotations.json">Bảng chấm</a></p><h2>10 ví dụ</h2>']
parts += [f'<a href="#v{id}">{id}</a>' for id in chosen]
parts += ['<h2>Tất cả 64</h2>']+[f'<a href="#v{x["video_id"]}">{x["video_id"]}</a>' for x in A]
for x in A:
    id=x['video_id'];r=R[id];v=os.path.relpath(r['video_path'],P)
    parts += [f'<article id="v{id}"><h2>{id} — {labels[x["status"]]}</h2><p>{e(x["summary_vi"])}</p><video id="player{id}" controls preload="none" src="{e(v)}"></video>']
    for c in x['claims']:
        t=c['source_interval'][0]
        parts += [f'<blockquote>{e(c["quote"])}</blockquote><p>{e(c["judgment"])} / {e(c["claim_type"])} — model: {c["claimed_interval"]}s; nguồn: {c["source_interval"]}s. <button onclick="document.getElementById(\'player{id}\').currentTime={t}">Tới {t}s</button></p><p>{e(c["observation_vi"])}</p>']
    parts += [f'<details><summary>Reasoning đầy đủ</summary><pre>{e(r["raw_output"])}</pre></details><details><summary>Frame model và nguồn đã kiểm tra</summary>']
    for im in x['inspected_images']:parts += [f'<p><a href="{e(im)}">{e(im)}</a></p><img loading="lazy" src="{e(im)}" alt="{e(im)}">']
    parts += ['</details></article>']
parts += ['</html>'];(P/'review.html').write_text('\n'.join(parts))
print(json.dumps(s,ensure_ascii=False,indent=2))
