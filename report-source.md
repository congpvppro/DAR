# Nghiên cứu sâu và tái lập benchmark DAR-R1

**Bài báo:** *Benchmarking Dynamic Affective Reasoning: A Viewer-Centric Video Emotion Dataset*  
**Phiên bản khảo sát:** arXiv:2607.10238v1, 11-07-2026; bài báo ghi “Accepted by ECCV 2026”  
**Ngày truy cập và thực nghiệm:** 03-09-2026 (UTC)  
**Mã nguồn được cố định tại commit:** `229baf251a48d14bfae417a4f36cc59f6fabaac1`

## Tóm tắt điều hành

DAR biến nhận diện cảm xúc video từ một nhãn tĩnh thành bài toán ba phần: phát hiện lúc cảm xúc chủ đạo của người xem thay đổi, gán một trong 27 cảm xúc cho từng đoạn, rồi giải thích nguyên nhân bằng bằng chứng thị giác và diễn tiến trước đó. Bản phát hành có 15.087 video và 36.908 đoạn cảm xúc; chia 13.646 video huấn luyện và 1.441 video kiểm thử. DAR-R1 là checkpoint Qwen2.5-VL-3B được tinh chỉnh bằng một giai đoạn SFT “cold start”, tiếp theo là GRPO với phần thưởng cấu trúc, số đoạn, định vị thời gian, nhãn cảm xúc và chất lượng lý giải.

Tái lập ở đây dùng đúng annotations và checkpoint công khai, toàn bộ 1.441 video test lấy từ VCE, và script `test.py` của tác giả. Một smoke test đầu-cuối đã thành công; benchmark đầy đủ đang được chạy trên bốn RTX 3090 độc lập. Kết quả cuối chỉ được điền sau khi kiểm tra đủ coverage, parse thành công và tái tính metric độc lập.

## 1. Bài toán và đóng góp

Nhãn cảm xúc toàn video che khuất chuyển biến theo thời gian, còn nhận diện cảm xúc nhân vật không đồng nghĩa với cảm xúc mà video gây ra cho người xem. DAR vì vậy yêu cầu đầu ra JSON gồm các đoạn liên tục phủ kín video; mỗi đoạn có `start_time`, `end_time`, `emotion` và một `reason` hướng đến người xem. Video được xử lý như **video im lặng** trong prompt đánh giá công khai.

Ba đóng góp chính là:

1. benchmark viewer-centric có phân đoạn động và lý giải nhân quả;
2. quy trình tạo dữ liệu nhiều tầng, kết hợp mô hình sinh, căn chỉnh thị giác và hai mô hình giám khảo;
3. DAR-R1, một baseline 3B dùng SFT + GRPO, vượt các baseline mở và đóng được báo cáo trong bài.

Nguồn chính: [bài báo arXiv](https://arxiv.org/abs/2607.10238), [mã nguồn chính thức](https://github.com/Zhang-Zhiyan/DAR).

## 2. Dữ liệu DAR

### 2.1 Quy trình xây dựng

Tập nguồn là Viewer-Centric Emotion (VCE), gồm 61.046 video và cường độ của 27 cảm xúc. DAR loại clip dưới một giây và clip gần như tĩnh, sau đó:

1. Gemini 2.5 Pro đề xuất ranh giới sự kiện thô; PySceneDetect căn ranh giới với điểm cắt trong cửa sổ ±0,5 giây; InternVL3.5 kiểm tra và gộp sự kiện;
2. Qwen3-VL tạo mô tả vi sai giữa đoạn hiện tại và đoạn trước, còn Grounding DINO hỗ trợ grounding đối tượng;
3. Qwen3-VL sinh cặp cảm xúc–lý do từ ngữ cảnh lịch sử và ba cảm xúc VCE có điểm cao nhất;
4. Qwen3-Omni và InternVL3.5 chấm năm chiều: visual grounding, causal logic, viewer centricity, temporal consistency và answer consistency.

### 2.2 Thống kê công bố và kiểm tra độc lập

| Thuộc tính | Train | Test | Toàn bộ |
|---|---:|---:|---:|
| Video | 13.646 | 1.441 | 15.087 |
| Đoạn cảm xúc | 33.195 | 3.713 | 36.908 |
| Số đoạn/video trung bình | 2,433 | 2,577 | — |
| Thời lượng video trung bình (giây) | 14,247 | 14,642 | ≈14,3 theo bài báo |
| Thời lượng đoạn trung bình (giây) | 5,857 | 5,683 | ≈5,8 theo bài báo |
| Số từ lý do trung bình | 118,624 | 120,708 | ≈118,8 theo bài báo |
| Số lớp cảm xúc | 27 | 27 | 27 |

Các con số train/test trong bảng được tính lại trực tiếp từ JSONL công khai. Bài báo cho biết split được phân tầng theo cảm xúc trội, số đoạn, thời lượng và thể loại/bối cảnh, đồng thời giữ test chỉ để đánh giá cuối. Ba chuyên gia kiểm tra toàn bộ test và một mẫu train; tỷ lệ lỗi được báo cáo là 2,4% cho ranh giới lệch trên 0,5 giây, 3,1% bất đồng nhãn, 2,2% lỗi grounding và 1,3% lý giải không viewer-centric.

Nguồn: [annotations DAR-R1](https://huggingface.co/datasets/aiaiaizzy/DAR-R1), [VCE/emodiversity](https://github.com/hendrycks/emodiversity).

## 3. Mô hình và huấn luyện

DAR-R1 dùng Qwen2.5-VL-3B làm backbone. Vision encoder được đóng băng; LLM và aligner được tinh chỉnh. SFT chạy 0,5 epoch với AdamW, learning rate `1e-5`. GRPO chạy một epoch với learning rate `2e-6`; bài báo dùng bốn H100.

Tổng reward là tổ hợp trọng số:

| Thành phần | Trọng số | Vai trò |
|---|---:|---|
| Cấu trúc | 0,10 | JSON và schema hợp lệ |
| Số đoạn | 0,25 | đúng số pha cảm xúc |
| Định vị đoạn | 0,25 | IoU và sai lệch biên |
| Cảm xúc | 0,25 | nhãn đúng với ràng buộc temporal |
| Lý giải | 0,15 | độ dài mục tiêu và chống lặp |

Checkpoint Hugging Face là **checkpoint đầy đủ** BF16, không phải adapter PEFT. Hai shard safetensors có tổng dung lượng khoảng 7,51 GB (dung lượng thập phân). Config ghi kiến trúc `Qwen2_5_VLForConditionalGeneration`, 36 lớp, hidden size 2.048 và `transformers_version` 4.57.1.

Nguồn: [checkpoint DAR-R1](https://huggingface.co/aiaiaizzy/DAR-R1), bài báo §4–5.

## 4. Giao thức đánh giá và kết quả của bài báo

Các metric tự động là:

- **SC-Acc:** tỷ lệ video có số đoạn dự đoán bằng ground truth;
- **mIoU:** IoU thời gian trung bình giữa các đoạn dự đoán và ground truth được ghép theo chỉ số;
- **Emo-Acc:** nhãn cảm xúc trên các cặp đoạn thỏa IoU ≥ 0,5; đây là cách dòng metric theo ngưỡng trong code vận hành và cho thang số phù hợp Table 2;
- **GPT-Score:** GPT-4o chấm 0–5 trên năm chiều lý giải nói trên.

Kết quả DAR-R1 công bố ở Table 2:

| SC-Acc | mIoU | Emo-Acc | VG | CL | VC | TC | AC | GPT Avg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 41,5 | 52,3 | 28,6 | 3,1 | 3,8 | 3,2 | 3,2 | 3,1 | 3,3 |

Đánh giá con người với năm chuyên gia trên 100 video cho DAR-R1 trung bình 4,2/5. Phụ lục báo cáo ROUGE-L 24,8, SBERT 78,6 và mAP ngoài miền trên TSL lần lượt 13,2/11,4/9,4/7,5/6,7 tại ngưỡng IoU 0,10/0,15/0,20/0,25/0,30.

## 5. Tái lập benchmark công khai

### 5.1 Hiện vật và tính toàn vẹn

| Thành phần | Bản cố định / kiểm tra |
|---|---|
| Code | commit `229baf251a48d14bfae417a4f36cc59f6fabaac1` |
| Train JSONL | SHA-256 `a16982f8d13112db8ec504d334f44fb2b8cec1493ce254980893e3834aa87796` |
| Test JSONL | SHA-256 `458e305f0d22ec52fc5e3f7dbce410d6d8be74b774f278d76b5d0993f685c433` |
| Model shard 1 | SHA-256 `ae034b7eb4f890a6df5974c5b6a460eca90871bef7524ae47b8694234520bd10` |
| Model shard 2 | SHA-256 `968ab38cae19d4a6604d56277a855bd384b8549e2797084f5316aa8a7eaf5a95` |
| Video VCE | 61.046 MP4; đủ 1.441/1.441 ID test; không file test rỗng |

Môi trường tái lập: Python 3.10.19, PyTorch 2.8.0 + CUDA 12.8, vLLM 0.11.0, Transformers 4.57.1, qwen-vl-utils 0.0.14, bốn RTX 3090 24 GB. Mỗi GPU chạy một shard độc lập, cùng seed 1234, batch size 8, `max_tokens=4096`, `max_model_len=16384`, temperature 0,1 và top-p 0,9.

### 5.2 Kết quả thực chạy

**Trạng thái:** `[ĐANG CHẠY — chỉ thay bằng số sau xác minh cuối]`

### 5.3 Khoảng cách giữa paper và code phát hành

Đây là điểm cần đọc thận trọng khi so sánh số:

1. `test.py` đặt tên `emotion_accuracy` trong summary cho tỷ lệ nhãn trùng trên mọi cặp đoạn so theo chỉ số, **không áp điều kiện IoU ≥ 0,5**; con số này không phải Emo-Acc trong Table 2.
2. Script in riêng “IOU >= 0.5”: nó giữ các cặp có IoU đạt ngưỡng rồi tính tỷ lệ nhãn đúng. Kết quả sơ bộ có cùng thang với Emo-Acc 28,6% của bài, nên đây là phép tính vận hành dùng để đối chiếu Table 2. Bài báo không viết công thức/mẫu số đủ chi tiết để loại bỏ hoàn toàn sự nhập nhằng này.
3. Repository không có evaluator GPT-4o cho GPT-Score của benchmark. Hai script judge Qwen3-Omni/InternVL3.5 thuộc quy trình xây dựng dữ liệu, không phải bộ chấm GPT-4o cho output benchmark.
4. Transformers hiện dùng image processor nhanh theo mặc định; log cảnh báo checkpoint từng được lưu với slow processor. Điều này có thể tạo sai khác nhỏ so với môi trường tác giả dù phiên bản Transformers khớp config checkpoint.

Vì vậy kết quả tái lập sẽ báo song song: summary metric nguyên trạng của code, Emo-Acc ở dòng `IOU >= 0.5` để đối chiếu Table 2, coverage/parse failure, thêm một biến thể nghiêm ngặt `(label đúng ∧ IoU ≥ 0,5) / toàn bộ cặp` được ghi rõ là phân tích bổ sung, và không bịa GPT-Score khi evaluator chưa được phát hành.

## 6. Hạn chế và rủi ro diễn giải

Các điểm sau là **suy luận phương pháp luận của người phân tích**, không phải toàn bộ đều được tác giả nêu thành mục limitation riêng:

- cảm xúc người xem mang tính chủ quan, phụ thuộc văn hóa, trải nghiệm và ngữ cảnh; một ground truth duy nhất có thể không đại diện đầy đủ phân bố phản ứng;
- taxonomy cố định 27 lớp và việc dùng top-3 VCE làm ứng viên có thể giới hạn cảm xúc ngoài ontology;
- phần lớn pipeline annotation được mô hình lớn sinh/chấm, nên sai lệch hoặc hallucination có thể tương quan giữa teacher và judge;
- GPT-4o-as-judge tạo phụ thuộc vào model/API không cố định theo thời gian, nhưng evaluator và prompt đầy đủ không có trong code phát hành;
- prompt đánh giá bỏ âm thanh, nên benchmark đo suy luận cảm xúc từ thị giác chứ không bao phủ trải nghiệm đa phương thức hoàn chỉnh;
- video gốc giữ điều khoản của nguồn ban đầu; Apache-2.0 của code không tự động cấp lại quyền cho nội dung video.

## 7. Kết luận

DAR là một benchmark có ý nghĩa vì kết hợp temporal localization, nhãn cảm xúc viewer-centric và causal rationale trong cùng đầu ra. DAR-R1 cho thấy RL có cấu trúc reward giúp một backbone 3B cải thiện rõ so với base/SFT theo kết quả bài báo. Tuy nhiên, việc tái lập cần phân biệt metric mô tả trong paper với metric thực thi trong code và không thể tái tạo GPT-Score chỉ từ repository hiện tại.

## Tài liệu tham khảo

1. Zhiyan Zhang, Peipei Song, Jinpeng Hu, Jingyang Jia, Xun Yang, Xiaojun Chang. “Benchmarking Dynamic Affective Reasoning: A Viewer-Centric Video Emotion Dataset.” arXiv:2607.10238v1, 2026. [Abstract](https://arxiv.org/abs/2607.10238) · [PDF](https://arxiv.org/pdf/2607.10238).
2. Zhang et al. DAR official repository, commit `229baf251a48d14bfae417a4f36cc59f6fabaac1`. [GitHub](https://github.com/Zhang-Zhiyan/DAR).
3. aiaiaizzy. DAR-R1 annotations. [Hugging Face Datasets](https://huggingface.co/datasets/aiaiaizzy/DAR-R1).
4. aiaiaizzy. DAR-R1 checkpoint. [Hugging Face Models](https://huggingface.co/aiaiaizzy/DAR-R1).
5. Dan Hendrycks et al. Viewer-Centric Emotion / Emotion Diversity resources. [GitHub](https://github.com/hendrycks/emodiversity).
6. Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner. “Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents.” arXiv:2609.00065v1, 2026. [arXiv](https://arxiv.org/abs/2609.00065v1).
