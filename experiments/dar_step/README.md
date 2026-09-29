# DAR: STEP-inspired graph pilot

Notebook mới: [`dar_kaggle_step_graph_pilot.ipynb`](../../dar_kaggle_step_graph_pilot.ipynb).
Notebook gốc `dar_kaggle_stsg_pilot.ipynb` không bị sửa.

## Thiết kế

Ý tưởng phù hợp để thử **auxiliary graph supervision** cho DAR. Graph có thể cung cấp
ngữ cảnh đối tượng, tương tác và thay đổi theo thời gian; hiệu quả nhận diện cảm xúc
vẫn cần kiểm chứng trên dev. Scene cut không phải emotion boundary.

1. **Tách scene:** PySceneDetect `ContentDetector`, threshold 27, min scene 0.5 s.
   Lưu khoảng frame/time, không cần tạo lại các file video nhỏ. Không có cut thì
   toàn bộ video là một scene. Pipeline này yêu cầu CFR; kiểm tra decoder timestamps
   và độ dài video so với annotation, không kéo giãn thời gian để khớp nhãn.
2. **Keyframe — Hecate source port:** mọi frame được xét, ảnh phân tích resize cạnh
   dài tối đa 160px. Histogram HSV 128 bin/kênh + cạnh 8/8 bin trên 5 vùng = 2000 chiều.
   Global k-means++ dùng K=min(N//2, số scene), có guard tối thiểu 1 và số vector khác nhau.
   Trong mỗi scene, mỗi đoạn nhãn cụm liên tiếp là một subshot; chọn frame có sai khác
   nhỏ nhất với các frame gốc liền kề. Không dùng SigLIP để clustering hoặc thêm endpoints.
   Video một scene có thể chỉ cho 1 frame và không có motion pair. Tối đa 24 ảnh/video;
   vượt budget được ghi lỗi, không bỏ subshot. PNG được lưu ở độ phân giải gốc.
   Xem [nguồn và các điều chỉnh](HECATE_NOTICE.md), [giấy phép](HECATE_LICENSE).
   Đây là port phần keyframe từ source Hecate; PySceneDetect vẫn chia scene, không chạy
   quality filtering/long-shot post-processing hoặc xếp hạng thumbnail toàn video của Hecate.
3. **Semantic parsing:** VILA sinh objects (static/dynamic), attributes, relations
   và observation cho từng frame. Một prompt thực hiện ba bước parsing này.
   Một lần hỏi lại trên ảnh để lọc object/attribute/relation/description không được
   hỗ trợ. Lưu cả ứng viên, verdict, prompt, ảnh hash và output thô.
4. **Dynamic merging:** đối chiếu các keyframe kề nhau trong cùng scene. Chỉ hợp nhất
   identity khi teacher trả về match một-một; không match bằng nhãn đơn thuần.
   Static và dynamic object có entity cục bộ dùng chung qua frame; các lần xuất hiện
   và attributes vẫn riêng theo thời gian. Motion edges nối hai quan sát của cùng
   dynamic entity, mô tả thay đổi ở hai đầu mút, không suy diễn trajectory.
5. **Cross-clip bridging:** một quan sát đại diện cho mỗi entity cục bộ, gom theo
   frame và đối chiếu trên **mọi cặp scene**. Giữ `same_as` giữa entity cục bộ thay
   vì xóa mất nguồn scene. Các liên kết tạo xung đột danh tính bị từ chối. Có thể bỏ
   lỡ match khi ảnh đại diện không đủ rõ; không có tracker/bbox để xác minh độc lập.
   Mỗi scene có mô tả sự kiện ngắn, được hỏi xác minh lại trên tối đa 4 keyframe cách đều.
   Mọi keyframe vẫn được parse và merge; `event_frame_ids` ghi rõ ảnh dùng cho mô tả scene. Nối
   event bằng `before` theo thời gian, không suy diễn quan hệ nhân quả.
6. **Graph toàn video:** `scenes`, `entities`, `frames`, `motion_links`,
   `reference_links`, `event_links`. Timestamps tính từ đầu video gốc. Graph từng
   frame/scene và graph cuối đều được lưu. Schema checks không chứng minh factuality.
7. **Huấn luyện:** hai mẫu/video trên Qwen2.5-VL-3B dùng chung weights:
   - Video gốc (16 frame uniform như DAR pilot) → DAR target gốc.
   - Đúng PNG keyframe của teacher + timestamps + scene intervals → graph toàn video.
   Graph là assistant target, không có trong user input. Đây là auxiliary SFT,
   **chưa có GNN/graph encoder/head riêng**. Inference DAR vẫn video → DAR JSON.

Nhánh graph dùng ảnh riêng vì sampling không đều. Không gán một fps giả cho chuỗi
keyframe. Context SFT mặc định 16384; mọi mẫu được encode qua ms-swift 3.12.5 trước
khi train. Nếu vượt context, dừng để điều chỉnh cấu hình. Việc dùng nhiều frame,
multi-image và context lớn hơn là các yếu tố cần kiểm soát khi so sánh với pilot cũ.

## Chọn teacher

| Mô hình trong STEP | Nhận định cho pipeline này |
|---|---|
| VILA3B → checkpoint công khai VILA1.5-3B | Chọn cho lần thử đầu. Native multi-image phù hợp parse một ảnh, match hai ảnh và tóm tắt vài keyframe; mô hình 3B giảm phần bộ nhớ weights so với 7B. Chất lượng graph trên DAR chưa đo. |
| VideoChat2-Mistral7B | Ứng viên đối chứng sau. STEP có kết quả video reasoning, nhưng không phải benchmark so sánh độ đúng của graph. VideoChat2* trong STEP có bước instruction tuning riêng; không nên coi checkpoint stage3 công khai là checkpoint đó. |

Không có bằng chứng trực tiếp từ STEP rằng một model luôn sinh graph tốt hơn model
còn lại. Chọn cuối cùng nên dựa trên cùng một mẫu train đã cố định: độ đúng entity,
relation/motion, identity xuyên cảnh, valid-JSON rate và thời gian/VRAM. Không chọn
teacher hoặc threshold theo official test. VILA checkpoint ở đây không phải bản
checkpoint đã chạy toàn bộ self-training STEP.

Prompts nằm trong [`prompts.py`](prompts.py), là prompt mới của dự án. Pipeline có
visual verification một lượt, **chưa có n-sample self-consistency**, QRA generation,
CoT supervision hay iterative self-training như STEP đầy đủ.

## Chạy

Trên máy có Internet, chuẩn bị runtime bundle (không tải model weights):

```powershell
python experiments/dar_step/prepare_runtime.py --output experiments/dar_step/runtime-bundle
```

Nếu `runtime-bundle/` đã có `manifest.json` hoàn chỉnh từ lần chuẩn bị trước thì dùng
luôn thư mục đó. Script không ghi đè bundle đang tồn tại.

Attach bundle vào Kaggle, cùng snapshot đầy đủ
`Efficient-Large-Model/VILA1.5-3b` (`config.json`, `llm/`, `vision_tower/`,
`mm_projector/`, tokenizer trong `llm/`). Giữ các inputs DAR/student của notebook gốc.
Sửa `TEACHER_MODEL`, `TEACHER_BUNDLE` trong cell cấu hình rồi chạy `MODE='smoke'`.
Chỉ chuyển `pilot`/`full` sau khi xem graph và chi phí thực tế. Notebook chạy offline.
Không có checkpoint VILA thật trong workspace này, nên chưa chạy teacher GPU/SFT thật.

Runtime teacher dùng native VILA commit
`6b941da19e31ddfdfaa60160908ccf0978d96615`, Transformers 4.36.2 riêng với student
4.57.1. Source bootstrap chỉ chuyển imports của vision tower/S2 không dùng sang
lazy imports; diff + file hashes được lưu ở `vila/dar-runtime.json`. Không cài patch
training Transformers/FlashAttention của VILA. Adapter dùng `vicuna_v1`, FP16, eager
attention, native `model.generate(images=...)`; không dùng AutoModel chung để đoán
giao diện VILA. CUDA Torch của Kaggle được dùng lại; cần smoke xác nhận stack GPU đó.

Kiểm tra CPU (môi trường có NumPy, Pillow, OpenCV, PySceneDetect):

```powershell
python -m unittest discover -s experiments/dar_step -p test_step.py -v
python experiments/dar_step/build_notebook.py
```

Kiểm tra đã thực hiện: **14 test CPU pass**, gồm histogram Hecate, clustering,
subshot theo thời gian, stillness tại biên scene và PySceneDetect trên video tổng hợp
có/không có cut, merge/bridge, kiểm tra timestamp, nguồn ảnh và tạo mẫu SFT không lộ
nhãn DAR. Native VILA đã chạy embedding + generate hai ảnh trên CPU với mô hình
nhỏ/trọng số ngẫu nhiên (`check_native_vila.py`, Torch 2.5.1, Transformers 4.36.2).
Notebook qua nbformat/Python syntax checks; 17 file nhúng khớp source; hash runtime
bundle hợp lệ. Các phép thử này chưa thay thế inference checkpoint 3B hoặc SFT thật.

Artifacts mỗi run:

- `teacher/run.json`: input/config/code/version provenance.
- `teacher/<hash-id>/sampling.json`, `frames/*.png`, `calls/*.json`.
- `intermediate.json`: FSG, kết quả merge/bridge và clip graphs; `graph.json`: graph cuối.
- `evidence.jsonl`: một record/video, gồm cả lỗi; không chọn best retry.
- `arms/build.json`: accepted IDs/rejections; `stsg-lengths.json`: số token thực tế.
- SFT checkpoint và dev DAR predictions/evaluation theo pilot gốc.

Re-run dùng output directory mới. Không có resume teacher tự động. Bridging mọi cặp
scene có thể tăng chi phí đáng kể; bắt đầu 8 train / 4 dev để đo số calls/video.
`stsg_no_links` là ablation cạnh: bỏ motion/reference/event links, vẫn giữ timestamps
và scene intervals. Để so sánh khoa học cần cùng accepted IDs, selected frames,
ngân sách và nhiều seed; chưa có kết luận cải thiện DAR từ phần triển khai này.

## Nguồn

- [STEP §3.1, §3.2, §4.1](https://arxiv.org/html/2412.00161v2).
- [VILA1.5-3B model card](https://huggingface.co/Efficient-Large-Model/VILA1.5-3b).
- [VILA native source đã ghim revision](https://github.com/NVlabs/VILA/tree/6b941da19e31ddfdfaa60160908ccf0978d96615).
- [ms-swift 3.12.5 Qwen template](https://github.com/modelscope/ms-swift/blob/v3.12.5/swift/llm/template/template/qwen.py).
