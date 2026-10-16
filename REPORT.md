Họ tên: Trần Ngọc Khuyến
`dataset_version`: `d10-21e4070301`

Trạng thái: đã có evidence E0–E3 mới, chấm trên 311 ảnh test, tọa độ `norm1000`, không thiếu predictions và cùng `dataset_version`. E4/E5 chưa có evidence trong repo hiện tại. Bảng lấy số trực tiếp từ `report/evidence/e0.json`–`e3.json`; số thập phân được làm tròn đến 6 chữ số, file evidence được giữ nguyên.

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút (train; E0 infer) | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | 3.774805 | 25.044067 (infer) | 0.074505 | [0.049946, 0.101655] | 0.784566 | 0.136646 | 0.000000 | 0.150538 |
| E1 | LoRA LLM | 33030144 | 5.263924 | 60.893050 | 0.284900 | [0.246101, 0.325810] | 1.028939 | 0.154762 | 0.219048 | 0.533333 |
| E2 | + projector | 142135296 | 6.926299 | 61.454139 | 0.339236 | [0.300930, 0.379592] | 0.990354 | 0.189474 | 0.266442 | 0.586207 |
| E3 | + LoRA ViT | 148426752 | 7.060171 | 86.043607 | 0.446655 | [0.404573, 0.487761] | 0.887460 | 0.324873 | 0.362438 | 0.695364 |
| E4 | data_size = 0.1 | Chưa có kết quả | Chưa có kết quả | Chưa có kết quả | Chưa có kết quả | Chưa có kết quả | Chưa có kết quả | Chưa có kết quả | Chưa có kết quả | Chưa có kết quả |
| E4 | data_size = 1.0 | Chờ chọn baseline trên val | — | — | — | — | — | — | — | — |

E0 ghi nhận VRAM đỉnh 3.774805 GB và 25.044067 phút inference; `train_minutes = 0` vì đây là zero-shot. `parse_valid_rate` lần lượt là 0.520900, 0.987138, 1.000000 và 1.000000. E1 tăng F1 so với E0 là 0.210395, vượt yêu cầu tăng ít nhất 0.10; tỷ lệ parse hợp lệ của E1 cũng vượt ngưỡng 0.95.

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
ViT trích xuất đặc trưng ảnh; projector gồm `visual.merger` và `visual.deepstack_merger_list` chuyển và hợp nhất đặc trưng để LLM nhận đầu vào thị giác. LLM sinh chuỗi JSON theo cơ chế autoregressive. Theo quy mô tham chiếu trong GUIDE, LLM cùng embedding khoảng 4,0 tỷ tham số, khối thị giác khoảng 415 triệu; merger khoảng 27 triệu và các DeepStack merger khoảng 82 triệu nằm trong khối thị giác, không cộng lại hai lần. Đây là số tham chiếu, chưa phải phép đếm từ lượt Colab hiện tại.

E1 chỉ học LoRA trên LLM, giữ ViT và projector cố định: 33.030.144 tham số học. E2 giữ LoRA LLM và học toàn bộ hai nhóm projector: 142.135.296 tham số, tăng đúng 109.105.152 so với E1. E3 bổ sung LoRA vào attention và MLP của ViT: 148.426.752 tham số, tăng 6.291.456 so với E2. Các số này lấy từ `meta.trainable_params`. Mã dùng rank 16, alpha 32, dropout 0,05; có 252 module mục tiêu LLM và thêm 96 module ViT ở E3.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
E0 có `parse_valid_rate = 0.5209003215`, tương ứng 162/311 ảnh parse hợp lệ và 149 ảnh không hợp lệ. F1 là 0.074505, precision 0.115942, recall 0.054889; có 32 TP, 244 FP và 551 FN. Kết quả cho thấy hai vấn đề ở mức tổng hợp: định dạng đầu ra chưa ổn định và khả năng phát hiện/định vị còn thấp. Evidence không cho biết riêng nguyên nhân sai lớp hay sai box. E0 ghi nhận VRAM đỉnh 3.774805 GB và inference 25.044067 phút.

Mã lưu nguyên văn đầu ra vào `predictions/e0_test.jsonl`. File raw chưa được chuyển về repo hiện tại, nên chưa trích ví dụ output thật hoặc khẳng định hai dạng lỗi thô phổ biến nhất như markdown, sai thang hay sai lớp. Output được chấm bằng parser nguyên bản của đề, không tự sửa trước khi tính điểm.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
Với ảnh 640 × 360, tọa độ `[100,200,300,400]` trên thang 0–1000 tương ứng `[64,72,192,144]` pixel. Nếu scorer hiểu nhầm bốn số đầu là pixel, vùng đánh giá bị dịch và đổi kích thước; IoU có thể xuống dưới 0,5 dù JSON vẫn hợp lệ. F1 sẽ thay đổi tùy các box thật, không thể khẳng định một giá trị cụ thể trước khi chấm.

Hàm `yolo_to_box` nhân tọa độ chuẩn hóa với 1000, làm tròn số nguyên và kẹp biên; scorer được gọi với `--coord norm1000`. Việc này xác nhận tính nhất quán của đường xử lý. Chưa có phép chấm cùng predictions theo hai thang tọa độ của lượt Colab hiện tại; cần bổ sung hai kết quả đó trước khi đưa ra kết luận định lượng.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
Mục tiêu SFT là dự đoán JSON của assistant khi đã biết ảnh và câu hỏi. `encode_example` mã hóa riêng prompt và hội thoại đầy đủ, kiểm tra prompt là tiền tố chính xác, rồi đặt labels của toàn bộ tiền tố thành `-100`. Nhãn còn được học là JSON và token kết thúc `<|im_end|>`; padding cũng có label `-100`. Các tensor ảnh và grid được nối theo chiều 0, không đệm như token văn bản.

Nếu tính loss trên toàn chuỗi, các token prompt và placeholder thị giác cũng bị coi là mục tiêu dự đoán, dù chúng đã được cung cấp làm điều kiện. Điều đó thêm một mục tiêu không cần thiết và làm lệch trọng số giữa phần điều kiện dài với câu trả lời. Token ảnh trong chuỗi không phải mục tiêu tái tạo pixel. Mã kiểm tra số token ảnh 220 và giải mã riêng nhãn được học; kết quả kiểm tra của lượt Colab cần đối chiếu với `mask_check.json`.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
Trọng số projector ở dạng NF4 được đóng gói và không phải các tham số float mà AdamW có thể cập nhật đầy đủ như E2/E3 yêu cầu. LoRA vẫn có thể thích ứng một lớp lượng tử hóa, nhưng điều đó không thay thế việc học toàn bộ projector theo thiết kế thí nghiệm.

Mã skip `visual`, tên đầy đủ `model.visual` và `lm_head` khi tải 4-bit, đồng thời kiểm tra khối thị giác không chứa Linear4bit. Projector của E2/E3 được chuyển sang float32 để giữ độ chính xác cập nhật và giảm nguy cơ tràn gradient; phần forward vẫn dùng autocast fp16 phù hợp T4. Projector được lưu riêng vào `projector.pt` và nạp lại khi đánh giá, vì adapter PEFT không tự bao gồm các trọng số projector học đầy đủ này.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Các frame gần nhau có thể cùng chứa một ổ gà và gần như cùng bối cảnh. Random split đưa những frame đó vào cả train và test, làm mô hình được đánh giá trên cảnh đã gặp và khiến F1 có thể cao giả. Chia theo session giữ toàn bộ phiên quay trong một split; các phiên được gán tất định bằng thuật toán của đề.

Phân bổ yêu cầu của dataset, cũng được kiểm tra bằng assert trong mã, là:

| Split | Ảnh | Box pothole | Box crack | Box manhole |
|---|---:|---:|---:|---:|
| Train | 1422 | 1074 | 1750 | 691 |
| Val | 276 | 66 | 470 | 103 |
| Test | 311 | 121 | 299 | 163 |

Tổng cộng 2009 ảnh và 4737 box. Mã kiểm tra các tập session không giao nhau. Các số này là đặc điểm split chuẩn của đề, không phải điểm số huấn luyện. Bốn evidence đều ghi `d10-21e4070301`, 311 ảnh test; TP + FN ở mỗi lớp lần lượt khớp 121 pothole, 299 crack và 163 manhole.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
E1 đạt F1 0.284900 với CI95 [0.246101, 0.325810], 5.263924 GB và 60.893050 phút. E2 đạt 0.339236, tăng 0.054336; CI95 [0.300930, 0.379592] giao với E1. Mở projector tăng VRAM ghi nhận 1.662375 GB và thêm 109.105.152 tham số học; thời gian E2 cao hơn E1 0.561089 phút. Đây là số đo của từng lượt chạy, chưa đủ để kết luận chênh lệch thời gian ổn định.

E3 đạt F1 0.446655, tăng 0.107419 so với E2; CI95 [0.404573, 0.487761] không giao với E2. E3 có 148.426.752 tham số học, 7.060171 GB và 86.043607 phút: VRAM cao hơn E2 0.133872 GB, thời gian cao hơn 24.589468 phút. Đây là số đo của từng lượt chạy, không đủ để giải thích nguyên nhân chênh lệch. E3 có F1 cao nhất trong E0–E3; FP/ảnh thấp nhất là E0 (0.784566), còn E3 thấp nhất trong ba setup đã fine-tune. Kết quả ủng hộ khả năng thích ứng thêm ViT trong lượt chạy này; lựa chọn baseline E4/E5 vẫn cần dựa trên val.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
Scorer bootstrap theo ảnh trên 311 ảnh test và trả `f1_ci95`. Độ rộng CI của E0 là 0.051709; E1 là 0.079709; E2 là 0.078662; E3 là 0.083187. Đây là độ rộng toàn khoảng, không phải độ lệch một phía ±. Với E3, CI [0.404573, 0.487761] lệch khoảng -0.042081 và +0.041106 so với F1 0.446655. CI của E1 và E2 giao nhau; CI của E2 và E3 không giao nhau. So sánh CI riêng lẻ không thay thế kiểm định bootstrap ghép cặp cho chênh lệch.

CI rộng làm chênh lệch F1 nhỏ khó được diễn giải chắc chắn. Hai CI giao nhau không tự chứng minh hai setup tương đương; hai CI riêng lẻ cũng không phải CI cho chênh lệch giữa setup. Vì cùng đánh giá trên một tập ảnh, bootstrap ghép cặp cho chênh lệch sẽ phù hợp hơn khi cần kiểm định thêm. Frame cùng session còn tương quan, nên bootstrap theo ảnh có thể đánh giá thấp bất định so với lấy mẫu theo session.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
Ở E3, pothole kém nhất: F1 0.324873, thấp hơn crack 0.362438 và manhole 0.695364. Pothole có 32 TP, 44 FP, 89 FN; recall 0.264463 thấp hơn precision 0.421053, cho thấy nhiều vật thể thật chưa được khớp. Crack có 110 TP, 198 FP, 189 FN; còn cả dự đoán dư lẫn bỏ sót đáng kể. Manhole có 105 TP, 34 FP, 58 FN và là lớp tốt nhất.

Pothole có F1 thấp nhất ở cả E1 (0.154762), E2 (0.189474) và E3 (0.324873); crack E1 đạt 0.219048. Pothole có 121 GT trên test và 66 GT trên val, ít hơn crack; box nhỏ hoặc nhầm với bóng/mảng đường là các giả thuyết cần kiểm tra bằng ảnh. Crack mảnh có thể mất IoU nhanh khi box lệch, nhưng evidence hiện tại chưa có thống kê kích thước hoặc hình để xác nhận nguyên nhân. Mã export dự kiến tạo sáu ảnh có GT màu xanh và dự đoán màu đỏ cùng caption; hiện chưa đính kèm các hình đó.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
E4 được triển khai với đúng biến `data_size`: 142 ảnh train so với 1422 ảnh. Setup nền được chọn trên val sau E1–E3; nhánh 10% giữ nguyên rank, alpha, LR, số bước, seed, batch, prompt, mask và thang tọa độ. Nhánh 100% tái dùng predictions của baseline đầy đủ, đúng ngoại lệ được GUIDE cho phép.

E4 chưa có kết quả được xác nhận, nên chưa kết luận tác động của giảm dữ liệu hoặc mức giao nhau CI95. Mã sẽ xuất `e4_data_0.1.json`, `e4_data_1.0.json` và phân tích CI; chỉ bổ sung kết luận khi các file này được tạo từ lượt chạy thật.

## Nhìn lại

Đã có evidence hoàn chỉnh mới của E0–E3; E4/E5 và bộ ảnh minh họa chưa được chuyển về repo tại thời điểm cập nhật. E1 đạt hai ngưỡng chính của rubric: parse hợp lệ 0.987138 (≥0.95) và tăng F1 so với E0 là 0.210395 (≥0.10). Báo cáo chưa kết luận về E4/GRPO hoặc tự bổ sung số đo còn thiếu.

Một vấn đề đã được xử lý trong mã là box có chiều rộng hoặc chiều cao bằng 0 sau làm tròn. Hàm chuyển đổi giữ annotation, biểu diễn bằng tối thiểu một đơn vị trên thang 0–1000 và xử lý riêng biên 1000; không sửa dữ liệu hay GT của scorer. Điều chỉnh này giúp nhãn văn bản hợp lệ nhưng không bảo đảm IoU tốt cho box cực mảnh, nên vẫn cần xem ảnh và phân tích lỗi.

E5 là phần thưởng tùy chọn, chưa có kết quả GRPO được xác nhận. Mã reward đã có kiểm thử cho JSON hỏng, dự đoán hoàn hảo, sai lớp, thừa box và cả hai rỗng. Cấu hình dự kiến là GRPO 4 completion/ảnh, 60 bước, LR `5e-6`, khởi tạo từ SFT tốt nhất trên val. Chưa có log reward, F1 hoặc CI95 để so sánh GRPO với SFT hay đánh giá reward hacking.
