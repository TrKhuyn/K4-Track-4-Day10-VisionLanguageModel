Họ tên: Trần Ngọc Khuyến
`dataset_version`: `d10-21e4070301`

Trạng thái: đã có evidence E0–E3, chấm trên 311 ảnh test, tọa độ `norm1000`, không thiếu predictions và cùng `dataset_version`. E4/E5 chưa có evidence trong repo hiện tại. Số liệu được lấy nguyên từ `report/evidence/e0.json`–`e3.json`; các số thập phân trong bảng được làm tròn đến 6 chữ số, không sửa file evidence.

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | Không ghi nhận | Không ghi nhận | 0.085903 | [0.061364, 0.112798] | 0.919614 | 0.107383 | 0.032000 | 0.177606 |
| E1 | LoRA LLM | 33030144 | 6.343995 | 70.578394 | 0.252874 | [0.219106, 0.292683] | 1.302251 | 0.173077 | 0.166667 | 0.532319 |
| E2 | + projector | 142135296 | 8.039420 | 62.220783 | 0.350442 | [0.311864, 0.391621] | 1.122186 | 0.214876 | 0.279534 | 0.613240 |
| E3 | + LoRA ViT | 148426752 | 7.062068 | 67.329259 | 0.453039 | [0.409434, 0.495928] | 0.826367 | 0.319149 | 0.369128 | 0.701987 |
| E4 | data_size = 0.1 | Chưa có kết quả | Chưa có kết quả | Chưa có kết quả | Chưa có kết quả | Chưa có kết quả | Chưa có kết quả | Chưa có kết quả | Chưa có kết quả | Chưa có kết quả |
| E4 | data_size = 1.0 | Chờ chọn baseline trên val | — | — | — | — | — | — | — | — |

`meta` của E0 rỗng nên không suy ra VRAM hoặc thời gian. `parse_valid_rate` lần lượt là 0.836013, 0.993569, 1.000000 và 1.000000. E1 tăng F1 so với E0 đúng 0.16697047951795024, vượt yêu cầu tăng ít nhất 0.10; tỷ lệ parse hợp lệ của E1 cũng vượt ngưỡng 0.95.

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
ViT trích xuất đặc trưng ảnh; projector gồm `visual.merger` và `visual.deepstack_merger_list` chuyển và hợp nhất đặc trưng để LLM nhận đầu vào thị giác. LLM sinh chuỗi JSON theo cơ chế autoregressive. Theo quy mô tham chiếu trong GUIDE, LLM cùng embedding khoảng 4,0 tỷ tham số, khối thị giác khoảng 415 triệu; merger khoảng 27 triệu và các DeepStack merger khoảng 82 triệu nằm trong khối thị giác, không cộng lại hai lần. Đây là số tham chiếu, chưa phải phép đếm từ lượt Colab hiện tại.

E1 chỉ học LoRA trên LLM, giữ ViT và projector cố định: 33.030.144 tham số học. E2 giữ LoRA LLM và học toàn bộ hai nhóm projector: 142.135.296 tham số, tăng đúng 109.105.152 so với E1. E3 bổ sung LoRA vào attention và MLP của ViT: 148.426.752 tham số, tăng 6.291.456 so với E2. Các số này lấy từ `meta.trainable_params`. Mã dùng rank 16, alpha 32, dropout 0,05; có 252 module mục tiêu LLM và thêm 96 module ViT ở E3.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
E0 có `parse_valid_rate = 0.8360128617363344`, tương ứng 260/311 ảnh parse hợp lệ và 51 ảnh không hợp lệ. F1 là 0.085903, precision 0.120000, recall 0.066895; có 39 TP, 286 FP và 544 FN. Kết quả cho thấy hai vấn đề ở mức tổng hợp: chưa bảo đảm định dạng đầu ra và phát hiện/định vị còn kém. Evidence không cho biết riêng nguyên nhân sai lớp hay sai box.

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
E1 đạt F1 0.252874 với CI95 [0.219106, 0.292683], 6.343995 GB và 70.578394 phút. E2 đạt 0.350442, tăng 0.097569; CI95 [0.311864, 0.391621] không giao với E1. Mở projector tăng VRAM ghi nhận 1.695425 GB, thêm 109.105.152 tham số học; thời gian của lượt E2 lại thấp hơn E1 8.357611 phút. Vì chưa có log giải thích mức chênh thời gian, không quy kết E2 luôn nhanh hơn chỉ từ một lượt chạy.

E3 đạt F1 0.453039, tăng 0.102596 so với E2; CI95 [0.409434, 0.495928] cũng không giao với E2. E3 có 148.426.752 tham số học, 7.062068 GB và 67.329259 phút: VRAM báo cáo thấp hơn E2 0.977352 GB, thời gian cao hơn 5.108476 phút. Số đo này cần giữ nguyên và đối chiếu log/profile trước khi giải thích; không giả định VRAM phải tăng theo số tham số. Trong các kết quả test hiện có, E3 có F1 cao nhất và FP/ảnh thấp nhất. Mức cải thiện ủng hộ việc thích ứng cả projector và ViT trong lượt chạy này; chọn baseline E4/E5 vẫn phải dựa trên val, chưa có file chọn baseline trong repo hiện tại.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
Scorer bootstrap theo ảnh trên 311 ảnh test và trả `f1_ci95`. Độ rộng CI của E0 là 0.051435; E1 là 0.073577; E2 là 0.079757; E3 là 0.086494. Đây là độ rộng toàn khoảng, không phải độ lệch một phía ±. Với E3, CI [0.409434, 0.495928] lệch khoảng -0.043605 và +0.042889 so với F1 0.453039.

CI rộng làm chênh lệch F1 nhỏ khó được diễn giải chắc chắn. Hai CI giao nhau không tự chứng minh hai setup tương đương; hai CI riêng lẻ cũng không phải CI cho chênh lệch giữa setup. Vì cùng đánh giá trên một tập ảnh, bootstrap ghép cặp cho chênh lệch sẽ phù hợp hơn khi cần kiểm định thêm. Frame cùng session còn tương quan, nên bootstrap theo ảnh có thể đánh giá thấp bất định so với lấy mẫu theo session.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
Ở E3, pothole kém nhất: F1 0.319149, thấp hơn crack 0.369128 và manhole 0.701987. Pothole có 30 TP, 37 FP, 91 FN; recall 0.247934 thấp hơn precision 0.447761, cho thấy nhiều vật thể thật chưa được khớp. Crack có 110 TP, 187 FP, 189 FN; còn cả dự đoán dư lẫn bỏ sót đáng kể. Manhole có 106 TP, 33 FP, 57 FN và là lớp tốt nhất.

Ở E1, crack có F1 thấp nhất 0.166667; sang E2/E3, pothole trở thành lớp thấp nhất. Pothole chỉ có 121 GT trên test và 66 GT trên val, ít hơn crack; đặc điểm box nhỏ hoặc nhầm với bóng/mảng đường là các giả thuyết cần kiểm tra bằng ảnh. Crack mảnh có thể mất IoU nhanh khi box lệch, nhưng chưa có thống kê kích thước hay hình để xác nhận nguyên nhân. Mã export dự kiến tạo sáu ảnh có GT màu xanh và dự đoán màu đỏ cùng caption; hiện chưa đính kèm các hình đó.

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
E4 được triển khai với đúng biến `data_size`: 142 ảnh train so với 1422 ảnh. Setup nền được chọn trên val sau E1–E3; nhánh 10% giữ nguyên rank, alpha, LR, số bước, seed, batch, prompt, mask và thang tọa độ. Nhánh 100% tái dùng predictions của baseline đầy đủ, đúng ngoại lệ được GUIDE cho phép.

E4 chưa có kết quả được xác nhận, nên chưa kết luận tác động của giảm dữ liệu hoặc mức giao nhau CI95. Mã sẽ xuất `e4_data_0.1.json`, `e4_data_1.0.json` và phân tích CI; chỉ bổ sung kết luận khi các file này được tạo từ lượt chạy thật.

## Nhìn lại

Đã có evidence hoàn chỉnh của E0–E3; E4/E5 và bộ ảnh minh họa chưa được chuyển về repo tại thời điểm cập nhật. E1 đạt hai ngưỡng chính của rubric: parse hợp lệ ≥0.95 và tăng F1 so với E0 ≥0.10. Báo cáo chưa kết luận về E4/GRPO hoặc tự bổ sung số đo còn thiếu.

Một vấn đề đã được xử lý trong mã là box có chiều rộng hoặc chiều cao bằng 0 sau làm tròn. Hàm chuyển đổi giữ annotation, biểu diễn bằng tối thiểu một đơn vị trên thang 0–1000 và xử lý riêng biên 1000; không sửa dữ liệu hay GT của scorer. Điều chỉnh này giúp nhãn văn bản hợp lệ nhưng không bảo đảm IoU tốt cho box cực mảnh, nên vẫn cần xem ảnh và phân tích lỗi.

E5 là phần thưởng tùy chọn, chưa có kết quả GRPO được xác nhận. Mã reward đã có kiểm thử cho JSON hỏng, dự đoán hoàn hảo, sai lớp, thừa box và cả hai rỗng. Cấu hình dự kiến là GRPO 4 completion/ảnh, 60 bước, LR `5e-6`, khởi tạo từ SFT tốt nhất trên val. Chưa có log reward, F1 hoặc CI95 để so sánh GRPO với SFT hay đánh giá reward hacking.
