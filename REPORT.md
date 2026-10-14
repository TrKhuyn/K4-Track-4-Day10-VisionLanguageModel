Họ tên: Trần Ngọc Khuyến
`dataset_version`: `d10-21e4070301`

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | Không ghi nhận | Không ghi nhận | 0.085903 | [0.061364, 0.112798] | 0.919614 | 0.107383 | 0.032000 | 0.177606 |
| E1 | LoRA LLM | | | | | | | | | |
| E2 | + projector | | | | | | | | | |
| E3 | + LoRA ViT | | | | | | | | | |
| E4 | (ghi biến và giá trị) | | | | | | | | | |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
<điền>

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
E0 có `parse_valid_rate = 0.8360128617363344`, tương ứng 260/311 ảnh parse hợp lệ và 51 ảnh không hợp lệ. F1 là 0.085903, precision 0.120000, recall 0.066895; có 39 TP, 286 FP và 544 FN. Kết quả cho thấy hai vấn đề ở mức tổng hợp: chưa bảo đảm định dạng đầu ra và phát hiện/định vị còn kém. Evidence không cho biết riêng nguyên nhân sai lớp hay sai box.

Mã lưu nguyên văn đầu ra vào `predictions/e0_test.jsonl`. File raw chưa được chuyển về repo hiện tại, nên chưa trích ví dụ output thật hoặc khẳng định hai dạng lỗi thô phổ biến nhất như markdown, sai thang hay sai lớp. Output được chấm bằng parser nguyên bản của đề, không tự sửa trước khi tính điểm.

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
Với ảnh 640 × 360, tọa độ `[100,200,300,400]` trên thang 0–1000 tương ứng `[64,72,192,144]` pixel. Nếu scorer hiểu nhầm bốn số đầu là pixel, vùng đánh giá bị dịch và đổi kích thước; IoU có thể xuống dưới 0,5 dù JSON vẫn hợp lệ. F1 sẽ thay đổi tùy các box thật, không thể khẳng định một giá trị cụ thể trước khi chấm.

Hàm `yolo_to_box` nhân tọa độ chuẩn hóa với 1000, làm tròn số nguyên và kẹp biên; scorer được gọi với `--coord norm1000`. Việc này xác nhận tính nhất quán của đường xử lý. Chưa có phép chấm cùng predictions theo hai thang tọa độ của lượt Colab hiện tại; cần bổ sung hai kết quả đó trước khi đưa ra kết luận định lượng.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
<điền>

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
<điền>

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
Các frame gần nhau có thể cùng chứa một ổ gà và gần như cùng bối cảnh. Random split đưa những frame đó vào cả train và test, làm mô hình được đánh giá trên cảnh đã gặp và khiến F1 có thể cao giả. Chia theo session giữ toàn bộ phiên quay trong một split; các phiên được gán tất định bằng thuật toán của đề.

Phân bổ yêu cầu của dataset, cũng được kiểm tra bằng assert trong mã, là:

| Split | Ảnh | Box pothole | Box crack | Box manhole |
|---|---:|---:|---:|---:|
| Train | 1422 | 1074 | 1750 | 691 |
| Val | 276 | 66 | 470 | 103 |
| Test | 311 | 121 | 299 | 163 |

Tổng cộng 2009 ảnh và 4737 box. Mã kiểm tra các tập session không giao nhau. Các số này là đặc điểm split chuẩn của đề, không phải điểm số huấn luyện.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
<điền>

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
<điền>

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
<điền>

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
<điền>

## Nhìn lại

<điền: điều bạn chưa hiểu, lỗi dữ liệu hoặc mô hình bạn gặp mà công cụ chấm chưa bắt được>
