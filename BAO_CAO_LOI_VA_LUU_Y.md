# Các điểm kỹ thuật đã xử lý

Bản notebook đã nhận chỉ có E4/E5 trống và chọn tay E3. Bản mới chạy tuần tự E0–E5, lưu artifact lên Drive và đóng gói kết quả thực; `REPORT.md` chờ viết sau khi chạy. Markdown gốc của notebook giữ nguyên.

| Điểm | Xử lý |
|---|---|
| Notebook nhúng mã Python dài và có thể ghi đè bản đã sửa | Notebook clone/pull fork cá nhân rồi gọi các stage của `run_lab.py`. Mã nằm trong repo, không còn `STUDENT_FILES` hoặc cell ghi `lab_runtime.py`. |
| `%run` in traceback nhưng không truyền lỗi ra cell | Cờ hoàn thành được đặt lại trước mỗi stage; cell kiểm tra cờ sau `%run`, dừng Run all nếu stage thất bại. Kiểm thử bằng IPython thật xác nhận giữ trạng thái giữa stage và không đánh dấu thành công khi lỗi. |
| `config/` trong notebook không tồn tại | Đọc các file nguyên bản trong `configs/`. |
| Hai cách sinh nhãn khác nhau; sort x1 trước y1 | Dùng thống nhất JSON list có `label`, `bbox_2d`; sort y1 rồi x1 theo GUIDE. Các lượt SFT/E4 của bản Run all dùng cùng định dạng. |
| Nhãn thật có chiều cao 0 | Dataset `d10-21e4070301` có 1 box crack ở ảnh train `vlcsnap-00058`, YOLO `[0.639062,0.786111,0.003125,0.0]`. Sau làm tròn là `[637,786,641,786]`. Giữ annotation, biểu diễn tối thiểu 1 đơn vị thành `[637,786,641,787]`; xử lý cả biên 1000 bằng dời cạnh dưới về 999. Không sửa nhãn/split/GT của scorer. `label_checks.json` ghi số box và các điều chỉnh. |
| Prompt và token ảnh bị học nhầm | Kiểm tra prefix thật của chat template, mask bằng -100; giữ JSON và `im_end`, bỏ newline sau EOS. `mm_token_type_ids` được giữ; pixel/grid nối, không pad. |
| ViT vô tình bị quantize | Skip cả `visual`, tên đầy đủ `model.visual`, và `lm_head`. Kiểm tra ViT/Projector không có Linear4bit. Chỉ LLM được NF4, các khối được skip có trọng số fp16. |
| TorchAO cũ có sẵn trong Colab | Setup gỡ backend TorchAO không dùng, giữ requirements đề. PEFT không còn kiểm tra phiên bản backend cũ khi dispatch LoRA. |
| HTTP 429 khi đọc cây repo HF | Tải file riêng qua `hf_hub_download`, lấy tên shard từ weight index; retry hữu hạn khi 429/5xx. Dùng snapshot local cho model/processor sau đó. HF_TOKEN lấy từ Secrets. |
| T4 không phù hợp bf16 | SFT dùng autocast fp16 và GradScaler, tham số LoRA/Projector cần học ở fp32. TRL 1.9.2 tự cast QLoRA trainables sang bf16; E5 chuyển chúng lại fp32 ngay sau tạo trainer, trước optimizer. |
| Phép tính xác suất GRPO ở fp16 | Logits cho log-probability/entropy được chuyển fp32 qua forward hook, giữ trọng số lm_head ở fp16. Kiểm tra NF4/fp16 trên CPU đã tái hiện gradient NaN khi không có bước này; sau xử lý, gradient và entropy hữu hạn. Hook dừng khi model xuất logits không hữu hạn để tránh lưu policy hỏng. |
| Grad overflow vẫn tăng step | Chỉ tăng step/scheduler sau update hữu hạn. Norm không hữu hạn thì bỏ update trước AdamW, giảm scale; dừng nếu 20 lượt liên tiếp lỗi. |
| E2/E3 bị mất Projector sau reload | Lưu đủ merger và DeepStack vào `projector.pt`, kiểm tra tên và shape khi nạp. GRPO nạp và giữ Projector SFT đó, chỉ tối ưu LoRA. |
| E4 thay nhiều biến hoặc dùng test chọn baseline | Chọn baseline theo val, đổi đúng phần dữ liệu train 142/1.422 ảnh. Giữ rank/alpha/LR/steps/seed/batch/mask/prompt/tọa độ. Nhánh full tái dùng baseline theo GUIDE. |
| GRPO batch 1 không đủ nhóm | Generation batch 4, accumulation 4, microbatch 1. Sinh tuần tự từng ảnh, ghép lại nhóm 4 đúng thứ tự để tính reward/advantage. Không giảm `num_generations`. |
| Reward sai mẫu số hoặc ghép trùng GT | Dùng parser/IoU nguyên bản của đề; greedy một-một cùng lớp, ngưỡng 0.5, mẫu số max(n_gt,n_pred,1). Test nằm ở `student_checks/`, không sửa `tests/`. |
| Mất phiên Colab | Checkpoint có marker hoàn thành; resume optimizer/scheduler/scaler/RNG của SFT. GRPO dùng resume của Trainer. Manifest tách dataset/model/config/mã/phiên bản; prediction tiếp tục từ ID đã ghi. |
| Ngắt sau bước GRPO cuối nhưng trước export | Nếu checkpoint cuối đã hoàn chỉnh, khôi phục artifact cuối trực tiếp, không gọi Trainer để chạy thêm bước. Đã thử ngắt GRPO ở bước 1, resume bước 2 và khôi phục export sau checkpoint cuối. |
| Chỉ nhìn reward tăng | `e5_analysis.json` so sánh F1/CI95, đếm output rỗng trên ảnh có GT, JSON hỏng và box diện tích ≥80% ảnh. Đây là tín hiệu quan sát, không chứng minh không có hacking. |
| CI95 bị diễn giải quá mức | In giao nhau của hai CI E4; giao nhau không phải kiểm định chênh lệch F1. CI từ scorer bootstrap theo ảnh có giới hạn khi frame cùng session tương quan. |
| Kết quả không tải về | Backup ZIP vào Drive rồi gọi `google.colab.files.download`. ZIP chứa evidence/hình/log/prediction, không chứa dataset hoặc trọng số. |

Không có evidence hoặc kết quả 4B/T4 được tạo giả trong repo. Kiểm thử CPU bằng model nhỏ xác minh logic và API; VRAM, tốc độ, khả năng hội tụ và F1 cần lượt chạy thực trên T4. Thư viện nguồn đối chiếu: [TRL v1.9.2 GRPOTrainer](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_trainer.py), [GRPOConfig](https://github.com/huggingface/trl/blob/v1.9.2/trl/trainer/grpo_config.py).

Kiểm tra cục bộ: **100 passed, 1 skipped** (68 test gốc và 32 test sinh viên). Đã tải dataset thật, xác nhận 2.009 ảnh/4.737 box và split đúng; chạy constructor, kiểm tra JPEG và processor trên tất cả ảnh, xác nhận 220 token và decode JSON + EOS. Model Qwen3-VL nhỏ với 36 lớp LLM, 24 block ViT xác nhận target 252/252/348, forward/backward E1–E3, phạm vi tham số học và logits khớp tuyệt đối sau save/reload adapter + Projector. Kiểm tra thêm NF4 thật + fp16 + tham số học fp32 trên ảnh có nhãn suy biến và ảnh train có đáp án dài nhất, gồm lm_head/embedding dùng chung như model thật. TRL 1.9.2 chạy GRPO đa phương thức trên CPU; nhóm reward khác nhau tạo gradient khác 0 và thay đổi trọng số. Resume SFT giữa hai update cho trọng số/loss khớp tuyệt đối; kiểm tra GRPO resume và phục hồi export. Pipeline rút gọn trong thư mục tạm kiểm tra E4 một biến, Projector E5, 7 evidence do scorer gốc tạo, 6 ảnh và ZIP không chứa trọng số. Các số của phép thử dữ liệu giả không được dùng trong hồ sơ nộp.
