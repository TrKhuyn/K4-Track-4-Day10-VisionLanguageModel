# Chạy Lab Day 10 trên Colab

1. Mở [lab_day10.ipynb từ GitHub trên Colab](https://colab.research.google.com/github/TrKhuyn/K4-Track-4-Day10-VisionLanguageModel/blob/main/lab_day10.ipynb). Cell đầu clone fork cá nhân vào `/content/day10-vlm-repo`; nếu repo đã tồn tại, cập nhật bằng `git pull --ff-only`. Các cell gọi `run_lab.py` trong repo; notebook không nhúng hoặc ghi đè mã Python.
2. Runtime → Change runtime type → T4 GPU. Dùng một runtime mới cho notebook này.
3. Trong Secrets, thêm `KAGGLE_USERNAME`, `KAGGLE_KEY` và bật quyền truy cập notebook. Có thể thêm `HF_TOKEN` để hạn chế lỗi giới hạn tải Hugging Face; mã không in hoặc lưu token. Kaggle cần khóa API dạng username/key tương ứng với hai secret.
4. Chuẩn bị Google Drive còn đủ chỗ cho adapter, Projector và trạng thái optimizer. Nên để trống khoảng 8–10 GB cho một lượt chạy. Runtime Colab cũng cần đủ chỗ tải model gốc.
5. Chọn Runtime → Run all. Chấp thuận mount Drive và truy cập Secrets khi Colab hỏi. Không cần đổi `SETUP` hoặc chạy tay E1–E3. Profile mặc định là batch 2 / accumulation 4; probe E3 gặp OOM sẽ chuyển toàn bộ SFT/E4 sang batch 1 / accumulation 8 trước các lượt chính thức.
6. Notebook kiểm tra công cụ đề và mã sinh viên, xác nhận split, mask loss và một update E3 thử; sau đó chạy E0, E1, E2, E3, E4 và E5. Trọng số probe bị bỏ. Box chiều cao/rộng 0 được biểu diễn bằng tối thiểu 1 đơn vị như hướng dẫn lab, không bỏ annotation. Model gốc chỉ tải một lần, các lần nạp tiếp dùng snapshot local; revision được giữ trên Drive khi nối lại phiên.
7. Khi kết thúc, notebook tự yêu cầu tải `day10_results.zip`. Nếu trình duyệt chặn tải, lấy ZIP trong đường dẫn Drive được in ở cell cuối.

Sau mỗi lời gọi `%run`, notebook kiểm tra bước đó hoàn thành. Nếu có lỗi, Run all dừng tại bước lỗi; xem traceback trước khi chạy tiếp. Khi cập nhật mã, mở notebook mới từ GitHub để nhận cả cell và mã runner tương ứng. Nếu mã thay đổi, run-hash có thể thay đổi; muốn tiếp tục đúng lượt cũ thì giữ nguyên phiên bản mã đã chạy.

E4 chọn đúng biến `data_size`. Setup được chọn bằng micro F1 trên toàn bộ val; nhánh 10% lấy 142 ảnh bằng seed cố định, giữ nguyên mọi hyperparameter SFT. Nhánh 100% tái dùng lượt SFT 1.422 ảnh tương ứng, rồi chấm lại chính raw predictions đó với tên E4. GUIDE cho phép một nhánh là setup gốc.

E5 dùng `trl==1.9.2`, `GRPOTrainer`, 4 completion/ảnh, tối đa 128 token/completion, 60 bước, LR `5e-6`, beta 0, không vLLM. LoRA được khởi tạo từ setup SFT tốt nhất; Projector đã học được nạp đầy đủ và đóng băng trong GRPO. Bốn completion được sinh tuần tự để giảm VRAM, nhưng vẫn tính advantage theo nhóm bốn.

Lượt chạy đầy đủ có bốn lượt SFT × 300 update (E1–E3, E4 10%), GRPO và suy luận nhiều split. Thời gian phụ thuộc T4, độ dài đáp án và tốc độ Drive; không đảm bảo Colab miễn phí hoàn thành trong một phiên. Checkpoint SFT lưu mỗi 100 bước, GRPO mỗi 20 bước vào `MyDrive/day10-vlm/<dataset_version>/<run-hash>/`. Nếu ngắt, mở lại đúng notebook, cùng tài khoản Drive, Run all: SFT/GRPO tiếp tục từ checkpoint hoàn chỉnh và suy luận tiếp các ID còn thiếu. Giai đoạn probe ngắn chạy lại nếu chưa có marker hoàn thành.

## Sau khi có kết quả thật

- [ ] Giải nén ZIP vào repo cá nhân để nhận `report/evidence/e0.json`–`e3.json`, hai file E4, `e5.json`, sáu ảnh và caption.
- [ ] Đối chiếu evidence với `analysis/summary.json`, loss/resource log, `e4_analysis.json`, `e5_analysis.json` và output thô.
- [ ] Điền thông tin cá nhân, bảng kết quả và từng câu hỏi của `REPORT.md` bằng số liệu này. Báo cáo hiện giữ nguyên mẫu.
- [ ] Commit theo từng chặng với bằng chứng thực: dữ liệu/split, E0, mask/setup, E1–E3, E4, E5/báo cáo. Chỉ stage file liên quan; không tạo commit giả để đạt số lượng.
- [ ] Giữ `scripts/`, `tests/`, `configs/`, GUIDE và requirements của đề nguyên bản. Không commit data, work, API keys hoặc trọng số.
- [ ] Chạy `python scripts/check_submission.py`; xử lý mọi FAIL trước khi nộp. Điểm E1 chưa đạt ngưỡng thì ghi nhận đúng kết quả, không sửa evidence.
- [ ] Push lên fork Public và nộp link repo cá nhân.
