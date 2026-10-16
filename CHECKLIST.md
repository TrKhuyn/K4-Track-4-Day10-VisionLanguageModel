# Chạy Lab Day 10 trên Colab

Hướng dẫn khớp với notebook hiện tại: [COLAB_SETUP.md](COLAB_SETUP.md).

1. Upload bản gọn `work/lab_day10_colab.ipynb` vào Colab và **Save a copy in Drive**. Code giống notebook gốc, bỏ output/widget cũ.
2. Chọn runtime mới có **T4 GPU**.
3. Bật Colab Secrets: `KAGGLE_API_TOKEN` hoặc `KAGGLE_USERNAME` + `KAGGLE_KEY`; `HF_TOKEN` tùy chọn.
4. Upload `work/colab_source.zip` vào gốc My Drive để dùng đúng source local. Nếu không có ZIP, notebook clone fork cá nhân được cấu hình trong cell đầu.
5. Chọn **Run all**, xác nhận mount Drive. Luồng chính chạy E0–E4. E5 chỉ có kiểm tra reward.
6. Chờ probe GPU báo PASS trước E0. Kiểm tra checkpoint, `backup_status.json`, evidence và `report_bundle.zip` trong `MyDrive/lab_day10_colab_runs/run_all_v3/`.

Nếu ngắt phiên, mở lại cùng notebook/mã/cấu hình/RUN_LABEL rồi Run all: train tiếp tục checkpoint hợp lệ gần nhất, inference tiếp tục ID đã lưu. Checkpoint mặc định mỗi 25 bước hoặc khoảng 5 phút; log và inference được backup khoảng mỗi phút tại ranh giới update/batch. Không bảo đảm hoàn tất toàn bộ trong một phiên Colab.

`lab_day10_e4_recovery.ipynb` dành riêng cho artifact cũ `run_all_v1`, không chạy cùng luồng chính. Chi tiết phạm vi tự lưu notebook, checkpoint và cách dùng artifact cũ nằm trong [COLAB_SETUP.md](COLAB_SETUP.md).

Trước khi nộp:

- [ ] Đối chiếu evidence, prediction thô, hình và log từ lượt chạy thật.
- [ ] Cập nhật `REPORT.md` bằng kết quả tương ứng; kiểm tra E4 và ghi rõ nếu không chạy E5.
- [ ] Chạy `python scripts/check_submission.py` và xử lý các FAIL.
- [ ] Không commit credential, `data/`, `work/`, cache hoặc trọng số.
- [ ] Giữ công cụ đề `scripts/`, `tests/`, `configs/` nguyên bản.
