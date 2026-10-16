# Chạy notebook trên Colab và lưu Google Drive

Notebook chính là `lab_day10.ipynb`. Bản này chạy tuần tự E0 → E1 → E2 → E3 → E4, với E4 thay đúng một biến là lượng dữ liệu train. Phần E5 cuối notebook chỉ kiểm tra hàm reward; không tự chạy GRPO. `run_lab.py` / `lab_runtime.py` là pipeline riêng đã có trong repo, không được notebook chính gọi.

## Cách chạy

1. Upload bản gọn **`work/lab_day10_colab.ipynb`** vào Colab, chọn **File → Save a copy in Drive** rồi làm việc trên bản sao đó. Bản gọn có cùng code với `lab_day10.ipynb`, bỏ output/widget cũ; notebook gốc vẫn được giữ lại.
2. Chọn **Runtime → Change runtime type → T4 GPU**, dùng runtime mới.
3. Trong **Secrets**, thêm `KAGGLE_API_TOKEN`, hoặc cặp `KAGGLE_USERNAME` và `KAGGLE_KEY`. Bật quyền **Notebook access**. `HF_TOKEN` là tùy chọn. Nếu dùng dataset local, đặt `DATASET_SOURCE` hoặc `DATASET_ZIP` ở cell cấu hình.
4. Để dùng đúng các file local vừa chuẩn bị mà không cần push GitHub, upload **`work/colab_source.zip`** vào thư mục gốc **My Drive**, giữ tên `colab_source.zip`. Notebook mở ở bước 1; ZIP cũng chứa bản notebook bỏ output để các kiểm tra của đề đọc được. Đường dẫn Colab tương ứng là `/content/drive/MyDrive/colab_source.zip`.
5. Chọn **Runtime → Run all** và chấp thuận kết nối Drive. Không cần đổi setup giữa E1/E2/E3. Khi không có ZIP, notebook tự clone fork `TrKhuyn/K4-Track-4-Day10-VisionLanguageModel`, nhánh `main`, để lấy công cụ đề, config và báo cáo.

Notebook đã nhúng mã hỗ trợ và danh sách thư viện nên không cần upload riêng từng file `.py`. `requirements-colab-run.txt` là danh sách tương ứng khi cài bằng tay; giữ torch/torchvision có sẵn của Colab. Thiết lập gỡ backend TorchAO không dùng trước khi import PEFT.

Muốn kiểm tra nhanh luồng chạy, đặt `RUN_MODE = 'smoke'`: E1–E4 chỉ train 2 bước và không tạo evidence test. Smoke vẫn tải model 4B và dataset thật. Sau đó đổi về `full` để chạy bài chính thức.

Trước E0, notebook kiểm tra giải mã tất cả ảnh và thử một update trên setup train nặng nhất với ảnh có đáp án dài nhất. Probe dùng model riêng rồi bỏ trọng số đó. Các lượt chính vẫn độc lập. Model được tải một lần theo commit cố định, thử lại lỗi mạng tạm thời khi tải, rồi các lần nạp tiếp chỉ dùng snapshot local. Inference thiếu VRAM sẽ tự giảm batch 4 → 2 → 1 và thử lại cùng ID; nếu batch 1 vẫn không đủ thì dừng, giữ tiến độ đã lưu.

## Những gì được tự lưu

| Nội dung | Khi lưu | Nơi lưu mặc định |
|---|---|---|
| Nội dung/code/output của `.ipynb` | Colab tự lưu khi làm việc trên bản notebook trong Drive; xem trạng thái lưu trên giao diện | File notebook bạn đã Save a copy in Drive |
| Checkpoint train | Mỗi **25 bước optimizer** hoặc **300 giây**, kiểm tra sau một update thành công; luôn lưu cuối lượt | `MyDrive/lab_day10_colab_runs/run_all_v3/work/<full hoặc smoke>/<run>/` |
| Log loss và tiến độ | Bước đầu, mỗi 10 bước hoặc khoảng 60 giây; kiểm tra sau update | `loss.jsonl`, `progress.json` trong thư mục run |
| Inference đang chạy | Sau batch đầu, mỗi khoảng 60 giây, và khi thoát hàm nếu Python vẫn xử lý được | `predictions/<mode>/*.progress.jsonl` và manifest tương ứng |
| Predictions hoàn tất, evidence, hình, báo cáo ZIP | Khi tạo xong từng artifact | Cùng thư mục Drive của RUN_LABEL |
| Trạng thái backup | Sau mỗi lần sao lưu thành công | `backup_status.json`, trường `last_success_utc` |

Đây là **lưu định kỳ**, không ghi toàn bộ trọng số theo từng giây. Mốc thời gian được kiểm tra giữa các update/batch; một update dài hoặc Drive chậm có thể làm khoảng cách thực dài hơn cấu hình. Nếu Colab bị thu hồi runtime đột ngột, phần công việc sau checkpoint/backup thành công cuối cùng có thể phải chạy lại. `backup_status.json` ghi nhận sao chép thành công qua Drive mount, không phải xác nhận đồng bộ ở phía máy chủ Google.

`SAVE_TO_DRIVE=True` bật sao lưu; `RESTORE_FROM_DRIVE=True` khôi phục artifact khi bắt đầu. Hai giá trị đều bật mặc định. Lỗi I/O tạm thời được thử lại tối đa 3 lần; lỗi hết dung lượng/quyền truy cập không bị che giấu. `drive.mount()` chỉ cấp quyền đọc/ghi Drive; bản thân nó không lưu notebook hay checkpoint. Mã Python trong notebook không tự ghi lại file `.ipynb` đang mở. Colab lưu notebook trong Drive, còn máy ảo thực thi có thể bị xóa khi hết phiên: [FAQ chính thức](https://research.google.com/colaboratory/faq.html).

## Tiếp tục sau khi ngắt phiên

Mở lại **cùng bản notebook**, giữ nguyên `RUN_LABEL`, model, cấu hình và mã hỗ trợ, rồi Run all trên runtime GPU mới. Revision của model được lưu và dùng lại; không tự lấy trọng số mới của nhánh `main` cho lượt cũ.

Checkpoint mới lưu LoRA, projector khi cần, processor, optimizer, scheduler, GradScaler, RNG Python/NumPy/Torch/CUDA, trạng thái sampler và vị trí batch trong epoch. File marker `checkpoint_complete.json` được ghi cuối; bản thiếu file, sai kích thước hoặc chưa có marker không được chọn để resume. Mặc định giữ một checkpoint bước gần nhất cùng với `final`, và chỉ dọn checkpoint cũ trong chính thư mục run đó sau khi bản mới đã sao lưu thành công.

Run đã hoàn tất được dùng lại. Inference tiếp tục từ prefix ID đã lưu nếu manifest khớp model/run/cấu hình. Khi đổi cấu hình hoặc mã, dùng `RUN_LABEL` mới để tránh trộn kết quả; fingerprint sẽ chặn dùng nhầm checkpoint.

E1, E2 và E3 kiểm tra lần lượt `report/evidence/e1.json`, `e2.json`, `e3.json`: thấy file là bỏ qua toàn bộ lượt đó. Ngay trước E4 có cell khôi phục checkpoint `final`, predictions và evidence còn thiếu. Thông thường giữ `RECOVERY_SOURCE_ROOT = None` để dùng artifact của chính `RUN_LABEL`; nếu cần lấy E1–E3 từ nhãn Drive cũ, đặt biến này thành `Path('/content/drive/MyDrive/lab_day10_colab_runs/<run_cu>')`. Cell không ghi đè artifact đang có và sẽ dừng nếu checkpoint cũ không khớp cấu hình hiện tại.

Mặc định sau review là **`run_all_v3`**, tách khỏi artifact cũ `run_all_v1`/`run_all_v2`. Code v3 giữ annotation suy biến bằng box tối thiểu 1 đơn vị, và chỉ tính loss trên JSON + EOS, bỏ newline sau EOS; không dùng lại checkpoint khác phiên bản mã. Checkpoint v1 chỉ có trọng số không thể resume optimizer. Notebook **`lab_day10_e4_recovery.ipynb`** vẫn giữ luồng E4 độc lập cho artifact legacy. Notebook chính đã có cell khôi phục trước E4 cho các run cùng cấu hình hiện tại.

## File hỗ trợ và kiểm tra

- `colab_storage.py`: sao lưu/khôi phục Drive, marker checkpoint và giới hạn thư mục dọn bản cũ.
- `colab_training.py`: mã cell train có resume.
- `colab_inference.py`: mã inference có tiếp tục và backup tiến độ.
- `colab_model.py`: tải snapshot có retry; tái sử dụng cache theo commit và nạp offline.
- `colab_preflight.py`: giải mã ảnh và thử forward/backward/optimizer trước E0.
- `prepare_colab.py`: đồng bộ các mã này vào notebook, kiểm tra cú pháp và tạo ZIP nguồn. Không chạy `scripts/make_notebook.py`, vì file đó sinh lại khung đề trống.

Sau khi chỉnh file hỗ trợ, chạy:

```bash
python prepare_colab.py --bundle work/colab_source.zip --clean-notebook work/lab_day10_colab.ipynb
python prepare_colab.py --check
python -m pytest -q tests student_checks -k "not test_report_template_has_student_markers_and_ten_questions"
```

Một test gốc kiểm tra `REPORT.md` vẫn còn chuỗi `<điền` của mẫu báo cáo. Báo cáo hiện đã điền nên test đó không còn phù hợp cho bước setup và được loại riêng; không sửa `tests/`, `scripts/`, `configs/` hay báo cáo của bạn. Các kiểm tra công cụ còn lại vẫn chạy. ZIP nguồn chỉ đóng gói file code/config/tài liệu theo danh sách cho phép; không chứa `.env`, dataset, trọng số hay HF cache.

Đã kiểm tra cục bộ bằng model nhỏ trên CPU: ngắt ở checkpoint rồi khôi phục từ thư mục mô phỏng Drive cho trọng số LoRA và chuỗi loss khớp chạy liền mạch; kiểm tra inference tiếp tục, backup lỗi không tạo marker hoàn tất, và khôi phục bỏ qua checkpoint thiếu. Qwen nhỏ dùng processor thật đã qua forward/backward E1/E2/E3 và so sánh logits khớp tuyệt đối sau lưu–nạp LoRA/projector. Luồng cached E4 → scorer → hình → ZIP đã chạy trong thư mục dữ liệu tổng hợp riêng; các số thử không được ghi vào evidence của repo. Máy cục bộ không có CUDA; chưa thực thi toàn bộ Qwen3-VL-4B/T4 hay xác thực Drive thật trong phiên này. Run all vẫn cần GPU, credential hợp lệ, kết nối mạng và dung lượng Drive/đĩa đủ cho checkpoint cùng optimizer.
