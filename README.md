# Hệ thống Nhận diện và Trích xuất Thông tin CCCD

Đồ án: Trích xuất thông tin Căn cước công dân (CCCD) sử dụng mạng nơ-ron tích chập.

Hệ thống được thiết kế để tự động định vị thẻ CCCD, căn chỉnh góc xoay thông qua mã QR, và trích xuất các trường thông tin quan trọng (Số CCCD, Họ tên, Ngày sinh, v.v.) bằng sự kết hợp giữa mô hình YOLO và VietOCR.

---

## 🌐 Trải nghiệm Demo Trực tuyến (Live Web)

Nhóm đã triển khai (deploy) thành công mô hình thực tế lên nền tảng web thông qua giao diện Gradio và Cloudflare Tunnel. 
Bạn có thể truy cập để upload ảnh và test trực tiếp hệ thống tại đường link:
👉 **[https://cccd.nvhoa.xyz](https://cccd.nvhoa.xyz)**

*(Lưu ý: Web được host trực tiếp trên máy cá nhân cục bộ của nhóm, do đó đường link chỉ hoạt động khi máy chủ nội bộ đang được bật).*

---

## 💻 Hướng dẫn chạy Demo Local (Trên máy tính cá nhân)

Toàn bộ code giao diện và luồng xử lý chính được đặt trong thư mục `demo/`. Để tự chạy trên máy của bạn, thực hiện các bước sau:

**Bước 1: Cài đặt thư viện**
```bash
cd demo
pip install -r requirements.txt
```

**Bước 2: Chạy ứng dụng**
Khởi chạy giao diện chính (có tích hợp đầy đủ các bước hậu xử lý làm sạch dữ liệu):
```bash
python app.py
```
*Truy cập trình duyệt tại địa chỉ `http://127.0.0.1:7860` để sử dụng.*

**Bản so sánh (Tùy chọn):**
Nếu bạn muốn xem kết quả trích xuất **OCR thô (không có hậu xử lý)** để đối chiếu sự khác biệt, hãy chạy file `app_raw.py`:
```bash
python app_raw.py
```

---

## 📁 Cấu trúc thư mục (Pipeline Huấn luyện)

Toàn bộ mã nguồn nghiên cứu và huấn luyện mô hình được lưu trong thư mục `notebooks/`, tổ chức theo đúng trình tự phát triển dự án:

- **01_data_classification.ipynb**: Phân loại và chuẩn bị dữ liệu hình ảnh ban đầu.
- **02_data_preprocessing.ipynb**: Tiền xử lý dữ liệu trước khi đưa vào mô hình.
- **03_yolo_detection_training.ipynb**: Huấn luyện mô hình YOLO (phiên bản v8/v11) để phát hiện vùng chứa CCCD và cắt các trường thông tin.
- **04_paddleocr_training.ipynb**: Huấn luyện và tinh chỉnh mô hình PaddleOCR để nhận diện chữ trên thẻ.
- **05_vietocr_training.ipynb**: Huấn luyện mô hình VietOCR chuyên dụng đọc văn bản tiếng Việt.

---

## 💾 Nguồn dữ liệu (Datasets)

Do giới hạn về dung lượng, dữ liệu thực tế không được đính kèm trong thư mục code. Các bộ dữ liệu (dataset) đã được nhóm sử dụng được lưu trữ công khai tại các liên kết sau:

### 1. Dữ liệu huấn luyện Phát hiện vùng (YOLO)
- [Roboflow - card-dstr0](https://universe.roboflow.com/nguyn-khoa-ng/card-dstr0)
- [Roboflow - card-detection-1cawm](https://universe.roboflow.com/soicodoc/card-detection-1cawm)

### 2. Dữ liệu huấn luyện Nhận diện chữ (OCR)
- [Kaggle - VietOCR Dataset (sử dụng thư mục meta)](https://www.kaggle.com/datasets/vulamnguyen/vietocr-dataset)
