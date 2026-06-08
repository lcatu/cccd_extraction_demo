---
title: Demo CCCD Extraction
emoji: 🪪
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.16.0"
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# 🪪 Nhận dạng và Trích xuất thông tin CCCD

Ứng dụng demo trích xuất thông tin từ ảnh thẻ **Căn cước Công dân** (CCCD) Việt Nam sử dụng:

- **YOLOv11** — Phát hiện vùng thông tin trên thẻ
- **VietOCR** — Nhận dạng ký tự tiếng Việt
- **QRDet** — Tự động xoay ảnh bị nghiêng/lật

## Cách dùng
1. Upload ảnh mặt trước thẻ CCCD
2. Bấm **Trích xuất thông tin**
3. Kết quả sẽ hiển thị bên phải

> ⚠️ Chỉ hỗ trợ ảnh mặt trước CCCD Việt Nam.
